#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitmamba.h"

/* --- GPT-2 Byte Encoder/Decoder ---
 *
 * GPT-2 maps each byte value to a unique unicode codepoint so that the
 * vocabulary contains only valid unicode strings. Printable bytes (33-126,
 * 161-172, 174-255) map to themselves; other bytes map to 256+.
 */
static int byte_to_unicode[256];
static int unicode_to_byte[512]; /* max codepoint ~320, but pad to 512 */
static bool byte_encoder_ready;

static void init_byte_encoder(void)
{
    if (byte_encoder_ready)
        return;

    memset(byte_to_unicode, 0xFF, sizeof(byte_to_unicode)); /* -1 */
    memset(unicode_to_byte, 0xFF, sizeof(unicode_to_byte));

    /* Printable ASCII: 33-126 */
    for (int i = 33; i <= 126; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    /* Extended Latin: 161-172 */
    for (int i = 161; i <= 172; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    /* Extended Latin: 174-255 */
    for (int i = 174; i <= 255; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }

    /* Remaining bytes → 256+ range */
    int offset = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode[i] < 0) {
            byte_to_unicode[i] = offset;
            unicode_to_byte[offset] = i;
            offset++;
        }
    }
    byte_encoder_ready = true;
}

/* Encode one byte to its GPT-2 UTF-8 representation. Returns bytes written. */
static int encode_byte_to_utf8(unsigned char b, char *out)
{
    int cp = byte_to_unicode[b];
    if (cp < 128) {
        out[0] = (char) cp;
        return 1;
    }
    /* 2-byte UTF-8 (codepoints 128-2047, covers all GPT-2 mappings) */
    out[0] = (char) (0xC0 | (cp >> 6));
    out[1] = (char) (0x80 | (cp & 0x3F));
    return 2;
}

/* Convert raw text to GPT-2 byte-encoded UTF-8 string */
static char *text_to_byte_encoding(const char *text, int len)
{
    init_byte_encoder();
    char *result = xmalloc((size_t) len * 2 + 1);
    int j = 0;
    for (int i = 0; i < len; i++)
        j += encode_byte_to_utf8((unsigned char) text[i], result + j);
    result[j] = '\0';
    return result;
}

/* Decode UTF-8 codepoint, return bytes consumed */
static int decode_utf8_codepoint(const char *s, int remaining, int *cp)
{
    unsigned char c = (unsigned char) s[0];
    if ((c & 0x80) == 0) {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && remaining >= 2) {
        *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && remaining >= 3) {
        *cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    *cp = c;
    return 1;
}

/* Decode GPT-2 byte-encoded token string to raw bytes */
static char *decode_token_to_bytes(const char *token, int *out_len)
{
    init_byte_encoder();
    int token_len = (int) strlen(token);
    char *result = xmalloc(token_len + 1);
    int j = 0;

    for (int i = 0; i < token_len;) {
        int cp;
        int consumed = decode_utf8_codepoint(token + i, token_len - i, &cp);
        if (cp >= 0 && cp < 512 && unicode_to_byte[cp] >= 0)
            result[j++] = (char) unicode_to_byte[cp];
        else
            result[j++] = (char) cp;
        i += consumed;
    }
    result[j] = '\0';
    *out_len = j;
    return result;
}

/* FNV-1a hash */
static uint32_t fnv1a(const char *s, int len)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char) s[i];
        h *= 16777619u;
    }
    return h;
}

/*
 * GPT-2 regex word splitter.
 * Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+
 */
static int match_gpt2_word(const char *s, int len)
{
    if (len <= 0)
        return 0;

    /* Contractions */
    if (s[0] == '\'') {
        if (len >= 2) {
            char c = s[1];
            if (c == 's' || c == 't' || c == 'm' || c == 'd')
                return 2;
        }
        if (len >= 3) {
            if (s[1] == 'r' && s[2] == 'e')
                return 3;
            if (s[1] == 'v' && s[2] == 'e')
                return 3;
            if (s[1] == 'l' && s[2] == 'l')
                return 3;
        }
    }

    /* Optional space + alpha+ */
    {
        int i = (s[0] == ' ') ? 1 : 0;
        if (i < len && isalpha((unsigned char) s[i])) {
            while (i < len && isalpha((unsigned char) s[i]))
                i++;
            return i;
        }
    }

    /* Optional space + digit+ */
    {
        int i = (s[0] == ' ') ? 1 : 0;
        if (i < len && isdigit((unsigned char) s[i])) {
            while (i < len && isdigit((unsigned char) s[i]))
                i++;
            return i;
        }
    }

    /* Optional space + non-ws/non-alpha/non-digit+ */
    {
        int i = (s[0] == ' ') ? 1 : 0;
        if (i < len) {
            unsigned char c = (unsigned char) s[i];
            if (!isspace(c) && !isalpha(c) && !isdigit(c)) {
                while (i < len) {
                    c = (unsigned char) s[i];
                    if (isspace(c) || isalpha(c) || isdigit(c))
                        break;
                    i++;
                }
                return i;
            }
        }
    }

    /* Whitespace+ */
    if (isspace((unsigned char) s[0])) {
        int i = 1;
        while (i < len && isspace((unsigned char) s[i]))
            i++;
        return i;
    }

    return 1;
}

/* Hash table lookup: returns token_id or -1 */
static int tokenizer_lookup(const gpt2_tokenizer_t *tok, const char *s, int len)
{
    uint32_t h = fnv1a(s, len) & TOK_HT_MASK;
    for (;;) {
        int id = tok->ht_ids[h];
        if (id < 0)
            return -1;
        if (tok->token_lengths[id] == len &&
            memcmp(tok->id_to_token[id], s, len) == 0)
            return id;
        h = (h + 1) & TOK_HT_MASK;
    }
}

bool tokenizer_load(gpt2_tokenizer_t *tok, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: tokenizer.bin not found: %s\n", path);
        return false;
    }

    tok->vocab_size = VOCAB_SIZE;
    tok->id_to_token = xcalloc(VOCAB_SIZE, sizeof(char *));
    tok->token_lengths = xmalloc(VOCAB_SIZE * sizeof(int));
    tok->ht_ids = xmalloc(TOK_HT_SIZE * sizeof(int));
    memset(tok->ht_ids, -1, TOK_HT_SIZE * sizeof(int));

    for (int i = 0; i < VOCAB_SIZE; i++) {
        uint32_t len;
        if (fread(&len, sizeof(uint32_t), 1, fp) != 1)
            goto fail;
        if (len > MAX_TOKEN_LENGTH)
            goto fail;

        tok->id_to_token[i] = xmalloc(len + 1);
        tok->token_lengths[i] = (int) len;
        if (fread(tok->id_to_token[i], 1, len, fp) != len)
            goto fail;
        tok->id_to_token[i][len] = '\0';

        /* Insert into hash table */
        uint32_t h = fnv1a(tok->id_to_token[i], (int) len) & TOK_HT_MASK;
        while (tok->ht_ids[h] >= 0)
            h = (h + 1) & TOK_HT_MASK;
        tok->ht_ids[h] = i;
    }
    fclose(fp);

    /* Auto-detect format: token 220 should represent a single space.
     * Byte-encoded format: 2 bytes (0xC4, 0xA0) = "Ġ"
     * Raw text format: 1 byte (0x20) = " " */
    tok->is_byte_encoded = (tok->token_lengths[220] == 2 &&
                            (unsigned char) tok->id_to_token[220][0] == 0xC4);

    init_byte_encoder();
    return true;

fail:
    fprintf(stderr, "Error: Failed to load tokenizer\n");
    fclose(fp);
    tokenizer_free(tok);
    memset(tok, 0, sizeof(*tok));
    return false;
}

int tokenizer_encode(const gpt2_tokenizer_t *tok,
                     const char *text,
                     int32_t *tokens,
                     int max_tokens)
{
    int n_tokens = 0;
    int text_len = (int) strlen(text);
    int pos = 0;

    while (pos < text_len && n_tokens < max_tokens) {
        int word_len = match_gpt2_word(text + pos, text_len - pos);
        if (word_len <= 0) {
            pos++;
            continue;
        }

        /* Convert word to the format used in tokenizer.bin */
        const char *lookup_str;
        int lookup_len;
        char *encoded = NULL;

        if (tok->is_byte_encoded) {
            encoded = text_to_byte_encoding(text + pos, word_len);
            lookup_str = encoded;
            lookup_len = (int) strlen(encoded);
        } else {
            lookup_str = text + pos;
            lookup_len = word_len;
        }

        /* Greedy longest-token matching */
        int i = 0;
        while (i < lookup_len && n_tokens < max_tokens) {
            int best_len = 0, best_id = -1;

            for (int j = lookup_len; j > i; j--) {
                int id = tokenizer_lookup(tok, lookup_str + i, j - i);
                if (id >= 0) {
                    best_len = j - i;
                    best_id = id;
                    break;
                }
            }

            if (best_id >= 0) {
                tokens[n_tokens++] = best_id;
                i += best_len;
            } else {
                i++;
            }
        }

        free(encoded);
        pos += word_len;
    }

    return n_tokens;
}

const char *tokenizer_decode(const gpt2_tokenizer_t *tok, int id)
{
    if (id < 0 || id >= tok->vocab_size)
        return "";
    return tok->id_to_token[id];
}

/*
 * Decode a token to raw output bytes. For byte-encoded tokenizer.bin,
 * converts GPT-2 byte encoding back to raw bytes (e.g., "Ġ" → space).
 * For raw-text format, returns the token string directly.
 *
 * Caller must free the returned buffer.
 */
char *tokenizer_decode_to_bytes(const gpt2_tokenizer_t *tok,
                                int id,
                                int *out_len)
{
    const char *s = tokenizer_decode(tok, id);
    if (!tok->is_byte_encoded) {
        int len = (int) strlen(s);
        char *copy = xmalloc(len + 1);
        memcpy(copy, s, len + 1);
        *out_len = len;
        return copy;
    }
    return decode_token_to_bytes(s, out_len);
}

void tokenizer_free(gpt2_tokenizer_t *tok)
{
    if (tok->id_to_token) {
        for (int i = 0; i < tok->vocab_size; i++)
            free(tok->id_to_token[i]);
        free(tok->id_to_token);
    }
    free(tok->token_lengths);
    free(tok->ht_ids);
}
