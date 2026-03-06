/*
 * BitMamba Chat Mode - Interactive AI Assistant
 *
 * - Pipe-friendly: detect tty vs piped stdin, adjust behavior
 * - State persistence: save/load SSM state to skip system prompt prefill
 * - Streaming output: flush tokens as they are generated
 * - Clean separation: stderr for diagnostics, stdout for model output
 */

#define _GNU_SOURCE

#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "chat.h"
#include "linenoise.h"

#define HISTORY_MAX 256
#define MAX_PROMPT_TOKENS 2048

/* Strip leading whitespace and common Q&A prefixes from input
 * so the template doesn't double-wrap. Returns pointer into the same string.
 */
static const char *strip_qa_prefix(const char *s)
{
    while (*s == ' ' || *s == '\t')
        s++;
    /* "Question: " or "question: " */
    if (!strncasecmp(s, "question:", 9)) {
        s += 9;
        while (*s == ' ')
            s++;
    } else if ((s[0] == 'Q' || s[0] == 'q') && s[1] == ':') {
        s += 2;
        while (*s == ' ')
            s++;
    }
    return s;
}

/* Rewrite vague questions into forms that produce better base-model
 * completions. Base models are document continuers — short ambiguous queries
 * lack context. Expanding into recognizable FAQ/encyclopedia patterns improves
 * quality.
 *
 * Returns malloc'd string, or NULL if no rewrite needed (caller uses original).
 */
static char *rewrite_query(const char *s)
{
    size_t len = strlen(s);
    /* Strip trailing whitespace for matching */
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t'))
        len--;

    /* Only rewrite short questions ending with '?' */
    if (len < 5 || s[len - 1] != '?')
        return NULL;

/* Helper: replace trailing '?' with new text */
#define INSERT_BEFORE_QMARK(word)                   \
    do {                                            \
        const char *_w = (word);                    \
        size_t _wlen = strlen(_w);                  \
        size_t end = len - 1;                       \
        while (end > 0 && s[end - 1] == ' ')        \
            end--;                                  \
        /* Skip separator space if word starts */   \
        /* with punctuation (comma, space) */       \
        int _sep = (_w[0] != ',' && _w[0] != ' ');  \
        char *out = malloc(end + _sep + _wlen + 1); \
        if (!out)                                   \
            return NULL;                            \
        memcpy(out, s, end);                        \
        if (_sep)                                   \
            out[end] = ' ';                         \
        memcpy(out + end + _sep, _w, _wlen);        \
        out[end + _sep + _wlen] = '\0';             \
        return out;                                 \
    } while (0)

    /* "Where is X?" -> "Where is X located, and what is it known for?" */
    if (!strncasecmp(s, "where is ", 9) || !strncasecmp(s, "where's ", 8)) {
        if (!strcasestr(s, "located") && !strcasestr(s, "found") &&
            !strcasestr(s, "known for"))
            INSERT_BEFORE_QMARK("located, and what is it known for?");
    }

    /* "Who is X?" -> "Who is X, and what are they known for?"
     * Skip self-referential ("who are you", "who am I") — not biographical. */
    if (!strcasestr(s, " you") && !strcasestr(s, " I ") &&
        !strcasestr(s, " me")) {
        if (!strncasecmp(s, "who is ", 7) || !strncasecmp(s, "who's ", 6)) {
            if (!strcasestr(s, "known for") && !strcasestr(s, "famous"))
                INSERT_BEFORE_QMARK(", and what are they known for?");
        }
        if (!strncasecmp(s, "who was ", 8)) {
            if (!strcasestr(s, "known for") && !strcasestr(s, "famous"))
                INSERT_BEFORE_QMARK(", and what were they known for?");
        }
    }

    /* "What is X?" -> "What is X, and how does it work?" */
    if (!strncasecmp(s, "what is ", 8) || !strncasecmp(s, "what's ", 7)) {
        if (!strcasestr(s, "work") && !strcasestr(s, "mean") &&
            !strcasestr(s, "explain"))
            INSERT_BEFORE_QMARK(", and how does it work?");
    }

    /* "What are X?" -> "What are X, and why are they important?" */
    if (!strncasecmp(s, "what are ", 9)) {
        if (!strcasestr(s, "important") && !strcasestr(s, "used"))
            INSERT_BEFORE_QMARK(", and why are they important?");
    }

    /* "How does X?" / "How do X?" -> expand for detail */
    if (!strncasecmp(s, "how does ", 9) || !strncasecmp(s, "how do ", 7)) {
        if (!strcasestr(s, "explain") && !strcasestr(s, "detail"))
            INSERT_BEFORE_QMARK(", and can you explain in detail?");
    }

    /* "Why is X?" / "Why are X?" / "Why do X?" -> expand */
    if (!strncasecmp(s, "why ", 4)) {
        if (!strcasestr(s, "reason") && !strcasestr(s, "because"))
            INSERT_BEFORE_QMARK(", and what are the main reasons?");
    }

    /* "When was X?" / "When did X?" -> add context */
    if (!strncasecmp(s, "when ", 5)) {
        if (!strcasestr(s, "context") && !strcasestr(s, "detail"))
            INSERT_BEFORE_QMARK(", and what was the context?");
    }

    /* "Can you X?" / "Could you X?" -> drop hedging, make direct */
    if (!strncasecmp(s, "can you ", 8))
        INSERT_BEFORE_QMARK("in detail?");
    if (!strncasecmp(s, "could you ", 10))
        INSERT_BEFORE_QMARK("in detail?");

    /* "Which X?" -> expand for explanation */
    if (!strncasecmp(s, "which ", 6)) {
        if (!strcasestr(s, "explain") && !strcasestr(s, "detail"))
            INSERT_BEFORE_QMARK(", and why?");
    }

    /* "Tell me about X" (no ?, just imperative) is handled elsewhere,
     * but "Is X?" questions benefit from expansion */
    if (!strncasecmp(s, "is ", 3) || !strncasecmp(s, "are ", 4) ||
        !strncasecmp(s, "does ", 5) || !strncasecmp(s, "do ", 3)) {
        if (!strcasestr(s, "explain") && !strcasestr(s, "why"))
            INSERT_BEFORE_QMARK(", and why or why not?");
    }

#undef INSERT_BEFORE_QMARK
    return NULL;
}

/* Apply prompt template to user input via manual splice (no printf formatting,
 * so %n/%x/etc. in user-supplied --template can't cause UB).
 * Returns malloc'd string. If no template, returns strdup of input.
 */
static char *apply_template(const char *input, const char *tpl)
{
    if (!tpl)
        return strdup(input);

    /* Find the single %s placeholder */
    const char *marker = strstr(tpl, "%s");
    if (!marker) {
        fprintf(stderr, "[chat] Warning: template has no %%s placeholder\n");
        return strdup(input);
    }
    if (strstr(marker + 2, "%s")) {
        fprintf(stderr, "[chat] Warning: template has multiple %%s\n");
        return strdup(input);
    }

    /* Strip Q: prefix to avoid "Q: Q: ..." double-wrap */
    input = strip_qa_prefix(input);

    /* Manual splice: prefix + input + suffix */
    size_t prefix_len = (size_t) (marker - tpl);
    size_t in_len = strlen(input);
    size_t suffix_len = strlen(marker + 2);
    size_t out_len = prefix_len + in_len + suffix_len + 1;
    char *out = malloc(out_len);
    if (!out)
        return strdup(input);
    memcpy(out, tpl, prefix_len);
    memcpy(out + prefix_len, input, in_len);
    memcpy(out + prefix_len + in_len, marker + 2, suffix_len);
    out[prefix_len + in_len + suffix_len] = '\0';
    return out;
}

/* --- Arithmetic Expression Detection and Evaluation ---
 *
 * Tokenize input, recognize number words / operator words / digits,
 * skip leading noise words ("calculate", "what is", typos),
 * build a pure arithmetic string, and pipe to bc.
 */

/* Max tokens in a math expression (words separated by spaces) */
#define MATH_MAX_TOKENS 32

/* Lookup: English unit words (0-19) -> value, or -1 */
static int word_unit(const char *w)
{
    static const struct {
        const char *w;
        int v;
    } tbl[] = {
        {"zero", 0},     {"one", 1},        {"two", 2},       {"three", 3},
        {"four", 4},     {"five", 5},       {"six", 6},       {"seven", 7},
        {"eight", 8},    {"nine", 9},       {"ten", 10},      {"eleven", 11},
        {"twelve", 12},  {"thirteen", 13},  {"fourteen", 14}, {"fifteen", 15},
        {"sixteen", 16}, {"seventeen", 17}, {"eighteen", 18}, {"nineteen", 19},
    };
    for (size_t i = 0; i < sizeof(tbl) / sizeof(tbl[0]); i++)
        if (!strcmp(w, tbl[i].w))
            return tbl[i].v;
    return -1;
}

/* Lookup: English tens words (20-90) -> value, or -1 */
static int word_tens(const char *w)
{
    static const struct {
        const char *w;
        int v;
    } tbl[] = {
        {"twenty", 20}, {"thirty", 30},  {"forty", 40},  {"fifty", 50},
        {"sixty", 60},  {"seventy", 70}, {"eighty", 80}, {"ninety", 90},
    };
    for (size_t i = 0; i < sizeof(tbl) / sizeof(tbl[0]); i++)
        if (!strcmp(w, tbl[i].w))
            return tbl[i].v;
    return -1;
}

/* Parse a number-word phrase from tokens[*pos..].
 * "twenty three" -> 23, "one hundred and five" -> 105.
 * Advances *pos past consumed tokens. Returns true if any consumed.
 */
static bool parse_number_phrase(char **tok, int ntok, int *pos, long *out)
{
    long total = 0, cur = 0;
    int j = *pos, consumed = 0;

    while (j < ntok) {
        int u = word_unit(tok[j]);
        if (u >= 0) {
            cur += u;
            j++;
            consumed++;
            continue;
        }
        int t = word_tens(tok[j]);
        if (t >= 0) {
            cur += t;
            j++;
            consumed++;
            continue;
        }
        if (!strcmp(tok[j], "hundred")) {
            if (cur == 0)
                cur = 1;
            cur *= 100;
            j++;
            consumed++;
            continue;
        }
        if (!strcmp(tok[j], "thousand")) {
            if (cur == 0)
                cur = 1;
            total += cur * 1000;
            cur = 0;
            j++;
            consumed++;
            continue;
        }
        /* "and" as connector: "one hundred and two" (only after hundred /
         * thousand)
         */
        if (!strcmp(tok[j], "and") && consumed > 0 && cur >= 100 &&
            j + 1 < ntok) {
            j++;
            consumed++;
            continue;
        }
        break;
    }
    if (!consumed)
        return false;
    *out = total + cur;
    *pos = j;
    return true;
}

/* Try matching an operator word at tokens[pos].
 * Returns operator string (+,-,*,/) or NULL.
 * Sets *advance to number of tokens consumed (1 or 2).
 */
static const char *match_op_word(char **tok, int ntok, int pos, int *advance)
{
    *advance = 1;
    const char *w = tok[pos];
    if (!strcmp(w, "plus") || !strcmp(w, "add"))
        return "+";
    if (!strcmp(w, "minus") || !strcmp(w, "subtract"))
        return "-";
    if (!strcmp(w, "times"))
        return "*";
    if (!strcmp(w, "over"))
        return "/";
    if (!strcmp(w, "multiplied") && pos + 1 < ntok &&
        !strcmp(tok[pos + 1], "by")) {
        *advance = 2;
        return "*";
    }
    if (!strcmp(w, "divided") && pos + 1 < ntok &&
        !strcmp(tok[pos + 1], "by")) {
        *advance = 2;
        return "/";
    }
    return NULL;
}

/* Check if a word is a known "noise" word that precedes a math expression.
 * Catches: "calculate", "compute", "what", "is", "how", "much", "please",
 * "the", "of", "tell", "me", "can", "you", "could", "solve", "evaluate",
 * and any unrecognized word that appears before the first math token.
 */
static bool is_noise_word(const char *w)
{
    static const char *noise[] = {
        "calculate", "compute",   "evaluate", "solve",  "what",  "what's",
        "is",        "how",       "much",     "please", "the",   "of",
        "tell",      "me",        "can",      "you",    "could", "find",
        "get",       "determine", "whats",
    };
    for (size_t i = 0; i < sizeof(noise) / sizeof(noise[0]); i++)
        if (!strcmp(w, noise[i]))
            return true;
    return false;
}

/* Check if token is a pure numeric literal (digits, dots) */
static bool is_digit_token(const char *w)
{
    if (!*w)
        return false;
    bool has_digit = false;
    for (const char *p = w; *p; p++) {
        if (*p >= '0' && *p <= '9')
            has_digit = true;
        else if (*p != '.')
            return false;
    }
    return has_digit;
}

/* Check if token is an arithmetic operator character */
static bool is_op_char_token(const char *w)
{
    return (w[0] && !w[1] &&
            (w[0] == '+' || w[0] == '-' || w[0] == '*' || w[0] == '/' ||
             w[0] == '^' || w[0] == '%'));
}

/* "sum/difference/product/quotient of A and B" pattern detection.
 * Returns operator char or 0 if not matched.
 */
static char match_aggregate_word(const char *w)
{
    if (!strcmp(w, "sum"))
        return '+';
    if (!strcmp(w, "difference"))
        return '-';
    if (!strcmp(w, "product"))
        return '*';
    if (!strcmp(w, "quotient"))
        return '/';
    return 0;
}

/* Tokenize-and-build: convert natural math input into a bc expression.
 *
 * Pipeline:
 * 1. Normalize input: lowercase, strip punctuation (?,!,=,:,;) to spaces
 * 2. Tokenize by whitespace
 * 3. Skip leading noise words (any word that isn't a number/operator/digit)
 * 4. Consume number-word phrases, operator words, digit literals, parens
 * 5. Build expression string for bc
 *
 * Returns malloc'd expression or NULL if input isn't arithmetic.
 */
static char *extract_math_expr(const char *input)
{
    /* Strip Q&A prefix */
    const char *s = strip_qa_prefix(input);
    size_t slen = strlen(s);
    if (slen == 0 || slen > 256)
        return NULL;

    /* Normalize: lowercase, replace punctuation with space,
     * pad operators with spaces so "2+3" tokenizes as "2 + 3" */
    char *norm = malloc(slen * 3 + 1); /* worst case: every char padded */
    if (!norm)
        return NULL;
    size_t ni = 0;
    for (size_t i = 0; i < slen; i++) {
        char c = s[i];
        if (c == '?' || c == '!' || c == '=' || c == ':' || c == ';' ||
            c == ',') {
            norm[ni++] = ' ';
        } else if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' ||
                   c == '%') {
            norm[ni++] = ' ';
            norm[ni++] = c;
            norm[ni++] = ' ';
        } else if (c == '(' || c == ')') {
            norm[ni++] = ' ';
            norm[ni++] = c;
            norm[ni++] = ' ';
        } else {
            norm[ni++] = (char) tolower((unsigned char) c);
        }
    }
    norm[ni] = '\0';

    /* Tokenize by whitespace */
    char *tokens[MATH_MAX_TOKENS];
    int ntok = 0;
    char *saveptr;
    char *tok = strtok_r(norm, " \t", &saveptr);
    while (tok && ntok < MATH_MAX_TOKENS) {
        tokens[ntok++] = tok;
        tok = strtok_r(NULL, " \t", &saveptr);
    }
    if (ntok == 0) {
        free(norm);
        return NULL;
    }
    /* Reject if tokenization was truncated (more tokens exist).
     * After the loop, tok is non-NULL if strtok_r found another token
     * but we hit MATH_MAX_TOKENS before consuming it. */
    if (ntok == MATH_MAX_TOKENS && strtok_r(NULL, " \t", &saveptr) != NULL) {
        free(norm);
        return NULL;
    }

    /* Build expression */
    char expr[512];
    size_t elen = 0;
    bool has_num = false, has_op = false;
    int i = 0;

    /* Skip leading noise words */
    while (i < ntok && is_noise_word(tokens[i]))
        i++;

    /* Check for "sum/difference/product/quotient of A and B" pattern */
    char agg_op = 0;
    if (i < ntok) {
        agg_op = match_aggregate_word(tokens[i]);
        if (agg_op) {
            i++; /* skip "sum" etc. */
            if (i < ntok && !strcmp(tokens[i], "of"))
                i++; /* skip "of" */
        }
    }

    /* If still at start and first token is unrecognized, it might be
     * a typo ("Calcuate") -- skip any remaining non-math leading words */
    while (i < ntok && !is_digit_token(tokens[i]) && word_unit(tokens[i]) < 0 &&
           word_tens(tokens[i]) < 0 && !is_op_char_token(tokens[i]) &&
           tokens[i][0] != '(' && !match_aggregate_word(tokens[i]))
        i++;

    /* Remaining capacity macro: ensures we never overflow expr[512] */
#define EXPR_REMAINING (sizeof(expr) - 1 - elen)
#define EXPR_CHECK(need)               \
    do {                               \
        if ((need) > EXPR_REMAINING) { \
            free(norm);                \
            return NULL;               \
        }                              \
    } while (0)

    while (i < ntok && elen < sizeof(expr) - 32) {
        /* Try number-word phrase ("twenty three" -> 23) */
        long numval;
        if (parse_number_phrase(tokens, ntok, &i, &numval)) {
            int n = snprintf(expr + elen, sizeof(expr) - elen, "%ld", numval);
            if (n < 0 || (size_t) n >= sizeof(expr) - elen) {
                free(norm);
                return NULL;
            }
            elen += n;
            has_num = true;
            /* If aggregate pattern, inject operator before second operand */
            if (agg_op && has_num && !has_op) {
                /* skip "and" between operands */
                if (i < ntok && !strcmp(tokens[i], "and"))
                    i++;
                EXPR_CHECK(3);
                expr[elen++] = ' ';
                expr[elen++] = agg_op;
                expr[elen++] = ' ';
                has_op = true;
            }
            continue;
        }

        /* Try digit literal */
        if (is_digit_token(tokens[i])) {
            size_t tlen = strlen(tokens[i]);
            EXPR_CHECK(tlen);
            memcpy(expr + elen, tokens[i], tlen);
            elen += tlen;
            has_num = true;
            i++;
            /* Aggregate "and" separator */
            if (agg_op && !has_op && i < ntok && !strcmp(tokens[i], "and")) {
                i++;
                EXPR_CHECK(3);
                expr[elen++] = ' ';
                expr[elen++] = agg_op;
                expr[elen++] = ' ';
                has_op = true;
            }
            continue;
        }

        /* Try operator word ("plus", "minus", "times", "divided by") */
        int adv;
        const char *op = match_op_word(tokens, ntok, i, &adv);
        if (op) {
            size_t oplen = strlen(op);
            EXPR_CHECK(oplen + 2);
            expr[elen++] = ' ';
            memcpy(expr + elen, op, oplen);
            elen += oplen;
            expr[elen++] = ' ';
            has_op = true;
            i += adv;
            continue;
        }

        /* Operator character token (+, -, *, /, ^, %) */
        if (is_op_char_token(tokens[i])) {
            EXPR_CHECK(3);
            expr[elen++] = ' ';
            expr[elen++] = tokens[i][0];
            expr[elen++] = ' ';
            has_op = true;
            i++;
            continue;
        }

        /* Parentheses */
        if (tokens[i][0] == '(' || tokens[i][0] == ')') {
            EXPR_CHECK(1);
            expr[elen++] = tokens[i][0];
            i++;
            continue;
        }

        /* "and" as implicit + between numbers: "10 and 20" */
        if (!strcmp(tokens[i], "and")) {
            EXPR_CHECK(3);
            if (agg_op && !has_op) {
                expr[elen++] = ' ';
                expr[elen++] = agg_op;
                expr[elen++] = ' ';
                has_op = true;
            } else if (has_num) {
                expr[elen++] = ' ';
                expr[elen++] = '+';
                expr[elen++] = ' ';
                has_op = true;
            }
            i++;
            continue;
        }

        /* Unknown token: not arithmetic */
        free(norm);
        return NULL;
    }
#undef EXPR_CHECK
#undef EXPR_REMAINING

    free(norm);
    expr[elen] = '\0';

    if (!has_num || !has_op)
        return NULL;

    /* Final safety check: only digits, ops, parens, dots, spaces */
    for (size_t j = 0; j < elen; j++) {
        char c = expr[j];
        if (!((c >= '0' && c <= '9') || c == '.' || c == '+' || c == '-' ||
              c == '*' || c == '/' || c == '^' || c == '%' || c == '(' ||
              c == ')' || c == ' '))
            return NULL;
    }

    return strdup(expr);
}

/* In-process arithmetic evaluator (no shell, no popen).
 * Recursive descent: expr = term ((+|-) term)*
 *                    term = factor ((*|/|%) factor)*
 *                    factor = base (^ factor)?
 *                    base = [-]? ( '(' expr ')' | number )
 */
typedef struct {
    const char *s;
    size_t pos;
    bool err;
} eval_ctx_t;

static void eval_skip_ws(eval_ctx_t *ctx)
{
    while (ctx->s[ctx->pos] == ' ')
        ctx->pos++;
}

static double eval_expr(eval_ctx_t *ctx);

static double eval_base(eval_ctx_t *ctx)
{
    eval_skip_ws(ctx);
    double val;
    if (ctx->s[ctx->pos] == '(') {
        ctx->pos++;
        val = eval_expr(ctx);
        eval_skip_ws(ctx);
        if (ctx->s[ctx->pos] == ')')
            ctx->pos++;
        else
            ctx->err = true;
    } else {
        char *end;
        val = strtod(ctx->s + ctx->pos, &end);
        if (end == ctx->s + ctx->pos) {
            ctx->err = true;
            return 0;
        }
        ctx->pos = (size_t) (end - ctx->s);
    }
    return val;
}

static double eval_factor(eval_ctx_t *ctx)
{
    /* Unary minus: lower precedence than ^ so -2^2 = -(2^2) = -4 */
    eval_skip_ws(ctx);
    bool neg = false;
    if (ctx->s[ctx->pos] == '-') {
        neg = true;
        ctx->pos++;
    }
    double val = eval_base(ctx);
    eval_skip_ws(ctx);
    if (ctx->s[ctx->pos] == '^') {
        ctx->pos++;
        double exp = eval_factor(ctx); /* right-associative */
        val = pow(val, exp);
    }
    return neg ? -val : val;
}

static double eval_term(eval_ctx_t *ctx)
{
    double val = eval_factor(ctx);
    for (;;) {
        eval_skip_ws(ctx);
        char op = ctx->s[ctx->pos];
        if (op != '*' && op != '/' && op != '%')
            break;
        ctx->pos++;
        double rhs = eval_factor(ctx);
        if (op == '*')
            val *= rhs;
        else if (op == '/') {
            if (rhs == 0) {
                ctx->err = true;
                return 0;
            }
            val /= rhs;
        } else {
            if (rhs == 0) {
                ctx->err = true;
                return 0;
            }
            val = fmod(val, rhs);
        }
    }
    return val;
}

static double eval_expr(eval_ctx_t *ctx)
{
    double val = eval_term(ctx);
    for (;;) {
        eval_skip_ws(ctx);
        char op = ctx->s[ctx->pos];
        if (op != '+' && op != '-')
            break;
        ctx->pos++;
        double rhs = eval_term(ctx);
        val = (op == '+') ? val + rhs : val - rhs;
    }
    return val;
}

/* Format a double, stripping trailing zeros. Writes to buf, returns buf. */
static const char *fmt_number(double val, char *buf, size_t bufsz)
{
    snprintf(buf, bufsz, "%.10f", val);
    size_t len = strlen(buf);
    if (strchr(buf, '.')) {
        while (len > 1 && buf[len - 1] == '0')
            buf[--len] = '\0';
        if (len > 1 && buf[len - 1] == '.')
            buf[--len] = '\0';
    }
    return buf;
}

/* Evaluate arithmetic expression in-process.
 * Returns malloc'd result string, or NULL if input isn't arithmetic. */
static char *try_eval_math(const char *input)
{
    char *expr = extract_math_expr(input);
    if (!expr)
        return NULL;

    eval_ctx_t ctx = {.s = expr, .pos = 0, .err = false};
    double val = eval_expr(&ctx);

    /* Check for parse errors or trailing garbage */
    eval_skip_ws(&ctx);
    if (ctx.err || ctx.s[ctx.pos] != '\0' || !isfinite(val)) {
        free(expr);
        return NULL;
    }
    free(expr);

    /* Format result, strip trailing zeros */
    char result[384];
    fmt_number(val, result, sizeof(result));
    return strdup(result);
}

/* Try to evaluate an arithmetic expression from a raw string.
 * Returns the numeric result in *out, true on success.
 * This is a thin wrapper: extract_math_expr + eval_expr. */
static bool eval_raw_expr(const char *s, double *out)
{
    char *expr = extract_math_expr(s);
    if (!expr) {
        /* Try as a bare number (e.g. "4") */
        char *end;
        double v = strtod(s, &end);
        if (end == s)
            return false;
        while (*end == ' ' || *end == '\t')
            end++;
        if (end != s && *end == '\0' && isfinite(v)) {
            *out = v;
            return true;
        }
        return false;
    }
    eval_ctx_t ctx = {.s = expr, .pos = 0, .err = false};
    double val = eval_expr(&ctx);
    eval_skip_ws(&ctx);
    if (ctx.err || ctx.s[ctx.pos] != '\0' || !isfinite(val)) {
        free(expr);
        return false;
    }
    free(expr);
    *out = val;
    return true;
}


/* Try to evaluate a comparison question: "X vs Y", "X versus Y", "X or Y".
 * Detects patterns like "Which is larger, 2*2 vs 2+2?" or "2*2 vs 2+2".
 * Returns malloc'd result string, or NULL if not a comparison.
 */
static char *try_eval_comparison(const char *input)
{
    /* Work on a lowercase mutable copy */
    size_t ilen = strlen(input);
    if (ilen > 256)
        return NULL;
    char *work = malloc(ilen + 1);
    if (!work)
        return NULL;
    for (size_t i = 0; i <= ilen; i++)
        work[i] = (char) tolower((unsigned char) input[i]);

    /* Strip trailing punctuation/whitespace */
    size_t wlen = ilen;
    while (wlen > 0 && (work[wlen - 1] == '?' || work[wlen - 1] == '!' ||
                        work[wlen - 1] == '.' || work[wlen - 1] == ' '))
        work[--wlen] = '\0';

    /* Find separator: "vs", "versus", "or", "larger/bigger/smaller than" */
    const char *sep = NULL;
    size_t sep_len = 0;
    static const char *separators[] = {
        " versus ",       " vs. ",          " vs ",
        " or ",           " larger than ",  " bigger than ",
        " greater than ", " smaller than ", " less than ",
    };
    for (int i = 0; i < (int) (sizeof(separators) / sizeof(separators[0]));
         i++) {
        sep = strstr(work, separators[i]);
        if (sep) {
            sep_len = strlen(separators[i]);
            break;
        }
    }
    if (!sep) {
        free(work);
        return NULL;
    }

    /* Split into left and right parts */
    size_t left_end = (size_t) (sep - work);
    const char *right_start = sep + sep_len;

    /* Extract left side: skip leading noise like "which is larger," */
    char left_buf[256];
    if (left_end >= sizeof(left_buf)) {
        free(work);
        return NULL;
    }
    memcpy(left_buf, work, left_end);
    left_buf[left_end] = '\0';

    /* Normalize punctuation in left side: ? ! , ; : -> space */
    for (char *p = left_buf; *p; p++) {
        if (*p == '?' || *p == '!' || *p == ',' || *p == ';' || *p == ':')
            *p = ' ';
    }

    /* Strip leading question phrases from left side.
     * Match flexible patterns like "which one is <adjective>",
     * "which is <adj>", "what is <adj>", "compare", "is".
     * Skip any non-numeric words after the pattern. */
    char *lhs = left_buf;
    static const char *prefixes[] = {
        "which one is ", "which is ", "what is ", "compare ", "is ",
    };
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        size_t plen = strlen(prefixes[i]);
        if (!strncmp(lhs, prefixes[i], plen)) {
            lhs += plen;
            /* Skip adjective words (larger, big, etc.) until we hit
             * a digit, operator, or parenthesis (the actual expression) */
            while (*lhs) {
                while (*lhs == ' ')
                    lhs++;
                if (*lhs == '\0')
                    break;
                /* Stop if we see start of a math expression */
                if ((*lhs >= '0' && *lhs <= '9') || *lhs == '(' ||
                    *lhs == '-' || *lhs == '.')
                    break;
                /* Skip this word */
                while (*lhs && *lhs != ' ')
                    lhs++;
            }
            break;
        }
    }

    /* Extract right side */
    char right_buf[256];
    size_t rlen = strlen(right_start);
    if (rlen >= sizeof(right_buf)) {
        free(work);
        return NULL;
    }
    memcpy(right_buf, right_start, rlen + 1);

    free(work);

    /* Evaluate both sides */
    double lval, rval;
    if (!eval_raw_expr(lhs, &lval) || !eval_raw_expr(right_buf, &rval))
        return NULL;

    /* Format result */
    char lstr[384], rstr[384];
    fmt_number(lval, lstr, sizeof(lstr));
    fmt_number(rval, rstr, sizeof(rstr));

    /* Find the original expression text for display (use input, not lowered) */
    char result[1024];
    const char *cmp;
    if (lval > rval)
        cmp = "the first is larger";
    else if (lval < rval)
        cmp = "the second is larger";
    else
        cmp = "they are equal";

    snprintf(result, sizeof(result), "%s = %s, %s = %s, so %s.", lhs, lstr,
             right_buf, rstr, cmp);
    return strdup(result);
}

/* Signal handler for clean shutdown */
static volatile sig_atomic_t g_interrupted = 0;

static void sigint_handler(int sig)
{
    (void) sig;
    if (g_interrupted) {
        /* Double signal: force exit */
        const char msg[] = "\n[interrupted]\n";
        (void) !write(STDERR_FILENO, msg, sizeof(msg) - 1);
        _exit(1);
    }
    g_interrupted = 1;
}

static void history_push(int *history, int *len, int token)
{
    if (*len < HISTORY_MAX) {
        history[(*len)++] = token;
    } else {
        memmove(history, history + 1, (HISTORY_MAX - 1) * sizeof(int));
        history[HISTORY_MAX - 1] = token;
    }
}

/* Linenoise command completion callback */
static void completion_callback(const char *buf, linenoiseCompletions *lc)
{
    if (buf[0] != '/')
        return;
    static const char *cmds[] = {"/quit", "/exit", "/reset", "/save", "/help"};
    for (size_t i = 0; i < sizeof(cmds) / sizeof(cmds[0]); i++) {
        if (!strncmp(buf, cmds[i], strlen(buf)))
            linenoiseAddCompletion(lc, cmds[i]);
    }
}

/* Read all of stdin into a buffer (for pipe mode).
 * Returns NULL on error. Caller must free.
 */
static char *read_stdin_all(void)
{
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf)
        return NULL;

    while (1) {
        size_t n = fread(buf + len, 1, cap - len - 1, stdin);
        len += n;
        if (n == 0) {
            if (ferror(stdin)) {
                free(buf);
                return NULL;
            }
            break;
        }
        if (len >= cap - 1) {
            cap *= 2;
            char *newbuf = realloc(buf, cap);
            if (!newbuf) {
                free(buf);
                return NULL;
            }
            buf = newbuf;
        }
    }
    buf[len] = '\0';

    /* Strip trailing whitespace */
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' ||
                       buf[len - 1] == ' ' || buf[len - 1] == '\t'))
        buf[--len] = '\0';

    return (len > 0) ? buf : (free(buf), NULL);
}

/* Generation timing (prefill vs decode separated) */
typedef struct {
    int n_generated;
    int n_prefill;
    double prefill_ms;
    double decode_ms;
} gen_stats_t;

/* Generate response for encoded token sequence.
 * Feeds prompt through model, generates up to max_tokens,
 * streams decoded text to stdout.
 * Returns generation stats (token counts + timing).
 */
static gen_stats_t generate_response(bitmamba_model_t *m,
                                     gpt2_tokenizer_t *tok,
                                     const int32_t *prompt_ids,
                                     int n_prompt,
                                     int *history,
                                     int *history_len,
                                     const chat_config_t *cfg)
{
    gen_stats_t stats = {.n_prefill = n_prompt > 1 ? n_prompt - 1 : 0};

    /* Prefill all but last prompt token */
    if (n_prompt > 1) {
        double t0 = get_time_ms();
        model_prefill(m, prompt_ids, n_prompt - 1);
        stats.prefill_ms = get_time_ms() - t0;
    }

    /* Push all prompt tokens into history */
    for (int i = 0; i < n_prompt; i++)
        history_push(history, history_len, prompt_ids[i]);

    int current = prompt_ids[n_prompt - 1];
    int prev_token = -1;

    /* Repetition detectors:
     * 1. Same token N times in a row (e.g. "89||89||89...")
     * 2. Repeating n-gram pattern (ring buffer, last K == prev K)
     * 3. Paragraph flood: too many total \n tokens = rambling output */
    int repeat_token = -1, repeat_count = 0;
    int repeat_limit = cfg->repeat_limit > 0 ? cfg->repeat_limit : 6;
#define NGRAM_BUF_SIZE 64
    int ngram_buf[NGRAM_BUF_SIZE];
    int ngram_pos = 0, ngram_fill = 0;
    int para_count = 0; /* total newline tokens (paragraph counter) */
    int para_limit = cfg->newline_limit > 0 ? cfg->newline_limit : 8;

    double decode_start = get_time_ms();
    for (int i = 0; i < cfg->max_tokens && !g_interrupted; i++) {
        int next =
            model_forward_step(m, current, history, *history_len, cfg->penalty,
                               cfg->temp, cfg->min_p, cfg->top_p, cfg->top_k);

        /* Stop on EOS or null */
        if (next == TOKEN_EOS || next == 0)
            break;

        /* Check user-configured stop tokens */
        if (cfg->n_stop_tokens > 0) {
            bool is_stop = false;
            for (int j = 0; j < cfg->n_stop_tokens; j++) {
                if (next == cfg->stop_tokens[j]) {
                    is_stop = true;
                    break;
                }
            }
            if (is_stop)
                break;
        }

        /* Turn-boundary detection when using prompt template.
         * Stop on: double newline, or new-turn marker immediately after \n.
         * Allow single newlines for multi-sentence answers.
         */
        if (cfg->prompt_tpl) {
            bool is_nl = (prev_token == TOKEN_NL || prev_token == TOKEN_NLNL);
            /* After \n or \n\n, stop if model starts a new turn, question,
             * or response boundary ("Answer", "In" for conclusions) */
            if (is_nl && (next == TOKEN_Q || next == TOKEN_QUESTION ||
                          next == TOKEN_A_UC || next == TOKEN_B_UC ||
                          next == TOKEN_C_UC || next == TOKEN_D_UC ||
                          next == TOKEN_WHAT || next == TOKEN_HOW ||
                          next == TOKEN_WHY || next == TOKEN_WHEN ||
                          next == TOKEN_WHERE || next == TOKEN_IS_UC ||
                          next == TOKEN_ARE_UC || next == TOKEN_DO_UC ||
                          next == TOKEN_DOES || next == TOKEN_CAN_UC ||
                          next == TOKEN_ANSWER || next == TOKEN_ANSWER2))
                break;
        }

        /* Paragraph flood: too many total \n tokens = rambling output.
         * A good detailed answer has 3-6 paragraphs; 8+ is rambling. */
        if (next == TOKEN_NL || next == TOKEN_NLNL)
            para_count += (next == TOKEN_NLNL ? 2 : 1);
        if (cfg->prompt_tpl && para_count >= para_limit)
            break;

        /* Repetition stop: same token N times in a row -> degenerate loop */
        if (next == repeat_token) {
            if (++repeat_count >= repeat_limit)
                break;
        } else {
            repeat_token = next;
            repeat_count = 1;
        }

        /* N-gram repetition: detect repeating multi-token patterns.
         * If the last K tokens (for K=4..16) match the K tokens before
         * them, the model is stuck in a loop. */
        ngram_buf[ngram_pos] = next;
        ngram_pos = (ngram_pos + 1) % NGRAM_BUF_SIZE;
        if (ngram_fill < NGRAM_BUF_SIZE)
            ngram_fill++;
        if (ngram_fill >= 6) {
            bool ngram_loop = false;
            for (int k = 3; k <= ngram_fill / 2 && k <= 16; k++) {
                bool match = true;
                for (int j = 0; j < k; j++) {
                    int a = ngram_buf[(ngram_pos - 1 - j + NGRAM_BUF_SIZE) %
                                      NGRAM_BUF_SIZE];
                    int b = ngram_buf[(ngram_pos - 1 - j - k + NGRAM_BUF_SIZE) %
                                      NGRAM_BUF_SIZE];
                    if (a != b) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    ngram_loop = true;
                    break;
                }
            }
            if (ngram_loop)
                break;
        }

        /* Decode and stream to stdout */
        int decoded_len;
        char *decoded = tokenizer_decode_to_bytes(tok, next, &decoded_len);
        fwrite(decoded, 1, decoded_len, stdout);
        fflush(stdout);
        free(decoded);

        history_push(history, history_len, next);
        prev_token = next;
        current = next;
        stats.n_generated++;
    }
    stats.decode_ms = get_time_ms() - decode_start;

    return stats;
}

/* --- SSM State Persistence --- */

bool state_save(const bitmamba_model_t *m, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[chat] Failed to open state file for writing: %s\n",
                path);
        return false;
    }

    int n_layers = m->config.n_layers;
    const bitmamba_block_t *b0 = &m->layers[0];
    int d_inner = b0->d_inner;
    int d_conv = b0->d_conv;

    uint32_t magic = STATE_MAGIC;
    uint32_t nl = (uint32_t) n_layers;
    uint32_t di = (uint32_t) d_inner;
    uint32_t dc = (uint32_t) d_conv;
    bool ok = true;

    ok = ok && fwrite(&magic, sizeof(uint32_t), 1, f) == 1;
    ok = ok && fwrite(&nl, sizeof(uint32_t), 1, f) == 1;
    ok = ok && fwrite(&di, sizeof(uint32_t), 1, f) == 1;
    ok = ok && fwrite(&dc, sizeof(uint32_t), 1, f) == 1;

    for (int i = 0; i < n_layers && ok; i++) {
        const bitmamba_block_t *b = &m->layers[i];
        /* conv_state: (d_conv-1) * d_inner floats */
        int conv_size = (d_conv - 1) * d_inner;
        ok = ok && fwrite(b->conv_state, sizeof(float), conv_size, f) ==
                       (size_t) conv_size;
        /* ssm_state: d_inner floats */
        ok = ok && fwrite(b->ssm_state, sizeof(float), d_inner, f) ==
                       (size_t) d_inner;
        /* conv_pos: int */
        ok = ok && fwrite(&b->conv_pos, sizeof(int), 1, f) == 1;
    }

    if (fclose(f) != 0)
        ok = false;

    if (!ok) {
        fprintf(stderr, "[chat] Error writing state file: %s\n", path);
        return false;
    }
    return true;
}

bool state_load(bitmamba_model_t *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return false;

    uint32_t magic, nl, di, dc;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&nl, sizeof(uint32_t), 1, f) != 1 ||
        fread(&di, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dc, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    if (magic != STATE_MAGIC) {
        fprintf(stderr, "[chat] Invalid state file magic\n");
        fclose(f);
        return false;
    }

    /* Validate dimensions match current model */
    const bitmamba_block_t *b0 = &m->layers[0];
    if ((int) nl != m->config.n_layers || (int) di != b0->d_inner ||
        (int) dc != b0->d_conv) {
        fprintf(stderr,
                "[chat] State file dimensions mismatch "
                "(layers=%u/%d, d_inner=%u/%d, d_conv=%u/%d)\n",
                nl, m->config.n_layers, di, b0->d_inner, dc, b0->d_conv);
        fclose(f);
        return false;
    }

    for (int i = 0; i < m->config.n_layers; i++) {
        bitmamba_block_t *b = &m->layers[i];
        int conv_size = (dc - 1) * di;
        if (fread(b->conv_state, sizeof(float), conv_size, f) !=
                (size_t) conv_size ||
            fread(b->ssm_state, sizeof(float), di, f) != (size_t) di ||
            fread(&b->conv_pos, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "[chat] Truncated state file at layer %d\n", i);
            fclose(f);
            state_reset(m);
            return false;
        }
        /* Validate conv_pos is within ring-buffer bounds.
         * A corrupted state file could set conv_pos out of range,
         * causing a buffer overflow when the model steps forward.
         */
        if (b->conv_pos < 0 || b->conv_pos >= b->d_conv - 1) {
            fprintf(stderr,
                    "[chat] Invalid conv_pos %d in state file at layer %d\n",
                    b->conv_pos, i);
            fclose(f);
            state_reset(m);
            return false;
        }
    }

    fclose(f);
    return true;
}

void state_reset(bitmamba_model_t *m)
{
    for (int i = 0; i < m->config.n_layers; i++) {
        bitmamba_block_t *b = &m->layers[i];
        int conv_size = (b->d_conv - 1) * b->d_inner;
        memset(b->conv_state, 0, conv_size * sizeof(float));
        memset(b->ssm_state, 0, b->d_inner * sizeof(float));
        b->conv_pos = 0;
    }
}

/* Process one user turn: math eval or model generation.
 * Returns gen_stats (n_generated=0 if math-handled or error). */
static gen_stats_t handle_turn(const char *input,
                               bitmamba_model_t *m,
                               gpt2_tokenizer_t *tok,
                               int32_t *prompt_ids,
                               int *history,
                               int *history_len,
                               const chat_config_t *cfg)
{
    gen_stats_t empty = {0};

    /* Try comparison questions first ("X vs Y", "which is larger").
     * Then try plain arithmetic. If either succeeds, print the result
     * and feed the full Q&A exchange into the SSM state so multi-turn
     * follow-ups ("what was that times 2?") have context. */
    char *math_result = try_eval_comparison(input);
    if (!math_result)
        math_result = try_eval_math(input);
    if (math_result) {
        printf("%s\n", math_result);
        fflush(stdout);

        /* Build "Question: <input>\nDetailed answer: <result>" and prefill */
        if (cfg->prompt_tpl) {
            char *qa_text = NULL;
            int qa_len =
                asprintf(&qa_text, "Question: %s\nDetailed answer: %s\n", input,
                         math_result);
            free(math_result);
            if (qa_len > 0 && qa_text) {
                int n = tokenizer_encode(tok, qa_text, prompt_ids,
                                         MAX_PROMPT_TOKENS);
                free(qa_text);
                if (n > 0) {
                    model_prefill(m, prompt_ids, n);
                    for (int i = 0; i < n; i++)
                        history_push(history, history_len, prompt_ids[i]);
                }
            }
        } else {
            free(math_result);
        }
        return empty;
    }

    /* Rewrite vague queries for better base-model completions */
    char *rewritten =
        cfg->prompt_tpl ? rewrite_query(strip_qa_prefix(input)) : NULL;
    const char *effective = rewritten ? rewritten : input;

    char *prompt = apply_template(effective, cfg->prompt_tpl);
    free(rewritten);
    int n = tokenizer_encode(tok, prompt, prompt_ids, MAX_PROMPT_TOKENS);
    if (n == MAX_PROMPT_TOKENS)
        fprintf(stderr, "[chat] Warning: prompt truncated at %d tokens\n",
                MAX_PROMPT_TOKENS);
    free(prompt);

    if (n == 0) {
        fprintf(stderr, "[chat] (no tokens)\n");
        return empty;
    }

    return generate_response(m, tok, prompt_ids, n, history, history_len, cfg);
}

/* --- Chat Loop --- */

int chat_run(bitmamba_model_t *m,
             gpt2_tokenizer_t *tok,
             const chat_config_t *cfg)
{
    /* Install signal handler for clean Ctrl-C */
    struct sigaction sa = {.sa_handler = sigint_handler};
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    /* Try loading cached state */
    if (cfg->state_path) {
        if (state_load(m, cfg->state_path))
            fprintf(stderr, "[chat] Restored SSM state from %s\n",
                    cfg->state_path);
    }

    int *history = xmalloc(HISTORY_MAX * sizeof(int));
    int history_len = 0;
    int32_t *prompt_ids = xmalloc(MAX_PROMPT_TOKENS * sizeof(int32_t));

    bool is_tty = isatty(STDIN_FILENO);

    /* Pipe mode: read all stdin, generate, exit */
    if (cfg->pipe_mode || !is_tty) {
        char *input = read_stdin_all();
        if (!input) {
            fprintf(stderr, "[chat] Empty input\n");
            free(history);
            free(prompt_ids);
            return 1;
        }

        gen_stats_t gs =
            handle_turn(input, m, tok, prompt_ids, history, &history_len, cfg);
        free(input);
        printf("\n");
        if (gs.n_generated > 0 && gs.decode_ms > 0) {
            double decode_tps = gs.n_generated / (gs.decode_ms / 1000.0);
            if (gs.prefill_ms > 0)
                fprintf(stderr,
                        "[%d tok, %.1f tok/s | prefill %d tok in %.0f ms]\n",
                        gs.n_generated, decode_tps, gs.n_prefill,
                        gs.prefill_ms);
            else
                fprintf(stderr, "[%d tok, %.1f tok/s]\n", gs.n_generated,
                        decode_tps);
        }

        /* Save state if path specified */
        if (cfg->state_path)
            state_save(m, cfg->state_path);

        free(history);
        free(prompt_ids);
        return 0;
    }

    /* Interactive REPL with linenoise line editing */
    linenoiseSetCompletionCallback(completion_callback);
    linenoiseHistorySetMaxLen(100);

    fprintf(stderr,
            "[chat] BitMamba-2 assistant ready (%d layers, d=%d)\n"
            "[chat] Commands: /reset /save /help /quit  (Tab to complete)\n",
            m->config.n_layers, m->config.d_model);

    while (!g_interrupted) {
        char *line = linenoise("> ");
        if (!line)
            break; /* EOF or Ctrl-D */

        /* Skip empty lines */
        if (line[0] == '\0') {
            linenoiseFree(line);
            continue;
        }

        linenoiseHistoryAdd(line);

        /* Handle commands */
        if (line[0] == '/') {
            if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) {
                linenoiseFree(line);
                break;
            }
            if (!strcmp(line, "/reset")) {
                state_reset(m);
                history_len = 0;
                fprintf(stderr, "[chat] State reset\n");
                linenoiseFree(line);
                continue;
            }
            if (!strcmp(line, "/save")) {
                if (cfg->state_path) {
                    if (state_save(m, cfg->state_path))
                        fprintf(stderr, "[chat] State saved to %s\n",
                                cfg->state_path);
                } else {
                    fprintf(stderr,
                            "[chat] No state path (use --state <file>)\n");
                }
                linenoiseFree(line);
                continue;
            }
            if (!strcmp(line, "/help")) {
                fprintf(stderr,
                        "[chat] Commands:\n"
                        "  /reset  - Clear SSM state and history\n"
                        "  /save   - Save SSM state to file\n"
                        "  /quit   - Exit (alias: /exit)\n"
                        "  /help   - Show this help\n"
                        "[chat] Line editing: arrows, Home/End, Ctrl-A/E/K/U\n"
                        "[chat] Tab completes commands, Up/Down for history\n");
                linenoiseFree(line);
                continue;
            }
            fprintf(stderr, "[chat] Unknown command: %s (try /help)\n", line);
            linenoiseFree(line);
            continue;
        }

        g_interrupted = 0;

        gen_stats_t gs =
            handle_turn(line, m, tok, prompt_ids, history, &history_len, cfg);
        linenoiseFree(line);

        if (gs.n_generated > 0 && gs.decode_ms > 0) {
            double decode_tps = gs.n_generated / (gs.decode_ms / 1000.0);
            if (gs.prefill_ms > 0)
                fprintf(stderr,
                        "\n[%d tok, %.1f tok/s | prefill %d tok in %.0f ms]\n",
                        gs.n_generated, decode_tps, gs.n_prefill,
                        gs.prefill_ms);
            else
                fprintf(stderr, "\n[%d tok, %.1f tok/s]\n", gs.n_generated,
                        decode_tps);
        } else {
            fprintf(stderr, "\n");
        }

        g_interrupted = 0;
    }

    /* Save state on exit */
    if (cfg->state_path)
        state_save(m, cfg->state_path);

    fprintf(stderr, "\n");
    free(history);
    free(prompt_ids);
    return 0;
}
