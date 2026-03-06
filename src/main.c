/*
 * BitMamba Inference Engine - CLI Entry Point
 * C11 (GNU) port of the C++ reference implementation.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bitmamba.h"
#include "chat.h"
#include "threadpool.h"

#define HISTORY_MAX 256

static int parse_int(const char *s, int fallback)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < INT_MIN ||
        v > INT_MAX)
        return fallback;
    return (int) v;
}

static float parse_float(const char *s, float fallback)
{
    char *end;
    errno = 0;
    float v = strtof(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE)
        return fallback;
    return v;
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

int main(int argc, char **argv)
{
    /* Check for flags anywhere in args */
    bool use_gpu = false;
    bool profile_enabled = false;
    bool chat_mode = false;
    const char *state_path = NULL;
    const char *prompt_tpl = NULL;
    bool no_template = false;
    int requested_threads = 0; /* 0 = auto-detect */
    bool seed_set = false;
    uint64_t seed = 0;
    int repeat_limit = 0;  /* 0 = use default */
    int newline_limit = 0; /* 0 = use default */
    int stop_tokens[CHAT_MAX_STOP_TOKENS];
    int n_stop_tokens = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--profile")) {
            profile_enabled = true;
            for (int j = i; j < argc - 1; j++)
                argv[j] = argv[j + 1];
            argc--;
            i--;
        } else if (!strcmp(argv[i], "--gpu")) {
            use_gpu = true;
            for (int j = i; j < argc - 1; j++)
                argv[j] = argv[j + 1];
            argc--;
            i--;
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            requested_threads = parse_int(argv[i + 1], 0);
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            char *endptr;
            errno = 0;
            seed = (uint64_t) strtoull(argv[i + 1], &endptr, 10);
            if (endptr == argv[i + 1] || *endptr != '\0' || errno == ERANGE ||
                argv[i + 1][0] == '-') {
                fprintf(stderr, "Error: Invalid --seed value '%s'\n",
                        argv[i + 1]);
                return 1;
            }
            seed_set = true;
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        } else if (!strcmp(argv[i], "--chat")) {
            chat_mode = true;
            for (int j = i; j < argc - 1; j++)
                argv[j] = argv[j + 1];
            argc--;
            i--;
        } else if (!strcmp(argv[i], "--state") && i + 1 < argc) {
            state_path = argv[i + 1];
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        } else if (!strcmp(argv[i], "--template") && i + 1 < argc) {
            prompt_tpl = argv[i + 1];
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        } else if (!strcmp(argv[i], "--no-template")) {
            no_template = true;
            for (int j = i; j < argc - 1; j++)
                argv[j] = argv[j + 1];
            argc--;
            i--;
        } else if (!strcmp(argv[i], "--repeat-limit") && i + 1 < argc) {
            repeat_limit = parse_int(argv[i + 1], 0);
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        } else if (!strcmp(argv[i], "--newline-limit") && i + 1 < argc) {
            newline_limit = parse_int(argv[i + 1], 0);
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        } else if (!strcmp(argv[i], "--stop-token") && i + 1 < argc) {
            if (n_stop_tokens < CHAT_MAX_STOP_TOKENS) {
                char *endptr;
                errno = 0;
                long v = strtol(argv[i + 1], &endptr, 10);
                if (endptr == argv[i + 1] || *endptr != '\0' ||
                    errno == ERANGE || v < 0 || v > INT_MAX) {
                    fprintf(stderr, "Error: Invalid --stop-token value '%s'\n",
                            argv[i + 1]);
                    return 1;
                }
                stop_tokens[n_stop_tokens++] = (int) v;
            } else {
                fprintf(stderr, "Warning: Too many --stop-token (max %d)\n",
                        CHAT_MAX_STOP_TOKENS);
            }
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        } else if (!strcmp(argv[i], "--poll") && i + 1 < argc) {
            bm_set_poll(parse_int(argv[i + 1], 100));
            for (int j = i; j < argc - 2; j++)
                argv[j] = argv[j + 2];
            argc -= 2;
            i--;
        }
    }

    /* Initialize thread pool and kernel dispatch */
    bm_set_threads(requested_threads);
    dispatch_init();

    /* Chat mode: only requires model path */
    if (chat_mode) {
        if (argc < 2) {
            fprintf(stderr,
                    "Usage: %s --chat [--state <file>] [--template <fmt>] "
                    "[--no-template]\n"
                    "       [--threads N] [--seed N] "
                    "[--stop-token ID] [--repeat-limit N]\n"
                    "       [--newline-limit N] <model.bin> "
                    "[temp] [penalty] [min_p] [top_p] [top_k] "
                    "[max_tokens]\n",
                    argv[0]);
            fprintf(stderr,
                    "\nInteractive chat mode. Type text, get responses.\n"
                    "Default template wraps input as "
                    "\"Question: <input>\\nDetailed answer:\".\n"
                    "Use --template to customize (%%s = user input), "
                    "--no-template for raw.\n"
                    "Pipe mode: echo \"prompt\" | %s --chat model.bin\n"
                    "\nCommands: /reset, /save, /quit\n",
                    argv[0]);
            return 1;
        }

        fprintf(stderr, "[INFO] Threads: %d\n", bm_get_threads());

        /* Load model */
        bitmamba_model_t model = {0};
        model.use_gpu = use_gpu;
        if (!model_load(&model, argv[1]))
            return 1;

        /* Load tokenizer (required for chat) */
        gpt2_tokenizer_t tokenizer = {0};
        if (!tokenizer_load(&tokenizer, "tokenizer.bin")) {
            model_free(&model);
            return 1;
        }

        rng_seed(seed_set ? (seed ? seed : 1) : (uint64_t) time(NULL));

        /* Parse optional sampling parameters */
        chat_config_t chat_cfg = {
            .temp = (argc > 2) ? parse_float(argv[2], 0.8f) : 0.8f,
            .penalty = (argc > 3) ? parse_float(argv[3], 1.15f) : 1.15f,
            .min_p = (argc > 4) ? parse_float(argv[4], 0.05f) : 0.05f,
            .top_p = (argc > 5) ? parse_float(argv[5], 0.9f) : 0.9f,
            .top_k = (argc > 6) ? parse_int(argv[6], 40) : 40,
            .max_tokens = (argc > 7) ? parse_int(argv[7], 400) : 400,
            .state_path = state_path,
            .prompt_tpl = no_template  ? NULL
                          : prompt_tpl ? prompt_tpl
                                       : "Question: %s\nDetailed answer:",
            .pipe_mode = false,
            .repeat_limit = repeat_limit,
            .newline_limit = newline_limit,
            .n_stop_tokens = n_stop_tokens,
        };
        memcpy(chat_cfg.stop_tokens, stop_tokens,
               (size_t) n_stop_tokens * sizeof(int));

        int ret = chat_run(&model, &tokenizer, &chat_cfg);

        tokenizer_free(&tokenizer);
        model_free(&model);
        bm_thread_pool_free();
        return ret;
    }

    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s [--profile] [--gpu] [--threads N] [--seed N] "
                "[--poll 0-100] "
                "<model.bin> <input> <mode> "
                "[temp] [penalty] [min_p] [top_p] [top_k] [max_tokens] "
                "[output_mode]\n",
                argv[0]);
        fprintf(stderr, "\nFlags:\n");
        fprintf(stderr, "  --profile   - Enable per-stage profiling\n");
        fprintf(stderr,
                "  --gpu       - Use Metal GPU for batched prefill "
                "(Apple Silicon only)\n");
        fprintf(stderr,
                "  --threads N - Set thread count (0=auto, 1=single, "
                "default: auto)\n");
        fprintf(stderr,
                "  --seed N    - Fixed PRNG seed for reproducible "
                "sampling (default: time-based)\n");
        fprintf(stderr,
                "  --poll 0-100 - Poll intensity (0=condvar, 100=spin, "
                "default: 100)\n");
        fprintf(stderr, "  --chat      - Interactive chat mode\n");
        fprintf(stderr,
                "  --state F   - SSM state file for chat persistence\n");
        fprintf(stderr,
                "  --template  - Chat prompt template (%%s = input, "
                "default: \"Question: %%s\\nDetailed answer:\")\n");
        fprintf(stderr,
                "  --no-template - Disable prompt template (raw input)\n");
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  model.bin   - Path to model file\n");
        fprintf(stderr,
                "  input       - Input text (tokenizer mode) or token IDs "
                "(raw mode)\n");
        fprintf(stderr,
                "  mode        - 'tokenizer' (text input/output) or 'raw' "
                "(token IDs input/output)\n");
        fprintf(stderr, "  temp        - Temperature (default: 0.8)\n");
        fprintf(stderr, "  penalty     - Repetition Penalty (default: 1.15)\n");
        fprintf(stderr, "  min_p       - Min-P sampling (default: 0.05)\n");
        fprintf(stderr,
                "  top_p       - Top-P/nucleus sampling (default: 0.90)\n");
        fprintf(stderr, "  top_k       - Top-K sampling (default: 40)\n");
        fprintf(stderr,
                "  max_tokens  - Max tokens to generate (default: 400)\n");
        fprintf(stderr,
                "  output_mode - 'bench' (default) shows stats, "
                "'clean' shows only output\n");
        fprintf(stderr, "\nExamples:\n");
        fprintf(stderr,
                "  Tokenizer mode: ./bitmamba model.bin \"Hello, I am\" "
                "tokenizer 0.7 1.1\n");
        fprintf(stderr,
                "  Raw mode:       ./bitmamba model.bin \"15496 11 314 716\" "
                "raw 0.7 1.1\n");
        fprintf(stderr,
                "  With profiling: ./bitmamba --profile model.bin \"Hello\" "
                "tokenizer\n");
        fprintf(stderr, "  Chat mode:      ./bitmamba --chat model.bin\n");
        fprintf(stderr,
                "  Pipe mode:      echo \"Hello\" | ./bitmamba --chat "
                "model.bin\n");
        return 1;
    }

    /* Validate mode */
    const char *mode = argv[3];
    if (strcmp(mode, "tokenizer") && strcmp(mode, "raw")) {
        fprintf(stderr, "Error: Invalid mode '%s'\n", mode);
        fprintf(stderr, "Mode must be either 'tokenizer' or 'raw'\n");
        return 1;
    }
    bool use_tokenizer = !strcmp(mode, "tokenizer");

    /* Parse optional parameters (matching C++ defaults) */
    float temp = 0.8f;
    float penalty = 1.15f;
    float min_p = 0.05f;
    float top_p = 0.90f;
    int top_k = 40;
    int max_tokens = 400;

    if (argc > 4)
        temp = parse_float(argv[4], temp);
    if (argc > 5)
        penalty = parse_float(argv[5], penalty);
    if (argc > 6)
        min_p = parse_float(argv[6], min_p);
    if (argc > 7)
        top_p = parse_float(argv[7], top_p);
    if (argc > 8)
        top_k = parse_int(argv[8], top_k);
    if (argc > 9) {
        max_tokens = parse_int(argv[9], max_tokens);
        if (max_tokens <= 0)
            max_tokens = 1;
    }

    /* Check for optional output mode */
    bool is_clean = false;
    if (argc > 10 && !strcmp(argv[10], "clean"))
        is_clean = true;

    if (!is_clean)
        fprintf(stderr, "[INFO] Threads: %d\n", bm_get_threads());

    /* Measure RAM before model loading */
    double ram_before = get_memory_usage_mb();
    if (!is_clean)
        fprintf(stderr, "[INFO] RAM before loading model: %.2f MB\n",
                ram_before);

    /* Load model */
    bitmamba_model_t model = {0};
    model.use_gpu = use_gpu;
    if (!model_load(&model, argv[1]))
        return 1;
    model.profile_enabled = profile_enabled;

    double ram_after = get_memory_usage_mb();
    if (!is_clean)
        fprintf(stderr,
                "[INFO] RAM after loading model: %.2f MB (model: %.2f MB)\n",
                ram_after, ram_after - ram_before);

    /* Initialize profiler */
    if (model.profile_enabled) {
        profile_reset(&model);
        if (!is_clean)
            fprintf(stderr, "[INFO] Profiling enabled\n");
    }

    /* Load tokenizer if needed */
    gpt2_tokenizer_t tokenizer = {0};
    if (use_tokenizer) {
        if (!tokenizer_load(&tokenizer, "tokenizer.bin")) {
            model_free(&model);
            return 1;
        }
    }

    /* Parse input */
    int32_t *prompt_ids = xmalloc(1024 * sizeof(int32_t));
    int n_prompt = 0;
    const char *input_str = argv[2];

    if (use_tokenizer) {
        n_prompt = tokenizer_encode(&tokenizer, input_str, prompt_ids, 1024);
        if (!is_clean)
            fprintf(stderr, "[INFO] Input Text: \"%s\"\n", input_str);
    } else {
        /* Parse space-separated token IDs */
        char *input_copy = strdup(input_str);
        if (!input_copy) {
            fprintf(stderr, "Error: Failed to allocate memory\n");
            free(prompt_ids);
            model_free(&model);
            return 1;
        }
        char *tok = strtok(input_copy, " ");
        while (tok && n_prompt < 1024) {
            char *endptr;
            errno = 0;
            long v = strtol(tok, &endptr, 10);
            if (endptr == tok || *endptr != '\0' || errno == ERANGE || v < 0 ||
                v > INT_MAX) {
                fprintf(stderr, "Error: Invalid token ID '%s' in raw input\n",
                        tok);
                free(input_copy);
                free(prompt_ids);
                model_free(&model);
                return 1;
            }
            prompt_ids[n_prompt++] = (int32_t) v;
            tok = strtok(NULL, " ");
        }
        free(input_copy);
    }

    if (n_prompt == 0) {
        fprintf(stderr, "Error: Empty prompt (no tokens after encoding)\n");
        free(prompt_ids);
        if (use_tokenizer)
            tokenizer_free(&tokenizer);
        model_free(&model);
        return 1;
    }

    if (!is_clean) {
        fprintf(stderr, "[INFO] Input Tokens (%d): ", n_prompt);
        for (int i = 0; i < n_prompt; i++)
            fprintf(stderr, "%d ", prompt_ids[i]);
        fprintf(stderr, "\n");
    }

    /* Initialize stats */
    double initial_mem = get_memory_usage_mb();
    inference_stats_t stats = {
        .initial_memory_mb = initial_mem,
        .peak_memory_mb = initial_mem,
    };

    /* Process prompt (prefill) */
    if (!is_clean)
        fprintf(stderr, "[INFO] Processing prompt...\n");
    double prefill_start = get_time_ms();

    int *history = xmalloc(HISTORY_MAX * sizeof(int));
    int history_len = 0;

    /* Batched prefill: process all but last token without LM head */
    if (n_prompt > 1)
        model_prefill(&model, prompt_ids, n_prompt - 1);

    int current = prompt_ids[n_prompt - 1];
    for (int i = 0; i < n_prompt; i++)
        history_push(history, &history_len, prompt_ids[i]);

    double prefill_end = get_time_ms();
    if (!is_clean)
        fprintf(stderr, "[INFO] Prefill completed in %.2f ms (%d tokens)\n",
                prefill_end - prefill_start, n_prompt);

    /* Generation */
    if (!is_clean)
        fprintf(stderr, "[INFO] Generating tokens...\n");

    rng_seed(seed_set ? (seed ? seed : 1) : (uint64_t) time(NULL));

    int *generated_tokens = xmalloc(max_tokens * sizeof(int));
    int n_generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        double token_start = get_time_ms();

        int next = model_forward_step(&model, current, history, history_len,
                                      penalty, temp, min_p, top_p, top_k);

        double token_end = get_time_ms();
        stats.total_tokens++;
        stats.total_time_ms += token_end - token_start;

        generated_tokens[n_generated++] = next;

        if (!is_clean && stats.total_tokens % 50 == 0) {
            double current_mem = get_memory_usage_mb();
            if (current_mem > stats.peak_memory_mb)
                stats.peak_memory_mb = current_mem;
            fprintf(stderr, "[STATS] %d tokens | %.2f tok/s | RAM: %.2f MB\n",
                    stats.total_tokens, stats_tokens_per_second(&stats),
                    current_mem);
        }

        current = next;
        history_push(history, &history_len, next);

        /* Stop tokens */
        if (next == TOKEN_EOS || next == 0)
            break;
    }

    /* Output */
    if (use_tokenizer) {
        if (!is_clean)
            printf("\n=== Generated Text ===\n");
        for (int i = 0; i < n_generated; i++) {
            int decoded_len;
            char *decoded = tokenizer_decode_to_bytes(
                &tokenizer, generated_tokens[i], &decoded_len);
            fwrite(decoded, 1, decoded_len, stdout);
            free(decoded);
        }
        if (!is_clean)
            printf("\n=== End Inference ===\n");
        else
            printf("\n");
    } else {
        if (!is_clean)
            printf("\n=== Generated Token IDs ===\n");
        for (int i = 0; i < n_generated; i++)
            printf("%d ", generated_tokens[i]);
        if (!is_clean)
            printf("\n=== End Inference ===\n");
        else
            printf("\n");
    }

    if (!is_clean) {
        stats_print_summary(&stats);
        profile_print(&model);
    }

    /* Cleanup */
    free(prompt_ids);
    free(history);
    free(generated_tokens);
    if (use_tokenizer)
        tokenizer_free(&tokenizer);
    model_free(&model);
    bm_thread_pool_free();

    return 0;
}
