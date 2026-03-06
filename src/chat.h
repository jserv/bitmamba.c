/*
 * BitMamba Chat Mode - Lightweight AI Assistant
 *
 * Interactive multi-turn conversation using BitMamba-2's SSM state.
 * Mamba's fixed-size state (conv_state + ssm_state per layer) persists
 * naturally across tokens, giving us multi-turn chat with constant
 * memory -- no growing KV cache, no context window limit.
 */

#pragma once

#include "bitmamba.h"

/* SSM state file magic: "BMST" */
#define STATE_MAGIC 0x424D5354

#define CHAT_MAX_STOP_TOKENS 16

/* Chat configuration */
typedef struct {
    float temp;
    float penalty;
    float min_p;
    float top_p;
    int top_k;
    int max_tokens;
    const char *state_path; /* SSM state save/load path (NULL = no caching) */
    const char
        *prompt_tpl;  /* prompt template with %s for user input (NULL = raw) */
    bool pipe_mode;   /* true = single stdin prompt, no REPL */
    int repeat_limit; /* consecutive identical tokens before stop (0=default) */
    int newline_limit; /* consecutive \n tokens before stop (0=default) */
    int stop_tokens[CHAT_MAX_STOP_TOKENS]; /* extra stop token IDs */
    int n_stop_tokens;                     /* number of extra stop tokens */
} chat_config_t;

/* Save SSM state (all conv_state + ssm_state + conv_pos per layer).
 * Returns true on success. */
bool state_save(const bitmamba_model_t *m, const char *path);

/* Load SSM state from file. Validates dimensions match the model.
 * Returns true on success (state restored), false on failure (state
 * unchanged). */
bool state_load(bitmamba_model_t *m, const char *path);

/* Reset all SSM state to zero (start fresh conversation). */
void state_reset(bitmamba_model_t *m);

/* Run interactive chat loop. Returns 0 on clean exit, 1 on error. */
int chat_run(bitmamba_model_t *m,
             gpt2_tokenizer_t *tok,
             const chat_config_t *cfg);
