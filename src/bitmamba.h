/*
 * BitMamba Inference Engine
 */

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Binary format magic number: "BIT2" */
#define MAGIC_NUMBER 0x42495432

/* GPT-2 vocabulary size */
#define VOCAB_SIZE 50257

/* Protected tokens (excluded from repetition penalty) */
#define TOKEN_EOS 50256
#define TOKEN_NL 198
#define TOKEN_NLNL 628
#define TOKEN_PERIOD 13

/* GPT-2 turn-boundary markers (used by chat mode to detect new turns).
 * These are GPT-2 byte-pair encoding IDs for specific word tokens. */
#define TOKEN_Q 48           /* "Q" */
#define TOKEN_QUESTION 24361 /* "Question" */
#define TOKEN_A_UC 32        /* "A" */
#define TOKEN_B_UC 33        /* "B" */
#define TOKEN_C_UC 34        /* "C" */
#define TOKEN_D_UC 35        /* "D" */
#define TOKEN_WHAT 2061      /* "What" */
#define TOKEN_HOW 2437       /* "How" */
#define TOKEN_WHY 5195       /* "Why" */
#define TOKEN_WHEN 2215      /* "When" */
#define TOKEN_WHERE 8496     /* "Where" */
#define TOKEN_IS_UC 3792     /* "Is" */
#define TOKEN_ARE_UC 8491    /* "Are" */
#define TOKEN_DO_UC 5211     /* "Do" */
#define TOKEN_DOES 13921     /* "Does" */
#define TOKEN_CAN_UC 6090    /* "Can" */
#define TOKEN_ANSWER 23998   /* "Answer" (with leading space) */
#define TOKEN_ANSWER2 33706  /* "Answer" (no leading space) */

/* BitMamba2 architecture constants */
#define BM_EXPAND_FACTOR 2 /* d_inner = d_model * BM_EXPAND_FACTOR */
#define BM_CONV_KERNEL 4   /* Conv1d kernel width */
#define BM_BUF_ALIGN 64    /* SIMD / cache-line alignment for buffers */

/* Safety limits for allocation bounds */
#define MAX_TOKEN_LENGTH 1024    /* Max bytes per tokenizer token */
#define MAX_TENSOR_DIM (1 << 20) /* Max dimension size (1M elements) */
#define MAX_VOCAB_SIZE 256000    /* Max vocabulary size */

/* Configuration */
typedef struct {
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;
} config_t;

/* Tensor (named tag required for dispatch.h forward declaration) */
typedef struct tensor {
    const float *data;          /* float32 weights (mmap'd, read-only) */
    const uint8_t *packed_data; /* packed 2-bit weights (mmap'd, read-only) */
    int rows;
    int cols;
    size_t n_elements; /* flat element count (handles ndim>2 tensors) */
    bool is_bitnet;
    float scale;
} tensor_t;

/* BitMamba Block */
typedef struct {
    tensor_t in_proj_norm, in_proj;
    tensor_t conv1d_w, conv1d_b;
    tensor_t dt_bias, A_log, D;
    tensor_t out_proj_norm, out_proj;

    float *conv1d_w_t; /* transposed: [tap][channel] for vectorized access */
    float *conv_state; /* SoA: [(d_conv-1) * d_inner] = [tap][channel] */
    float *ssm_state;  /* [d_inner] */
    float *A_precomp;  /* [n_heads] precomputed -exp(A_log) */
    int conv_pos;      /* ring-buffer write position [0, d_conv-2] */

    int d_model, d_inner, n_heads, head_dim, d_conv;

    /* NUMA weight replicas: per-node copies of packed weight data.
     * Index 0 is NULL (node 0 uses the mmap'd original).
     * Non-NULL entries are mmap'd+mbind'd copies on their NUMA node. */
    uint8_t *in_proj_packed_numa[8];
    uint8_t *out_proj_packed_numa[8];
} bitmamba_block_t;

/* Token Probability for Sampling */
typedef struct {
    int id;
    float val;
} token_prob_t;

/* Per-Stage Profiling */
#define MAX_PROFILE_LAYERS 256

typedef struct {
    double embed_ms;
    double layer_ms[MAX_PROFILE_LAYERS];
    double final_norm_ms;
    double lm_head_ms;
    double sample_ms;
    int n_layers;
    int n_tokens;

    /* Prefill profiling */
    double prefill_embed_ms;
    double prefill_layer_ms[MAX_PROFILE_LAYERS];
    int prefill_n_tokens;
} token_profile_t;

/* BitMamba Model */
typedef struct {
    config_t config;
    tensor_t embed, norm_f, lm_head_norm, lm_head;
    bitmamba_block_t *layers;

    /* mmap info */
    void *mmap_addr;  /* base of mmap'd model file */
    size_t mmap_size; /* size of mmap'd region */

    /* Pre-allocated working buffers */
    float *current_x;  /* [d_model] */
    float *next_x;     /* [d_model] */
    float *proj_out;   /* [max(in_proj.rows)] */
    float *y_buffer;   /* [d_inner] */
    float *final_feat; /* [d_model] */
    float *logits;     /* [vocab_size] */

    /* Shared bitlinear buffers (sized to max input dim) */
    float *bl_x_norm;   /* [bl_max_dim] */
    int8_t *bl_x_quant; /* [bl_max_dim + 32] (padded) */
    int bl_max_dim;

    /* Pre-allocated sampler buffer */
    token_prob_t *sampler_probs; /* [vocab_size] */

    /* NUMA weight replicas: per-node copies of LM head packed data.
     * Index 0 is NULL (node 0 uses the mmap'd original).
     * Non-NULL entries are heap-allocated copies on their respective node.
     */
    uint8_t *lm_head_packed_numa[8]; /* max 8 NUMA nodes */

    int32_t *scalar_lut_buf;   /* persistent T-MAC LUT (16KB) */
    uint64_t *penalty_visited; /* bitset for repetition penalty dedup */

    /* Prefill scratch buffers (grow-on-demand, reused across calls) */
    float *pf_x_batch;
    float *pf_out_batch;
    float *pf_proj_batch;
    float *pf_y_batch;
    int8_t *pf_quant_batch;
    int pf_capacity; /* n_tokens these buffers are sized for */

    /* Runtime flags (formerly globals) */
    bool use_gpu;
    bool profile_enabled;
    token_profile_t profile;
} bitmamba_model_t;

/* Tokenizer */
#define TOK_HT_SIZE 131072 /* 2^17, > 2 * VOCAB_SIZE */
#define TOK_HT_MASK (TOK_HT_SIZE - 1)

typedef struct {
    char **id_to_token; /* [vocab_size] token strings */
    int *token_lengths; /* [vocab_size] byte length of each token */
    int vocab_size;
    int *ht_ids; /* [TOK_HT_SIZE] hash table: -1=empty, else token_id */
    bool is_byte_encoded; /* true if tokens use GPT-2 byte encoding */
} gpt2_tokenizer_t;

/* Inference Stats */
typedef struct {
    int total_tokens;
    double total_time_ms;
    double peak_memory_mb;
    double initial_memory_mb;
} inference_stats_t;

/* Function declarations */

/* utils.c */
void *xmalloc(size_t size);
void *xcalloc(size_t nmemb, size_t size);
void *xmalloc_aligned(size_t size, size_t alignment);
void *xcalloc_aligned(size_t nmemb, size_t size, size_t alignment);
double get_memory_usage_mb(void);
double get_time_ms(void);
double stats_tokens_per_second(const inference_stats_t *s);
void stats_print_summary(const inference_stats_t *s);

/* utils.c - XorShift64 PRNG */
void rng_seed(uint64_t seed);
uint64_t rng_next(void);
float rng_float(void);

/* utils.c - Profiling */
void profile_reset(bitmamba_model_t *m);
void profile_print(const bitmamba_model_t *m);

/* quantization.c */
extern int8_t UNPACK_LUT[256][4];
extern uint8_t TMAC_P_MASK[256];
extern uint8_t TMAC_N_MASK[256];
void init_lut(void);

/* Kernel input validation (assert-like, guards programming errors) */
#define BITLINEAR_CHECK(n, w)                                                  \
    do {                                                                       \
        if ((n) != (w)->cols) {                                                \
            fprintf(stderr, "Fatal: bitlinear dim mismatch n=%d vs cols=%d\n", \
                    (n), (w)->cols);                                           \
            exit(1);                                                           \
        }                                                                      \
        if (!(w)->is_bitnet) {                                                 \
            fprintf(stderr,                                                    \
                    "Fatal: bitlinear requires packed BitNet weights\n");      \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* Inline math helpers (used by kernels and block.c) */
static inline float silu(float x)
{
    return x / (1.0f + expf(-x));
}
static inline float softplus(float x)
{
    if (x > 20.0f)
        return x;
    return logf(1.0f + expf(x));
}

/* Runtime kernel dispatch (replaces direct kernel declarations) */
#include "dispatch.h"

/* block.c */
void block_init_cache(bitmamba_block_t *b, int d_model, int n_heads);
void block_step(bitmamba_block_t *b,
                const float *u,
                float *out_buffer,
                float *proj_out,
                float *y,
                float *bl_x_norm,
                int8_t *bl_x_quant);
void block_step_batch(bitmamba_block_t *b,
                      const float *u_batch,
                      int n_tokens,
                      float *out_batch,
                      float *proj_batch,
                      float *y_batch,
                      float *bl_x_norm,
                      int8_t *bl_x_quant_batch,
                      int quant_stride,
                      bool use_gpu);
void block_ssm_scan(bitmamba_block_t *b, float *proj_out, float *y);
void block_free(bitmamba_block_t *b);

/* tokenizer.c */
bool tokenizer_load(gpt2_tokenizer_t *tok, const char *path);
int tokenizer_encode(const gpt2_tokenizer_t *tok,
                     const char *text,
                     int32_t *tokens,
                     int max_tokens);
const char *tokenizer_decode(const gpt2_tokenizer_t *tok, int id);
char *tokenizer_decode_to_bytes(const gpt2_tokenizer_t *tok,
                                int id,
                                int *out_len);
void tokenizer_free(gpt2_tokenizer_t *tok);

/* model.c */
bool model_load(bitmamba_model_t *m, const char *path);
void model_prefill(bitmamba_model_t *m, const int32_t *tokens, int n_tokens);
int model_forward_step(bitmamba_model_t *m,
                       int token,
                       const int *history,
                       int history_len,
                       float penalty,
                       float temp,
                       float min_p,
                       float top_p,
                       int top_k);
void model_free(bitmamba_model_t *m);
