/*
 * Runtime CPU kernel dispatch for BitMamba.
 *
 * A single binary per ISA family (x86_64, arm64) selects the fastest
 * kernel variant at runtime. The scalar fallback is always available.
 */

#pragma once

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "threadpool.h"

/* Forward declarations to avoid circular include with bitmamba.h */
struct tensor;

/* Kernel function pointer types */
typedef float (*rms_norm_fn_t)(const float *restrict x,
                               int size,
                               const float *restrict weight,
                               float *restrict out);

typedef void (*bitlinear_forward_fn_t)(const float *restrict x,
                                       int n,
                                       const struct tensor *restrict w,
                                       const struct tensor *restrict norm_w,
                                       float *restrict out,
                                       float *restrict x_norm_buf,
                                       int8_t *restrict x_quant_buf);

typedef void (*bitlinear_forward_batch_fn_t)(
    const float *restrict x_batch,
    int n_tokens,
    int n,
    const struct tensor *restrict w,
    const struct tensor *restrict norm_w,
    float *restrict out_batch,
    float *restrict x_norm_buf,
    int8_t *restrict x_quant_batch,
    int quant_stride);

/*
 * Decomposed bitlinear: norm+quantize phase (returns scale_x).
 * Caller computes inv_scale = 1 / (scale_x * w->scale).
 * NULL for scalar (T-MAC column blocking doesn't decompose cleanly).
 */
typedef float (*bitlinear_prepare_fn_t)(const float *restrict x,
                                        int n,
                                        const struct tensor *restrict norm_w,
                                        float *restrict x_norm_buf,
                                        int8_t *restrict x_quant_buf);

/* SSM inner loop arguments: passed to vectorized or scalar SSM kernel */
typedef struct {
    float *ptr_z, *ptr_x, *ptr_B, *ptr_C, *ptr_dt;
    const float *dt_bias;
    const float *A_precomp;
    const float *conv1d_w_t; /* [d_conv][d_inner] transposed layout */
    const float *conv1d_b;
    float *cs0, *cs1, *cs2, *cs_write;
    float *ssm_state;
    const float *D;
    float *y;
    int head_dim, d_inner, n_heads;
} ssm_inner_args_t;

typedef void (*ssm_inner_fn_t)(const ssm_inner_args_t *a,
                               int idx_start,
                               int idx_end);

/* Shared row-work struct for decomposed bitlinear dispatch */
typedef struct {
    const int8_t *x_quant;
    const uint8_t *packed_data;
    float *out;
    float inv_scale;
    int rows;
    int cols;
    int packed_stride;
    _Atomic int *chunk_ctr; /* NULL = static partition (NUMA or disabled) */
    int chunk_size;         /* rows per chunk (multiple of 4 for AVX2/NEON) */
} bitlinear_row_work_t;

/* Dispatch table: populated once at startup by dispatch_init() */
typedef struct {
    rms_norm_fn_t rms_norm;
    bitlinear_forward_fn_t bitlinear_forward;
    bitlinear_forward_batch_fn_t bitlinear_forward_batch;
    bitlinear_prepare_fn_t bitlinear_prepare;
    bm_task_fn_t bitlinear_row_worker;
    ssm_inner_fn_t ssm_inner;
    const char *name; /* "scalar", "avx2", "neon" */
} kernel_dispatch_t;

extern kernel_dispatch_t g_kernels;

/* Detect CPU features and populate g_kernels. Call once before model_load(). */
void dispatch_init(void);

/* Scalar fallback (always linked) */
float scalar_rms_norm(const float *restrict x,
                      int size,
                      const float *restrict weight,
                      float *restrict out);
void scalar_bitlinear_forward(const float *restrict x,
                              int n,
                              const struct tensor *restrict w,
                              const struct tensor *restrict norm_w,
                              float *restrict out,
                              float *restrict x_norm_buf,
                              int8_t *restrict x_quant_buf);
void scalar_bitlinear_forward_batch(const float *restrict x_batch,
                                    int n_tokens,
                                    int n,
                                    const struct tensor *restrict w,
                                    const struct tensor *restrict norm_w,
                                    float *restrict out_batch,
                                    float *restrict x_norm_buf,
                                    int8_t *restrict x_quant_batch,
                                    int quant_stride);
void scalar_set_lut_buf(int32_t *buf);
void scalar_ssm_inner(const ssm_inner_args_t *a, int idx_start, int idx_end);

/* AVX2 variant (linked on x86_64 builds) */
float avx2_rms_norm(const float *restrict x,
                    int size,
                    const float *restrict weight,
                    float *restrict out);
void avx2_bitlinear_forward(const float *restrict x,
                            int n,
                            const struct tensor *restrict w,
                            const struct tensor *restrict norm_w,
                            float *restrict out,
                            float *restrict x_norm_buf,
                            int8_t *restrict x_quant_buf);
void avx2_bitlinear_forward_batch(const float *restrict x_batch,
                                  int n_tokens,
                                  int n,
                                  const struct tensor *restrict w,
                                  const struct tensor *restrict norm_w,
                                  float *restrict out_batch,
                                  float *restrict x_norm_buf,
                                  int8_t *restrict x_quant_batch,
                                  int quant_stride);
float avx2_bitlinear_prepare(const float *restrict x,
                             int n,
                             const struct tensor *restrict norm_w,
                             float *restrict x_norm_buf,
                             int8_t *restrict x_quant_buf);
void avx2_row_worker(int tid, int n_threads, void *arg);
void avx2_ssm_inner(const ssm_inner_args_t *a, int idx_start, int idx_end);

/* NEON variant (linked on arm64 builds) */
float neon_rms_norm(const float *restrict x,
                    int size,
                    const float *restrict weight,
                    float *restrict out);
void neon_bitlinear_forward(const float *restrict x,
                            int n,
                            const struct tensor *restrict w,
                            const struct tensor *restrict norm_w,
                            float *restrict out,
                            float *restrict x_norm_buf,
                            int8_t *restrict x_quant_buf);
void neon_bitlinear_forward_batch(const float *restrict x_batch,
                                  int n_tokens,
                                  int n,
                                  const struct tensor *restrict w,
                                  const struct tensor *restrict norm_w,
                                  float *restrict out_batch,
                                  float *restrict x_norm_buf,
                                  int8_t *restrict x_quant_batch,
                                  int quant_stride);
float neon_bitlinear_prepare(const float *restrict x,
                             int n,
                             const struct tensor *restrict norm_w,
                             float *restrict x_norm_buf,
                             int8_t *restrict x_quant_buf);
void neon_row_worker(int tid, int n_threads, void *arg);
void neon_ssm_inner(const ssm_inner_args_t *a, int idx_start, int idx_end);
