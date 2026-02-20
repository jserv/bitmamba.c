/*
 * Metal GPU acceleration for BitMamba.
 *
 * Targets batched prefill only (bitlinear_forward_batch).
 * Single-token decode stays on CPU (GPU launch overhead > kernel runtime).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Initialize Metal device and compile shaders. Returns false on failure. */
bool bm_metal_init(void);

/* Check if Metal was successfully initialized. */
bool bm_metal_available(void);

/* Register the mmap'd model region for zero-copy GPU access.
 * Creates a single Metal buffer wrapping the entire mmap, eliminating
 * per-tensor copies (~200MB saved for 1B model on unified memory).
 * Returns false if buffer creation fails (caller should fall back to CPU). */
bool bm_metal_set_mmap(const void *base, size_t size);

/*
 * GPU-accelerated batched BitLinear forward pass (standalone, per-call).
 * Fused RMSNorm + quantize + direct ternary dot product on Metal compute
 * shaders.
 */
void bm_metal_bitlinear_forward_batch(const float *x_batch,
                                      int n_tokens,
                                      int n,
                                      const uint8_t *packed_data,
                                      int rows,
                                      int cols,
                                      float w_scale,
                                      const float *norm_weight,
                                      float *out_batch,
                                      float *x_norm_buf,
                                      int8_t *x_quant_batch,
                                      int quant_stride);

/* Release Metal resources. */
void bm_metal_cleanup(void);

/* === Batched prefill API ===
 *
 * Eliminates per-dispatch command buffer overhead by:
 * 1. Keeping activations in persistent Metal shared buffers (no memcpy)
 * 2. Batching out_proj + residual_add + next in_proj into one command buffer
 * 3. One submit+wait per layer instead of two
 *
 * CPU reads/writes Metal buffer contents pointers directly (unified memory).
 */

/* Buffer IDs for encode_bitlinear source/destination */
enum {
    BM_BUF_X = 0,    /* n_tokens * d_model floats */
    BM_BUF_PROJ = 1, /* n_tokens * max_proj floats */
    BM_BUF_Y = 2,    /* n_tokens * d_inner floats */
    BM_BUF_OUT = 3,  /* n_tokens * d_model floats */
};

typedef struct bm_metal_batch_ctx bm_metal_batch_ctx_t;

/* Create batch context with persistent Metal shared buffers.
 * Returns NULL on failure (caller should fall back to CPU). */
bm_metal_batch_ctx_t *bm_metal_batch_begin(int n_tokens,
                                           int d_model,
                                           int max_proj,
                                           int d_inner,
                                           int quant_stride);

/* Get CPU-accessible pointer to a Metal shared buffer. */
float *bm_metal_batch_ptr(bm_metal_batch_ctx_t *ctx, int buf_id);

/* Encode RMSNorm + quantize + bitlinear dot product into current
 * command buffer.  Does NOT submit -- call submit_and_wait when ready.
 * Returns false on error (bounds check, invalid dims). */
bool bm_metal_encode_bitlinear(bm_metal_batch_ctx_t *ctx,
                               int src_id,
                               int input_dim,
                               const uint8_t *packed_data,
                               int rows,
                               int cols,
                               float w_scale,
                               const float *norm_weight,
                               int dst_id);

/* Encode elementwise residual addition: x_buf[i] += out_buf[i]. */
void bm_metal_encode_residual(bm_metal_batch_ctx_t *ctx, int count);

/* Commit current command buffer and wait for GPU completion.
 * Creates a fresh command buffer for subsequent encodes. */
void bm_metal_submit_and_wait(bm_metal_batch_ctx_t *ctx);

/* Release batch context and its Metal buffers. */
void bm_metal_batch_end(bm_metal_batch_ctx_t *ctx);
