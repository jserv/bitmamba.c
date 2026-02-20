#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>

#include "bitmamba.h"
#include "dispatch.h"
#include "threadpool.h"

#if defined(__APPLE__) && !defined(BM_NO_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE 1
#endif

/*
 * Dot product abstraction: 16 x int8 dot product accumulated into int32x4.
 * With DOTPROD (Apple Silicon, Cortex-A76+): single instruction.
 * Without: vmull_s8 + vpaddlq_s16 fallback (all ARMv8-A).
 * Pattern from llama.cpp ggml-cpu-impl.h.
 */
#ifdef __ARM_FEATURE_DOTPROD
#define bm_vdotq_s32(acc, a, b) vdotq_s32(acc, a, b)
#else
static inline int32x4_t bm_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b)
{
    int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    /* vpaddq_s32 groups as (0+1,2+3,4+5,6+7) matching vdotq_s32 lane semantics.
     * vaddq_s32 would group as (0+4,1+5,2+6,3+7) -- wrong per-lane. */
    return vaddq_s32(acc, vpaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}
#endif

/*
 * Decode packed 2-bit ternary weights and dot-product against activations.
 * wb_expr: expression yielding uint8x16_t (evaluated once)
 * ac: accumulator (int32x4_t, modified in place)
 * act: int8x16x4_t deinterleaved activations
 *
 * Requires mask3, two, one_s8 locals in the calling scope.
 */
#define DECODE_DOT(wb_expr, ac, act)                                           \
    do {                                                                       \
        uint8x16_t _wb = (wb_expr);                                            \
        int8x16_t _s0 = vsubq_s8(                                              \
            vreinterpretq_s8_u8(vminq_u8(vandq_u8(_wb, mask3), two)), one_s8); \
        int8x16_t _s1 =                                                        \
            vsubq_s8(vreinterpretq_s8_u8(                                      \
                         vminq_u8(vandq_u8(vshrq_n_u8(_wb, 2), mask3), two)),  \
                     one_s8);                                                  \
        int8x16_t _s2 =                                                        \
            vsubq_s8(vreinterpretq_s8_u8(                                      \
                         vminq_u8(vandq_u8(vshrq_n_u8(_wb, 4), mask3), two)),  \
                     one_s8);                                                  \
        int8x16_t _s3 = vsubq_s8(                                              \
            vreinterpretq_s8_u8(vminq_u8(vshrq_n_u8(_wb, 6), two)), one_s8);   \
        (ac) = bm_vdotq_s32((ac), _s0, (act).val[0]);                          \
        (ac) = bm_vdotq_s32((ac), _s1, (act).val[1]);                          \
        (ac) = bm_vdotq_s32((ac), _s2, (act).val[2]);                          \
        (ac) = bm_vdotq_s32((ac), _s3, (act).val[3]);                          \
    } while (0)

float neon_rms_norm(const float *restrict x,
                    int size,
                    const float *restrict weight,
                    float *restrict out)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int i = 0;

    /* 16 floats per iteration, 4-way accumulator for ILP */
    for (; i <= size - 16; i += 16) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8);
        float32x4_t v3 = vld1q_f32(x + i + 12);
        acc0 = vfmaq_f32(acc0, v0, v0);
        acc1 = vfmaq_f32(acc1, v1, v1);
        acc2 = vfmaq_f32(acc2, v2, v2);
        acc3 = vfmaq_f32(acc3, v3, v3);
    }
    for (; i <= size - 4; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc0 = vfmaq_f32(acc0, v, v);
    }
    float sum_sq =
        vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < size; i++)
        sum_sq += x[i] * x[i];

    float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);
    float32x4_t vrms = vdupq_n_f32(rms);

    /* Scaling with fused max-abs tracking */
    float32x4_t vma = vdupq_n_f32(0.0f);
    for (i = 0; i <= size - 4; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);
        float32x4_t res = vmulq_f32(vmulq_f32(vx, vw), vrms);
        vst1q_f32(out + i, res);
        vma = vmaxq_f32(vma, vabsq_f32(res));
    }
    float max_abs = vmaxvq_f32(vma);
    for (; i < size; i++) {
        float v = x[i] * rms * weight[i];
        out[i] = v;
        v = fabsf(v);
        if (v > max_abs)
            max_abs = v;
    }
    return max_abs;
}

/*
 * Process rows [r_start, r_end) for NEON bitlinear forward.
 * Extracted helper for use by both static-partition and work-stealing paths.
 */
static void neon_row_block(bitlinear_row_work_t *w, int r_start, int r_end)
{
    if (r_start >= r_end)
        return;

    const int8_t *x_quant_buf = w->x_quant;
    const uint8_t *packed_ptr = w->packed_data;
    float inv_scale = w->inv_scale;
    int n_groups = w->cols / 4;
    int packed_stride = w->packed_stride;

    uint8x16_t mask3 = vdupq_n_u8(3);
    uint8x16_t two = vdupq_n_u8(2);
    int8x16_t one_s8 = vdupq_n_s8(1);

    int r = r_start;

    /* 8-row blocks */
    for (; r + 8 <= r_end; r += 8) {
        const uint8_t *rp0 = packed_ptr + (r + 0) * packed_stride;
        const uint8_t *rp1 = packed_ptr + (r + 1) * packed_stride;
        const uint8_t *rp2 = packed_ptr + (r + 2) * packed_stride;
        const uint8_t *rp3 = packed_ptr + (r + 3) * packed_stride;
        const uint8_t *rp4 = packed_ptr + (r + 4) * packed_stride;
        const uint8_t *rp5 = packed_ptr + (r + 5) * packed_stride;
        const uint8_t *rp6 = packed_ptr + (r + 6) * packed_stride;
        const uint8_t *rp7 = packed_ptr + (r + 7) * packed_stride;

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);

        int g = 0;
        for (; g + 16 <= n_groups; g += 16) {
            int8x16x4_t act = vld4q_s8(x_quant_buf + g * 4);
            DECODE_DOT(vld1q_u8(rp0 + g), acc0, act);
            DECODE_DOT(vld1q_u8(rp1 + g), acc1, act);
            DECODE_DOT(vld1q_u8(rp2 + g), acc2, act);
            DECODE_DOT(vld1q_u8(rp3 + g), acc3, act);
            DECODE_DOT(vld1q_u8(rp4 + g), acc4, act);
            DECODE_DOT(vld1q_u8(rp5 + g), acc5, act);
            DECODE_DOT(vld1q_u8(rp6 + g), acc6, act);
            DECODE_DOT(vld1q_u8(rp7 + g), acc7, act);
        }

        int32_t tot0 = vaddvq_s32(acc0);
        int32_t tot1 = vaddvq_s32(acc1);
        int32_t tot2 = vaddvq_s32(acc2);
        int32_t tot3 = vaddvq_s32(acc3);
        int32_t tot4 = vaddvq_s32(acc4);
        int32_t tot5 = vaddvq_s32(acc5);
        int32_t tot6 = vaddvq_s32(acc6);
        int32_t tot7 = vaddvq_s32(acc7);

        for (; g < n_groups; g++) {
            const int8_t *ap = x_quant_buf + g * 4;
            int8_t a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
            uint8_t b;
            b = rp0[g];
            tot0 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp1[g];
            tot1 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp2[g];
            tot2 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp3[g];
            tot3 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp4[g];
            tot4 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp5[g];
            tot5 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp6[g];
            tot6 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp7[g];
            tot7 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
        }

        w->out[r + 0] = (float) tot0 * inv_scale;
        w->out[r + 1] = (float) tot1 * inv_scale;
        w->out[r + 2] = (float) tot2 * inv_scale;
        w->out[r + 3] = (float) tot3 * inv_scale;
        w->out[r + 4] = (float) tot4 * inv_scale;
        w->out[r + 5] = (float) tot5 * inv_scale;
        w->out[r + 6] = (float) tot6 * inv_scale;
        w->out[r + 7] = (float) tot7 * inv_scale;
    }

    /* 4-row remainder */
    for (; r + 4 <= r_end; r += 4) {
        const uint8_t *rp0 = packed_ptr + (r + 0) * packed_stride;
        const uint8_t *rp1 = packed_ptr + (r + 1) * packed_stride;
        const uint8_t *rp2 = packed_ptr + (r + 2) * packed_stride;
        const uint8_t *rp3 = packed_ptr + (r + 3) * packed_stride;

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);

        int g = 0;
        for (; g + 16 <= n_groups; g += 16) {
            int8x16x4_t act = vld4q_s8(x_quant_buf + g * 4);
            DECODE_DOT(vld1q_u8(rp0 + g), acc0, act);
            DECODE_DOT(vld1q_u8(rp1 + g), acc1, act);
            DECODE_DOT(vld1q_u8(rp2 + g), acc2, act);
            DECODE_DOT(vld1q_u8(rp3 + g), acc3, act);
        }

        int32_t tot0 = vaddvq_s32(acc0);
        int32_t tot1 = vaddvq_s32(acc1);
        int32_t tot2 = vaddvq_s32(acc2);
        int32_t tot3 = vaddvq_s32(acc3);

        for (; g < n_groups; g++) {
            const int8_t *ap = x_quant_buf + g * 4;
            int8_t a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
            uint8_t b;
            b = rp0[g];
            tot0 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp1[g];
            tot1 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp2[g];
            tot2 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
            b = rp3[g];
            tot3 += UNPACK_LUT[b][0] * a0 + UNPACK_LUT[b][1] * a1 +
                    UNPACK_LUT[b][2] * a2 + UNPACK_LUT[b][3] * a3;
        }

        w->out[r + 0] = (float) tot0 * inv_scale;
        w->out[r + 1] = (float) tot1 * inv_scale;
        w->out[r + 2] = (float) tot2 * inv_scale;
        w->out[r + 3] = (float) tot3 * inv_scale;
    }

    /* Single-row remainder */
    for (; r < r_end; r++) {
        const uint8_t *row_ptr = packed_ptr + r * packed_stride;
        int32x4_t acc = vdupq_n_s32(0);
        int g = 0;

        for (; g + 16 <= n_groups; g += 16) {
            int8x16x4_t act = vld4q_s8(x_quant_buf + g * 4);
            DECODE_DOT(vld1q_u8(row_ptr + g), acc, act);
        }

        int32_t total = vaddvq_s32(acc);
        for (; g < n_groups; g++) {
            uint8_t b = row_ptr[g];
            const int8_t *ap = x_quant_buf + g * 4;
            total += UNPACK_LUT[b][0] * ap[0] + UNPACK_LUT[b][1] * ap[1] +
                     UNPACK_LUT[b][2] * ap[2] + UNPACK_LUT[b][3] * ap[3];
        }

        w->out[r] = (float) total * inv_scale;
    }
}

/*
 * Row-parallel worker for NEON bitlinear forward.
 * Supports two modes:
 *   - Static partition (chunk_ctr == NULL): bm_work_range for NUMA locality.
 *   - Work-stealing (chunk_ctr != NULL): atomic counter for load balancing.
 */
void neon_row_worker(int tid, int n_threads, void *arg)
{
    bitlinear_row_work_t *w = (bitlinear_row_work_t *) arg;

    if (w->chunk_ctr) {
        /* Work-stealing: grab chunks via atomic counter */
        int total_chunks = (w->rows + w->chunk_size - 1) / w->chunk_size;
        int chunk = tid; /* contention-free first access */
        while (chunk < total_chunks) {
            int s = chunk * w->chunk_size;
            int e = s + w->chunk_size;
            if (e > w->rows)
                e = w->rows;
            neon_row_block(w, s, e);
            chunk = atomic_fetch_add_explicit(w->chunk_ctr, 1,
                                              memory_order_relaxed);
        }
    } else {
        /* Static partition */
        int r_start, r_end;
        bm_work_range(tid, n_threads, w->rows, &r_start, &r_end);
        neon_row_block(w, r_start, r_end);
    }
}

/*
 * Decomposed bitlinear prepare: RMS normalize + quantize to int8.
 * Returns scale_x; caller computes inv_scale = 1/(scale_x * w->scale).
 */
float neon_bitlinear_prepare(const float *restrict x,
                             int n,
                             const tensor_t *restrict norm_w,
                             float *restrict x_norm_buf,
                             int8_t *restrict x_quant_buf)
{
    float max_abs = neon_rms_norm(x, n, norm_w->data, x_norm_buf);
    float scale_x = 127.0f / (max_abs + 1e-5f);
    int i;

    /* Quantize to int8 via NEON narrowing */
    for (i = 0; i <= n - 16; i += 16) {
        float32x4_t f0 = vmulq_n_f32(vld1q_f32(x_norm_buf + i), scale_x);
        float32x4_t f1 = vmulq_n_f32(vld1q_f32(x_norm_buf + i + 4), scale_x);
        float32x4_t f2 = vmulq_n_f32(vld1q_f32(x_norm_buf + i + 8), scale_x);
        float32x4_t f3 = vmulq_n_f32(vld1q_f32(x_norm_buf + i + 12), scale_x);

        /* Round to nearest, ties away from zero (matches scalar path) */
        int32x4_t i0 = vcvtaq_s32_f32(f0);
        int32x4_t i1 = vcvtaq_s32_f32(f1);
        int32x4_t i2 = vcvtaq_s32_f32(f2);
        int32x4_t i3 = vcvtaq_s32_f32(f3);

        /* Saturating narrow: s32 -> s16 -> s8 */
        int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        int8x16_t q = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
        vst1q_s8(x_quant_buf + i, q);
    }
    for (; i < n; i++) {
        float val = x_norm_buf[i] * scale_x;
        int ival = (int) (val + (val >= 0 ? 0.5f : -0.5f));
        x_quant_buf[i] =
            (int8_t) (ival > 127 ? 127 : (ival < -128 ? -128 : ival));
    }

    return scale_x;
}

/*
 * Batched row worker: each thread processes the same row range across all
 * tokens (weight-stationary). Weight bytes for each thread's rows stay in
 * L2 cache across the token loop, eliminating redundant DRAM fetches.
 */
typedef struct {
    const int8_t *x_quant_batch;
    int quant_stride;
    const uint8_t *packed_data;
    float *out_batch;
    float *inv_scales; /* [n_tokens] pre-computed */
    int n_tokens;
    int out_dim;
    int cols;
    int packed_stride;
} neon_batch_row_work_t;

static void neon_batch_row_worker(int tid, int n_threads, void *arg)
{
    neon_batch_row_work_t *bw = (neon_batch_row_work_t *) arg;
    int r_start, r_end;
    bm_work_range(tid, n_threads, bw->out_dim, &r_start, &r_end);
    if (r_start >= r_end)
        return;

    int n_rows = r_end - r_start;
    for (int t = 0; t < bw->n_tokens; t++) {
        bitlinear_row_work_t single = {
            .x_quant = bw->x_quant_batch + t * bw->quant_stride,
            .packed_data =
                bw->packed_data + (size_t) r_start * bw->packed_stride,
            .out = bw->out_batch + t * bw->out_dim + r_start,
            .inv_scale = bw->inv_scales[t],
            .rows = n_rows,
            .cols = bw->cols,
            .packed_stride = bw->packed_stride,
        };
        neon_row_worker(0, 1, &single);
    }
}

/*
 * Batched BitLinear forward pass for prefill (NEON).
 *
 * Phase 1: normalize + quantize all tokens upfront.
 * Phase 2: single-dispatch weight-stationary row work across all tokens.
 * One fork-join for all tokens instead of per-token dispatch.
 */
void neon_bitlinear_forward_batch(const float *restrict x_batch,
                                  int n_tokens,
                                  int n,
                                  const tensor_t *restrict w,
                                  const tensor_t *restrict norm_w,
                                  float *restrict out_batch,
                                  float *restrict x_norm_buf,
                                  int8_t *restrict x_quant_batch,
                                  int quant_stride)
{
    BITLINEAR_CHECK(n, w);

    int out_dim = w->rows;

    /* Phase 1: normalize + quantize all tokens */
    float scale_x_arr[1024];
    float *scale_x_heap = NULL;
    float *scale_x = scale_x_arr;
    if (n_tokens > 1024) {
        scale_x_heap = xmalloc(n_tokens * sizeof(float));
        scale_x = scale_x_heap;
    }

    for (int t = 0; t < n_tokens; t++) {
        int8_t *xq = x_quant_batch + t * quant_stride;
        scale_x[t] =
            neon_bitlinear_prepare(x_batch + t * n, n, norm_w, x_norm_buf, xq);
    }

    /* Convert scale_x to inv_scales in place (values no longer needed) */
    float w_scale = w->scale;
    for (int t = 0; t < n_tokens; t++)
        scale_x[t] = 1.0f / (scale_x[t] * w_scale);

    /* Phase 2: single-dispatch weight-stationary row work */
    int cols = w->cols;
    int packed_stride = (cols + 3) / 4;
    const uint8_t *packed_ptr = w->packed_data;

    if (out_dim >= BM_PAR_THRESHOLD && bm_get_threads() > 1 && n_tokens > 1) {
        neon_batch_row_work_t bw = {
            .x_quant_batch = x_quant_batch,
            .quant_stride = quant_stride,
            .packed_data = packed_ptr,
            .out_batch = out_batch,
            .inv_scales = scale_x,
            .n_tokens = n_tokens,
            .out_dim = out_dim,
            .cols = cols,
            .packed_stride = packed_stride,
        };
        bm_parallel_for(out_dim, neon_batch_row_worker, &bw);
    } else {
        for (int t = 0; t < n_tokens; t++) {
            bitlinear_row_work_t work = {
                .x_quant = x_quant_batch + t * quant_stride,
                .packed_data = packed_ptr,
                .out = out_batch + t * out_dim,
                .inv_scale = scale_x[t],
                .rows = out_dim,
                .cols = cols,
                .packed_stride = packed_stride,
            };
            neon_row_worker(0, 1, &work);
        }
    }

    /* Handle remaining columns (cols not divisible by 4) */
    int n_groups = cols / 4;
    int remainder_start = n_groups * 4;
    if (remainder_start < cols) {
        for (int r2 = 0; r2 < out_dim; r2++) {
            const uint8_t *row_ptr = packed_ptr + r2 * packed_stride + n_groups;
            for (int t = 0; t < n_tokens; t++) {
                const int8_t *act_ptr =
                    x_quant_batch + t * quant_stride + remainder_start;
                float inv_scale = scale_x[t];
                int32_t total = 0;
                const uint8_t *rp = row_ptr;
                for (int c = remainder_start; c < cols; c++) {
                    int8_t w_val = UNPACK_LUT[*rp][(c & 3)];
                    total += w_val * *act_ptr;
                    act_ptr++;
                    if ((c & 3) == 3)
                        rp++;
                }
                out_batch[t * out_dim + r2] += (float) total * inv_scale;
            }
        }
    }

    free(scale_x_heap);
}

#ifndef USE_ACCELERATE
/*
 * Vectorized expf for NEON (float32x4_t).
 * Polynomial approximation ported from llama.cpp ggml.c.
 * Max relative error ~1.5e-5 over [-87, 88], sufficient for SSM scan.
 * Used only in the non-Accelerate SSM path (bm_v_silu_nr).
 */
static inline __attribute__((always_inline)) float32x4_t
bm_v_expf(float32x4_t x)
{
    /* Clamp to prevent overflow/underflow in integer conversion */
    const float32x4_t max_val = vdupq_n_f32(88.0f);
    const float32x4_t min_val = vdupq_n_f32(-87.33654475f);
    x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    /* exp(x) = 2^(x / ln2) = 2^(n + r) where n = floor(x/ln2), r = frac */
    const float32x4_t log2e = vdupq_n_f32(1.44269504089f);
    const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);

    float32x4_t t = vmulq_f32(x, log2e);
    float32x4_t n_f = vrndmq_f32(t);        /* floor */
    float32x4_t r = vfmsq_f32(x, n_f, ln2); /* x - n * ln2 */

    /* Polynomial approximation of exp(r) for r in [0, ln2)
     * Horner's method: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120)))) */
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.16666666666f);
    const float32x4_t c4 = vdupq_n_f32(0.04166666666f);
    const float32x4_t c5 = vdupq_n_f32(0.00833333333f);

    float32x4_t p = vfmaq_f32(c4, c5, r);
    p = vfmaq_f32(c3, p, r);
    p = vfmaq_f32(c2, p, r);
    p = vfmaq_f32(c1, p, r);
    p = vfmaq_f32(c1, p, r);

    /* Multiply by 2^n: construct IEEE float with exponent = n+127 */
    int32x4_t n_i = vcvtq_s32_f32(n_f);
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);

    return vmulq_f32(p, pow2n);
}

/*
 * Vectorized silu(x) = x / (1 + exp(-x)) for NEON.
 * Uses Newton-Raphson reciprocal refinement for precision.
 * vrecpeq alone gives ~0.4% error; one NR step brings it to ~0.001%.
 */
static inline __attribute__((always_inline)) float32x4_t
bm_v_silu_nr(float32x4_t x)
{
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t exp_neg = bm_v_expf(vnegq_f32(x));
    float32x4_t denom = vaddq_f32(one, exp_neg);
    /* Newton-Raphson reciprocal: r = r * (2 - d*r) */
    float32x4_t r = vrecpeq_f32(denom);
    r = vmulq_f32(r, vrecpsq_f32(denom, r));
    return vmulq_f32(x, r);
}
#endif /* !USE_ACCELERATE */

/*
 * NEON vectorized SSM inner loop.
 *
 * With Accelerate (Apple Silicon): two-pass design batches all exp() calls
 * via vvexpf, plus vvlog1pf for per-head softplus. This replaces the inline
 * polynomial approximation with Apple's hardware-tuned implementations.
 *   Pass 1: conv1d for all elements + collect negated values for batch exp.
 *   Batch:  vvexpf for silu(conv) and silu(z).
 *   Pass 2: per-head SSM state update using pre-computed silu results.
 * Disable with BM_NO_ACCELERATE=1 at build time.
 *
 * Without Accelerate: single-pass with inline polynomial exp (bm_v_silu_nr).
 * Double-pumped ILP (8 elements/iter) hides FMA latency.
 */
void neon_ssm_inner(const ssm_inner_args_t *a, int idx_start, int idx_end)
{
    int head_dim = a->head_dim;
    int d_inner = a->d_inner;

    const float *cw_tap0 = a->conv1d_w_t;
    const float *cw_tap1 = a->conv1d_w_t + d_inner;
    const float *cw_tap2 = a->conv1d_w_t + 2 * d_inner;
    const float *cw_tap3 = a->conv1d_w_t + 3 * d_inner;

#ifdef USE_ACCELERATE
    int n = idx_end - idx_start;
    if (n <= 0)
        return;

    /* --- Pre-compute per-head scalars via batch Accelerate ---
     * softplus(x) = log(1 + exp(x)) via vvexpf + vvlog1pf.
     * decay = exp(A_precomp * dt_val) via vvexpf.
     * Batches all heads in [idx_start, idx_end) in 3 Accelerate calls. */
    int h_first = idx_start / head_dim;
    int h_last = (idx_end - 1) / head_dim + 1;
    int nh = h_last - h_first;
    float dt_raw[nh], sp_buf[nh], decay_buf[nh], bdt_buf[nh], c_buf[nh];

    for (int i = 0; i < nh; i++)
        dt_raw[i] = a->ptr_dt[h_first + i] + a->dt_bias[h_first + i];

    vvexpf(sp_buf, dt_raw, &nh);
    vvlog1pf(sp_buf, sp_buf, &nh);
    for (int i = 0; i < nh; i++) {
        if (dt_raw[i] > 20.0f)
            sp_buf[i] = dt_raw[i]; /* overflow guard */
    }

    for (int i = 0; i < nh; i++) {
        int h = h_first + i;
        decay_buf[i] = a->A_precomp[h] * sp_buf[i];
        bdt_buf[i] = a->ptr_B[h] * sp_buf[i];
        c_buf[i] = a->ptr_C[h];
    }
    vvexpf(decay_buf, decay_buf, &nh);

    /* --- Pass 1: conv1d + negate for batch exp ---
     * Head-independent: flat NEON sweep over [idx_start, idx_end). */
    float conv_buf[n]; /* conv1d results (silu numerator) */
    float exp_conv[n]; /* -> exp(-conv) -> silu(conv) */
    float exp_z[n];    /* -> exp(-z)    -> silu(z) */

    int d = idx_start;
    int off = 0;

    for (; off + 8 <= n; off += 8, d += 8) {
        float32x4_t c_a = vld1q_f32(a->conv1d_b + d);
        float32x4_t c_b = vld1q_f32(a->conv1d_b + d + 4);

        c_a = vfmaq_f32(c_a, vld1q_f32(a->cs0 + d), vld1q_f32(cw_tap0 + d));
        c_b = vfmaq_f32(c_b, vld1q_f32(a->cs0 + d + 4),
                        vld1q_f32(cw_tap0 + d + 4));
        c_a = vfmaq_f32(c_a, vld1q_f32(a->cs1 + d), vld1q_f32(cw_tap1 + d));
        c_b = vfmaq_f32(c_b, vld1q_f32(a->cs1 + d + 4),
                        vld1q_f32(cw_tap1 + d + 4));
        c_a = vfmaq_f32(c_a, vld1q_f32(a->cs2 + d), vld1q_f32(cw_tap2 + d));
        c_b = vfmaq_f32(c_b, vld1q_f32(a->cs2 + d + 4),
                        vld1q_f32(cw_tap2 + d + 4));

        float32x4_t in_a = vld1q_f32(a->ptr_x + d);
        float32x4_t in_b = vld1q_f32(a->ptr_x + d + 4);
        c_a = vfmaq_f32(c_a, in_a, vld1q_f32(cw_tap3 + d));
        c_b = vfmaq_f32(c_b, in_b, vld1q_f32(cw_tap3 + d + 4));

        vst1q_f32(a->cs_write + d, in_a);
        vst1q_f32(a->cs_write + d + 4, in_b);

        vst1q_f32(conv_buf + off, c_a);
        vst1q_f32(conv_buf + off + 4, c_b);
        vst1q_f32(exp_conv + off, vnegq_f32(c_a));
        vst1q_f32(exp_conv + off + 4, vnegq_f32(c_b));
        vst1q_f32(exp_z + off, vnegq_f32(vld1q_f32(a->ptr_z + d)));
        vst1q_f32(exp_z + off + 4, vnegq_f32(vld1q_f32(a->ptr_z + d + 4)));
    }

    for (; off + 4 <= n; off += 4, d += 4) {
        float32x4_t c = vld1q_f32(a->conv1d_b + d);
        c = vfmaq_f32(c, vld1q_f32(a->cs0 + d), vld1q_f32(cw_tap0 + d));
        c = vfmaq_f32(c, vld1q_f32(a->cs1 + d), vld1q_f32(cw_tap1 + d));
        c = vfmaq_f32(c, vld1q_f32(a->cs2 + d), vld1q_f32(cw_tap2 + d));
        float32x4_t in = vld1q_f32(a->ptr_x + d);
        c = vfmaq_f32(c, in, vld1q_f32(cw_tap3 + d));
        vst1q_f32(a->cs_write + d, in);
        vst1q_f32(conv_buf + off, c);
        vst1q_f32(exp_conv + off, vnegq_f32(c));
        vst1q_f32(exp_z + off, vnegq_f32(vld1q_f32(a->ptr_z + d)));
    }

    for (; off < n; off++, d++) {
        float input_x = a->ptr_x[d];
        float cv = a->conv1d_b[d] + a->cs0[d] * cw_tap0[d] +
                   a->cs1[d] * cw_tap1[d] + a->cs2[d] * cw_tap2[d] +
                   input_x * cw_tap3[d];
        a->cs_write[d] = input_x;
        conv_buf[off] = cv;
        exp_conv[off] = -cv;
        exp_z[off] = -a->ptr_z[d];
    }

    /* --- Batch exp via Accelerate vvexpf --- */
    vvexpf(exp_conv, exp_conv, &n);
    vvexpf(exp_z, exp_z, &n);

    /* --- Compute silu = x / (1 + exp(-x)) ---
     * Apple Silicon vdivq_f32 has 1-cycle throughput; true divide is
     * both faster and more accurate than NR reciprocal on M-series. */
    float32x4_t v_one = vdupq_n_f32(1.0f);
    off = 0;
    d = idx_start;
    for (; off + 4 <= n; off += 4, d += 4) {
        float32x4_t dc = vaddq_f32(v_one, vld1q_f32(exp_conv + off));
        vst1q_f32(exp_conv + off, vdivq_f32(vld1q_f32(conv_buf + off), dc));
        float32x4_t dz = vaddq_f32(v_one, vld1q_f32(exp_z + off));
        vst1q_f32(exp_z + off, vdivq_f32(vld1q_f32(a->ptr_z + d), dz));
    }
    for (; off < n; off++, d++) {
        exp_conv[off] = conv_buf[off] / (1.0f + exp_conv[off]);
        exp_z[off] = a->ptr_z[d] / (1.0f + exp_z[off]);
    }
    /* exp_conv[] = silu(conv), exp_z[] = silu(z) */

    /* --- Pass 2: per-head SSM state update --- */
    int idx = idx_start;
    while (idx < idx_end) {
        int h = idx / head_dim;
        int head_end = (h + 1) * head_dim;
        if (head_end > idx_end)
            head_end = idx_end;

        int hi = h - h_first;
        float decay_s = decay_buf[hi];
        float Bdt_s = bdt_buf[hi];
        float C_s = c_buf[hi];

        float32x4_t v_decay = vdupq_n_f32(decay_s);
        float32x4_t v_Bdt = vdupq_n_f32(Bdt_s);
        float32x4_t v_C = vdupq_n_f32(C_s);

        d = idx;
        off = d - idx_start;

        for (; d + 8 <= head_end; d += 8, off += 8) {
            float32x4_t xa_a = vld1q_f32(exp_conv + off);
            float32x4_t xa_b = vld1q_f32(exp_conv + off + 4);

            float32x4_t st_a = vld1q_f32(a->ssm_state + d);
            float32x4_t st_b = vld1q_f32(a->ssm_state + d + 4);
            st_a = vfmaq_f32(vmulq_f32(xa_a, v_Bdt), st_a, v_decay);
            st_b = vfmaq_f32(vmulq_f32(xa_b, v_Bdt), st_b, v_decay);
            vst1q_f32(a->ssm_state + d, st_a);
            vst1q_f32(a->ssm_state + d + 4, st_b);

            float32x4_t za_a = vld1q_f32(exp_z + off);
            float32x4_t za_b = vld1q_f32(exp_z + off + 4);
            vst1q_f32(a->y + d,
                      vmulq_f32(vfmaq_f32(vmulq_f32(xa_a, vld1q_f32(a->D + d)),
                                          st_a, v_C),
                                za_a));
            vst1q_f32(
                a->y + d + 4,
                vmulq_f32(vfmaq_f32(vmulq_f32(xa_b, vld1q_f32(a->D + d + 4)),
                                    st_b, v_C),
                          za_b));
        }

        for (; d + 4 <= head_end; d += 4, off += 4) {
            float32x4_t x_act = vld1q_f32(exp_conv + off);
            float32x4_t state = vld1q_f32(a->ssm_state + d);
            state = vfmaq_f32(vmulq_f32(x_act, v_Bdt), state, v_decay);
            vst1q_f32(a->ssm_state + d, state);

            float32x4_t z_act = vld1q_f32(exp_z + off);
            vst1q_f32(a->y + d,
                      vmulq_f32(vfmaq_f32(vmulq_f32(x_act, vld1q_f32(a->D + d)),
                                          state, v_C),
                                z_act));
        }

        for (; d < head_end; d++, off++) {
            float x_act = exp_conv[off];
            float state = a->ssm_state[d] * decay_s + x_act * Bdt_s;
            a->ssm_state[d] = state;
            a->y[d] = (state * C_s + x_act * a->D[d]) * exp_z[off];
        }

        idx = head_end;
    }

#else /* !USE_ACCELERATE: inline polynomial path */

    int idx = idx_start;

    while (idx < idx_end) {
        int h = idx / head_dim;
        int head_end = (h + 1) * head_dim;
        if (head_end > idx_end)
            head_end = idx_end;

        /* Per-head scalars (64x per token, not worth vectorizing) */
        float dt_val = softplus(a->ptr_dt[h] + a->dt_bias[h]);
        float decay_s = expf(a->A_precomp[h] * dt_val);
        float C_s = a->ptr_C[h];
        float Bdt_s = a->ptr_B[h] * dt_val;

        float32x4_t v_decay = vdupq_n_f32(decay_s);
        float32x4_t v_Bdt = vdupq_n_f32(Bdt_s);
        float32x4_t v_C = vdupq_n_f32(C_s);

        int d = idx;

        /* Double-pumped: 8 floats/iter (two independent 4-wide chains)
         * to hide FMA latency (~4-5 cycles) via ILP */
        for (; d + 8 <= head_end; d += 8) {
            float32x4_t conv_a = vld1q_f32(a->conv1d_b + d);
            float32x4_t conv_b = vld1q_f32(a->conv1d_b + d + 4);

            conv_a = vfmaq_f32(conv_a, vld1q_f32(a->cs0 + d),
                               vld1q_f32(cw_tap0 + d));
            conv_b = vfmaq_f32(conv_b, vld1q_f32(a->cs0 + d + 4),
                               vld1q_f32(cw_tap0 + d + 4));
            conv_a = vfmaq_f32(conv_a, vld1q_f32(a->cs1 + d),
                               vld1q_f32(cw_tap1 + d));
            conv_b = vfmaq_f32(conv_b, vld1q_f32(a->cs1 + d + 4),
                               vld1q_f32(cw_tap1 + d + 4));
            conv_a = vfmaq_f32(conv_a, vld1q_f32(a->cs2 + d),
                               vld1q_f32(cw_tap2 + d));
            conv_b = vfmaq_f32(conv_b, vld1q_f32(a->cs2 + d + 4),
                               vld1q_f32(cw_tap2 + d + 4));

            float32x4_t in_a = vld1q_f32(a->ptr_x + d);
            float32x4_t in_b = vld1q_f32(a->ptr_x + d + 4);
            conv_a = vfmaq_f32(conv_a, in_a, vld1q_f32(cw_tap3 + d));
            conv_b = vfmaq_f32(conv_b, in_b, vld1q_f32(cw_tap3 + d + 4));

            vst1q_f32(a->cs_write + d, in_a);
            vst1q_f32(a->cs_write + d + 4, in_b);

            float32x4_t xa_a = bm_v_silu_nr(conv_a);
            float32x4_t xa_b = bm_v_silu_nr(conv_b);

            float32x4_t st_a = vld1q_f32(a->ssm_state + d);
            float32x4_t st_b = vld1q_f32(a->ssm_state + d + 4);
            st_a = vfmaq_f32(vmulq_f32(xa_a, v_Bdt), st_a, v_decay);
            st_b = vfmaq_f32(vmulq_f32(xa_b, v_Bdt), st_b, v_decay);
            vst1q_f32(a->ssm_state + d, st_a);
            vst1q_f32(a->ssm_state + d + 4, st_b);

            float32x4_t za_a = bm_v_silu_nr(vld1q_f32(a->ptr_z + d));
            float32x4_t za_b = bm_v_silu_nr(vld1q_f32(a->ptr_z + d + 4));
            vst1q_f32(a->y + d,
                      vmulq_f32(vfmaq_f32(vmulq_f32(xa_a, vld1q_f32(a->D + d)),
                                          st_a, v_C),
                                za_a));
            vst1q_f32(
                a->y + d + 4,
                vmulq_f32(vfmaq_f32(vmulq_f32(xa_b, vld1q_f32(a->D + d + 4)),
                                    st_b, v_C),
                          za_b));
        }

        /* 4-float remainder (head_dim not divisible by 8) */
        for (; d + 4 <= head_end; d += 4) {
            float32x4_t conv = vld1q_f32(a->conv1d_b + d);
            conv =
                vfmaq_f32(conv, vld1q_f32(a->cs0 + d), vld1q_f32(cw_tap0 + d));
            conv =
                vfmaq_f32(conv, vld1q_f32(a->cs1 + d), vld1q_f32(cw_tap1 + d));
            conv =
                vfmaq_f32(conv, vld1q_f32(a->cs2 + d), vld1q_f32(cw_tap2 + d));
            float32x4_t v_input = vld1q_f32(a->ptr_x + d);
            conv = vfmaq_f32(conv, v_input, vld1q_f32(cw_tap3 + d));
            vst1q_f32(a->cs_write + d, v_input);

            float32x4_t x_act = bm_v_silu_nr(conv);
            float32x4_t state = vld1q_f32(a->ssm_state + d);
            state = vfmaq_f32(vmulq_f32(x_act, v_Bdt), state, v_decay);
            vst1q_f32(a->ssm_state + d, state);

            float32x4_t z_act = bm_v_silu_nr(vld1q_f32(a->ptr_z + d));
            vst1q_f32(a->y + d,
                      vmulq_f32(vfmaq_f32(vmulq_f32(x_act, vld1q_f32(a->D + d)),
                                          state, v_C),
                                z_act));
        }

        /* Scalar remainder (only if head_dim % 4 != 0) */
        for (; d < head_end; d++) {
            float input_x = a->ptr_x[d];
            float conv_res = a->conv1d_b[d] + a->cs0[d] * cw_tap0[d] +
                             a->cs1[d] * cw_tap1[d] + a->cs2[d] * cw_tap2[d] +
                             input_x * cw_tap3[d];
            a->cs_write[d] = input_x;
            float x_act = silu(conv_res);
            float state = a->ssm_state[d] * decay_s + x_act * Bdt_s;
            a->ssm_state[d] = state;
            a->y[d] = (state * C_s + x_act * a->D[d]) * silu(a->ptr_z[d]);
        }

        idx = head_end;
    }

#endif /* USE_ACCELERATE */
}

/*
 * BitLinear forward pass (NEON).
 *
 * Direct ternary dot product via hardware int8 dot-product instructions.
 * Decodes packed 2-bit weights to int8 {-1,0,+1} and uses vdotq_s32
 * (or vmull_s8 fallback) against quantized activations.
 * 8-row tiling amortizes 64-byte activation loads across 8 DECODE_DOT ops.
 */
void neon_bitlinear_forward(const float *restrict x,
                            int n,
                            const tensor_t *restrict w,
                            const tensor_t *restrict norm_w,
                            float *restrict out,
                            float *restrict x_norm_buf,
                            int8_t *restrict x_quant_buf)
{
    BITLINEAR_CHECK(n, w);

    /* Steps 1+2: RMS normalize + quantize (decomposed) */
    float scale_x =
        neon_bitlinear_prepare(x, n, norm_w, x_norm_buf, x_quant_buf);

    /* Step 3: Ternary dot product via NEON -- dispatch rows in parallel. */
    int cols = w->cols;
    int packed_stride = (cols + 3) / 4;
    const uint8_t *packed_ptr = w->packed_data;
    float inv_scale = 1.0f / (scale_x * w->scale);

    /* Work-stealing: enabled on single-NUMA to balance stragglers.
     * Disabled on multi-NUMA where static partition preserves locality. */
    int n_threads = bm_get_threads();
    int n_rows = w->rows;
    int cs = n_rows / (n_threads * 4);
    if (cs < 8)
        cs = 8; /* align to NEON 8-row unroll */
    _Atomic int steal_ctr = n_threads;
    bool use_steal = (bm_get_numa_nodes() <= 1 && n_rows >= BM_PAR_THRESHOLD &&
                      n_threads > 1);

    bitlinear_row_work_t work = {
        .x_quant = x_quant_buf,
        .packed_data = packed_ptr,
        .out = out,
        .inv_scale = inv_scale,
        .rows = n_rows,
        .cols = cols,
        .packed_stride = packed_stride,
        .chunk_ctr = use_steal ? &steal_ctr : NULL,
        .chunk_size = cs,
    };

    if (n_rows >= BM_PAR_THRESHOLD && n_threads > 1)
        bm_parallel_for(n_rows, neon_row_worker, &work);
    else
        neon_row_worker(0, 1, &work);

    /* Handle remaining columns (cols not divisible by 4) */
    int n_groups = cols / 4;
    int remainder_start = n_groups * 4;
    if (remainder_start < cols) {
        for (int r = 0; r < w->rows; r++) {
            const uint8_t *row_ptr = packed_ptr + r * packed_stride + n_groups;
            const int8_t *act_ptr = x_quant_buf + remainder_start;
            int32_t total = 0;

            for (int c = remainder_start; c < cols; c++) {
                int8_t w_val = UNPACK_LUT[*row_ptr][(c & 3)];
                total += w_val * *act_ptr;
                act_ptr++;
                if ((c & 3) == 3)
                    row_ptr++;
            }
            out[r] += (float) total * inv_scale;
        }
    }
}
