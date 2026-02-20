#include <stdio.h>
#include <stdlib.h>

#include "bitmamba.h"
#include "dispatch.h"
#include "threadpool.h"

/*
 * T-MAC cache blocking: 256 groups * 16 entries * 4 bytes = 16KB (fits 32KB
 * L1).
 */
enum { TMAC_BLOCK_SIZE = 256 };

/* Persistent LUT buffer to avoid malloc/free per bitlinear call.
 * Set once at model load via scalar_set_lut_buf().
 *
 * Thread safety: scalar_bitlinear_forward builds LUTs in this buffer then
 * dispatches row workers that only read from it. The scalar path never runs
 * from multiple callers concurrently (no fused block path for scalar, and
 * batch mode calls sequentially). If that ever changes, this must become
 * per-thread or passed via the work struct. */
static int32_t (*g_scalar_lut_buf)[16] = NULL;

void scalar_set_lut_buf(int32_t *buf)
{
    g_scalar_lut_buf = (int32_t (*)[16]) buf;
}

/*
 * Scalar SSM inner loop: conv1d + selective state space + gated output.
 * Uses transposed conv1d weights [tap][d_inner] for future vectorization
 * compatibility, though the scalar path sees no layout benefit.
 */
void scalar_ssm_inner(const ssm_inner_args_t *a, int idx_start, int idx_end)
{
    int head_dim = a->head_dim;
    int d_inner = a->d_inner;
    int prev_h = -1;
    float decay = 0, Bdt = 0, C_val = 0;

    const float *cw_tap0 = a->conv1d_w_t;
    const float *cw_tap1 = a->conv1d_w_t + d_inner;
    const float *cw_tap2 = a->conv1d_w_t + 2 * d_inner;
    const float *cw_tap3 = a->conv1d_w_t + 3 * d_inner;

    for (int idx = idx_start; idx < idx_end; idx++) {
        int h = idx / head_dim;

        if (h != prev_h) {
            float dt_val = softplus(a->ptr_dt[h] + a->dt_bias[h]);
            decay = expf(a->A_precomp[h] * dt_val);
            C_val = a->ptr_C[h];
            Bdt = a->ptr_B[h] * dt_val;
            prev_h = h;
        }

        float input_x = a->ptr_x[idx];

        float conv_res = a->conv1d_b[idx] + a->cs0[idx] * cw_tap0[idx] +
                         a->cs1[idx] * cw_tap1[idx] +
                         a->cs2[idx] * cw_tap2[idx] + input_x * cw_tap3[idx];

        a->cs_write[idx] = input_x;

        float x_act = silu(conv_res);
        float state = a->ssm_state[idx] * decay + x_act * Bdt;
        a->ssm_state[idx] = state;
        a->y[idx] = (state * C_val + x_act * a->D[idx]) * silu(a->ptr_z[idx]);
    }
}

float scalar_rms_norm(const float *restrict x,
                      int size,
                      const float *restrict weight,
                      float *restrict out)
{
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int i = 0;

    /* 4x unrolled sum of squares with multiple accumulators for ILP */
    for (; i <= size - 4; i += 4) {
        sum0 += x[i] * x[i];
        sum1 += x[i + 1] * x[i + 1];
        sum2 += x[i + 2] * x[i + 2];
        sum3 += x[i + 3] * x[i + 3];
    }
    float sum_sq = sum0 + sum1 + sum2 + sum3;
    for (; i < size; i++)
        sum_sq += x[i] * x[i];

    float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);

    /* 4x unrolled scaling with fused max-abs tracking */
    float ma0 = 0.0f, ma1 = 0.0f, ma2 = 0.0f, ma3 = 0.0f;
    for (i = 0; i <= size - 4; i += 4) {
        float v0 = x[i] * rms * weight[i];
        float v1 = x[i + 1] * rms * weight[i + 1];
        float v2 = x[i + 2] * rms * weight[i + 2];
        float v3 = x[i + 3] * rms * weight[i + 3];
        out[i] = v0;
        out[i + 1] = v1;
        out[i + 2] = v2;
        out[i + 3] = v3;
        v0 = fabsf(v0);
        v1 = fabsf(v1);
        v2 = fabsf(v2);
        v3 = fabsf(v3);
        if (v0 > ma0)
            ma0 = v0;
        if (v1 > ma1)
            ma1 = v1;
        if (v2 > ma2)
            ma2 = v2;
        if (v3 > ma3)
            ma3 = v3;
    }
    float max_abs = ma0 > ma1 ? ma0 : ma1;
    float ma23 = ma2 > ma3 ? ma2 : ma3;
    if (ma23 > max_abs)
        max_abs = ma23;
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
 * Build T-MAC activation LUT for 4 consecutive int8 activations.
 * lut[mask] = sum of x[i] where bit i of mask is set.
 * 16 entries, 64 bytes - fits in one cache line.
 */
static inline void build_tmac_lut(const int8_t *x, int32_t *lut)
{
    int32_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
    int32_t x01 = x0 + x1, x23 = x2 + x3;

    lut[0] = 0;
    lut[1] = x0;
    lut[2] = x1;
    lut[3] = x01;
    lut[4] = x2;
    lut[5] = x0 + x2;
    lut[6] = x1 + x2;
    lut[7] = x01 + x2;
    lut[8] = x3;
    lut[9] = x0 + x3;
    lut[10] = x1 + x3;
    lut[11] = x01 + x3;
    lut[12] = x23;
    lut[13] = x0 + x23;
    lut[14] = x1 + x23;
    lut[15] = x01 + x23;
}

/*
 * Row-parallel worker for scalar T-MAC bitlinear.
 * Each thread processes rows [r_start, r_end) for one column block.
 */
typedef struct {
    const uint8_t *packed_ptr;
    int packed_stride;
    const int32_t (*act_luts)[16];
    int g_start;
    int block_groups;
    float *out;
    float inv_scale;
    int total_rows;
} scalar_row_work_t;

static void scalar_row_worker(int tid, int n_threads, void *arg)
{
    scalar_row_work_t *w = (scalar_row_work_t *) arg;
    int r_start, r_end;
    bm_work_range(tid, n_threads, w->total_rows, &r_start, &r_end);

    const uint8_t *packed_ptr = w->packed_ptr;
    int packed_stride = w->packed_stride;
    const int32_t (*act_luts)[16] = w->act_luts;
    int g_start = w->g_start;
    int block_groups = w->block_groups;
    float inv_scale = w->inv_scale;

    int r = r_start;
    for (; r + 2 <= r_end; r += 2) {
        const uint8_t *rp0 = packed_ptr + r * packed_stride + g_start;
        const uint8_t *rp1 = packed_ptr + (r + 1) * packed_stride + g_start;

        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        int g = 0;

        for (; g + 8 <= block_groups; g += 8) {
            uint8_t r0b0 = rp0[0], r0b1 = rp0[1];
            uint8_t r0b2 = rp0[2], r0b3 = rp0[3];
            uint8_t r0b4 = rp0[4], r0b5 = rp0[5];
            uint8_t r0b6 = rp0[6], r0b7 = rp0[7];
            uint8_t r1b0 = rp1[0], r1b1 = rp1[1];
            uint8_t r1b2 = rp1[2], r1b3 = rp1[3];
            uint8_t r1b4 = rp1[4], r1b5 = rp1[5];
            uint8_t r1b6 = rp1[6], r1b7 = rp1[7];

            a0 +=
                act_luts[g][TMAC_P_MASK[r0b0]] - act_luts[g][TMAC_N_MASK[r0b0]];
            b0 +=
                act_luts[g][TMAC_P_MASK[r1b0]] - act_luts[g][TMAC_N_MASK[r1b0]];
            a1 += act_luts[g + 1][TMAC_P_MASK[r0b1]] -
                  act_luts[g + 1][TMAC_N_MASK[r0b1]];
            b1 += act_luts[g + 1][TMAC_P_MASK[r1b1]] -
                  act_luts[g + 1][TMAC_N_MASK[r1b1]];
            a2 += act_luts[g + 2][TMAC_P_MASK[r0b2]] -
                  act_luts[g + 2][TMAC_N_MASK[r0b2]];
            b2 += act_luts[g + 2][TMAC_P_MASK[r1b2]] -
                  act_luts[g + 2][TMAC_N_MASK[r1b2]];
            a3 += act_luts[g + 3][TMAC_P_MASK[r0b3]] -
                  act_luts[g + 3][TMAC_N_MASK[r0b3]];
            b3 += act_luts[g + 3][TMAC_P_MASK[r1b3]] -
                  act_luts[g + 3][TMAC_N_MASK[r1b3]];

            a0 += act_luts[g + 4][TMAC_P_MASK[r0b4]] -
                  act_luts[g + 4][TMAC_N_MASK[r0b4]];
            b0 += act_luts[g + 4][TMAC_P_MASK[r1b4]] -
                  act_luts[g + 4][TMAC_N_MASK[r1b4]];
            a1 += act_luts[g + 5][TMAC_P_MASK[r0b5]] -
                  act_luts[g + 5][TMAC_N_MASK[r0b5]];
            b1 += act_luts[g + 5][TMAC_P_MASK[r1b5]] -
                  act_luts[g + 5][TMAC_N_MASK[r1b5]];
            a2 += act_luts[g + 6][TMAC_P_MASK[r0b6]] -
                  act_luts[g + 6][TMAC_N_MASK[r0b6]];
            b2 += act_luts[g + 6][TMAC_P_MASK[r1b6]] -
                  act_luts[g + 6][TMAC_N_MASK[r1b6]];
            a3 += act_luts[g + 7][TMAC_P_MASK[r0b7]] -
                  act_luts[g + 7][TMAC_N_MASK[r0b7]];
            b3 += act_luts[g + 7][TMAC_P_MASK[r1b7]] -
                  act_luts[g + 7][TMAC_N_MASK[r1b7]];

            rp0 += 8;
            rp1 += 8;
        }

        int32_t total0 = a0 + a1 + a2 + a3;
        int32_t total1 = b0 + b1 + b2 + b3;

        for (; g < block_groups; g++) {
            uint8_t v0 = *rp0++, v1 = *rp1++;
            total0 +=
                act_luts[g][TMAC_P_MASK[v0]] - act_luts[g][TMAC_N_MASK[v0]];
            total1 +=
                act_luts[g][TMAC_P_MASK[v1]] - act_luts[g][TMAC_N_MASK[v1]];
        }

        w->out[r] += (float) total0 * inv_scale;
        w->out[r + 1] += (float) total1 * inv_scale;
    }

    /* Handle odd remainder row */
    for (; r < r_end; r++) {
        const uint8_t *row_ptr = packed_ptr + r * packed_stride + g_start;
        int32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        int g = 0;
        for (; g + 8 <= block_groups; g += 8) {
            t0 += act_luts[g][TMAC_P_MASK[row_ptr[0]]] -
                  act_luts[g][TMAC_N_MASK[row_ptr[0]]];
            t1 += act_luts[g + 1][TMAC_P_MASK[row_ptr[1]]] -
                  act_luts[g + 1][TMAC_N_MASK[row_ptr[1]]];
            t2 += act_luts[g + 2][TMAC_P_MASK[row_ptr[2]]] -
                  act_luts[g + 2][TMAC_N_MASK[row_ptr[2]]];
            t3 += act_luts[g + 3][TMAC_P_MASK[row_ptr[3]]] -
                  act_luts[g + 3][TMAC_N_MASK[row_ptr[3]]];
            t0 += act_luts[g + 4][TMAC_P_MASK[row_ptr[4]]] -
                  act_luts[g + 4][TMAC_N_MASK[row_ptr[4]]];
            t1 += act_luts[g + 5][TMAC_P_MASK[row_ptr[5]]] -
                  act_luts[g + 5][TMAC_N_MASK[row_ptr[5]]];
            t2 += act_luts[g + 6][TMAC_P_MASK[row_ptr[6]]] -
                  act_luts[g + 6][TMAC_N_MASK[row_ptr[6]]];
            t3 += act_luts[g + 7][TMAC_P_MASK[row_ptr[7]]] -
                  act_luts[g + 7][TMAC_N_MASK[row_ptr[7]]];
            row_ptr += 8;
        }
        int32_t total = t0 + t1 + t2 + t3;
        for (; g < block_groups; g++) {
            uint8_t b = *row_ptr++;
            total += act_luts[g][TMAC_P_MASK[b]] - act_luts[g][TMAC_N_MASK[b]];
        }
        w->out[r] += (float) total * inv_scale;
    }
}

static void scalar_tmac_rows(const uint8_t *packed_ptr,
                             int packed_stride,
                             int32_t (*act_luts)[16],
                             int g_start,
                             int block_groups,
                             float *out,
                             float inv_scale,
                             int total_rows)
{
    scalar_row_work_t work = {
        .packed_ptr = packed_ptr,
        .packed_stride = packed_stride,
        .act_luts = (const int32_t (*)[16]) act_luts,
        .g_start = g_start,
        .block_groups = block_groups,
        .out = out,
        .inv_scale = inv_scale,
        .total_rows = total_rows,
    };

    if (total_rows >= BM_PAR_THRESHOLD && bm_get_threads() > 1)
        bm_parallel_for(total_rows, scalar_row_worker, &work);
    else
        scalar_row_worker(0, 1, &work);
}

/*
 * Batched BitLinear forward pass for prefill (scalar).
 * Per-token calls to bitlinear_forward.
 */
void scalar_bitlinear_forward_batch(const float *restrict x_batch,
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

    for (int t = 0; t < n_tokens; t++)
        scalar_bitlinear_forward(x_batch + t * n, n, w, norm_w,
                                 out_batch + t * w->rows, x_norm_buf,
                                 x_quant_batch + t * quant_stride);
}

/*
 * BitLinear forward pass (scalar T-MAC).
 *
 * T-MAC LUT optimization with cache blocking: process columns in blocks
 * that fit L1 cache. Block size = 256 groups * 16 entries * 4 bytes = 16KB.
 */
void scalar_bitlinear_forward(const float *restrict x,
                              int n,
                              const tensor_t *restrict w,
                              const tensor_t *restrict norm_w,
                              float *restrict out,
                              float *restrict x_norm_buf,
                              int8_t *restrict x_quant_buf)
{
    BITLINEAR_CHECK(n, w);

    /* Step 1: RMS normalize (returns fused max_abs) */
    float max_abs = scalar_rms_norm(x, n, norm_w->data, x_norm_buf);
    float scale_x = 127.0f / (max_abs + 1e-5f);
    int i;

    /* Step 2: Quantize to int8 (4x unrolled) */
    for (i = 0; i <= n - 4; i += 4) {
        float v0 = x_norm_buf[i] * scale_x;
        float v1 = x_norm_buf[i + 1] * scale_x;
        float v2 = x_norm_buf[i + 2] * scale_x;
        float v3 = x_norm_buf[i + 3] * scale_x;

        int iv0 = (int) (v0 + (v0 >= 0 ? 0.5f : -0.5f));
        int iv1 = (int) (v1 + (v1 >= 0 ? 0.5f : -0.5f));
        int iv2 = (int) (v2 + (v2 >= 0 ? 0.5f : -0.5f));
        int iv3 = (int) (v3 + (v3 >= 0 ? 0.5f : -0.5f));

        x_quant_buf[i] = (int8_t) (iv0 > 127 ? 127 : (iv0 < -128 ? -128 : iv0));
        x_quant_buf[i + 1] =
            (int8_t) (iv1 > 127 ? 127 : (iv1 < -128 ? -128 : iv1));
        x_quant_buf[i + 2] =
            (int8_t) (iv2 > 127 ? 127 : (iv2 < -128 ? -128 : iv2));
        x_quant_buf[i + 3] =
            (int8_t) (iv3 > 127 ? 127 : (iv3 < -128 ? -128 : iv3));
    }
    for (; i < n; i++) {
        float val = x_norm_buf[i] * scale_x;
        int ival = (int) (val + (val >= 0 ? 0.5f : -0.5f));
        x_quant_buf[i] =
            (int8_t) (ival > 127 ? 127 : (ival < -128 ? -128 : ival));
    }

    int cols = w->cols;
    int packed_stride = (cols + 3) / 4;
    const uint8_t *packed_ptr = w->packed_data;
    float inv_scale = 1.0f / (scale_x * w->scale);

    int n_groups = cols / 4;

    /* Use persistent LUT buffer if available; fall back to heap alloc. */
    int32_t (*act_luts)[16] = g_scalar_lut_buf;
    bool lut_owned = false;
    if (!act_luts) {
        act_luts = malloc(TMAC_BLOCK_SIZE * 16 * sizeof(int32_t));
        if (!act_luts) {
            fprintf(stderr, "Fatal: act_luts allocation failed\n");
            exit(1);
        }
        lut_owned = true;
    }

    /* Initialize output to zero (accumulate across blocks) */
    for (int r = 0; r < w->rows; r++)
        out[r] = 0.0f;

    /* Process columns in cache-friendly blocks */
    for (int g_start = 0; g_start < n_groups; g_start += TMAC_BLOCK_SIZE) {
        int g_end = g_start + TMAC_BLOCK_SIZE;
        if (g_end > n_groups)
            g_end = n_groups;
        int block_groups = g_end - g_start;

        /* Build LUTs for this block */
        for (int g = 0; g < block_groups; g++)
            build_tmac_lut(x_quant_buf + (g_start + g) * 4, act_luts[g]);

        /* Dispatch row loop */
        scalar_tmac_rows(packed_ptr, packed_stride, act_luts, g_start,
                         block_groups, out, inv_scale, w->rows);
    }

    if (lut_owned)
        free(act_luts);

    /* Handle remaining columns (< 4) with scalar multiply approach */
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
