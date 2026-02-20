#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include "bitmamba.h"
#include "dispatch.h"
#include "threadpool.h"

/*
 * Horizontal sum of 8 x int32 in __m256i.
 */
static inline int32_t hsum256_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}

float avx2_rms_norm(const float *restrict x,
                    int size,
                    const float *restrict weight,
                    float *restrict out)
{
    float sum_sq = 0.0f;
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;

    /* 4 independent accumulators to hide FMA latency */
    for (; i <= size - 32; i += 32) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        __m256 v2 = _mm256_loadu_ps(x + i + 16);
        __m256 v3 = _mm256_loadu_ps(x + i + 24);
        s0 = _mm256_fmadd_ps(v0, v0, s0);
        s1 = _mm256_fmadd_ps(v1, v1, s1);
        s2 = _mm256_fmadd_ps(v2, v2, s2);
        s3 = _mm256_fmadd_ps(v3, v3, s3);
    }
    for (; i <= size - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        s0 = _mm256_fmadd_ps(v, v, s0);
    }

    __m256 sum_vec =
        _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    for (int k = 0; k < 8; k++)
        sum_sq += temp[k];
    for (; i < size; i++)
        sum_sq += x[i] * x[i];

    float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);

    /* Scaling with fused max-abs tracking */
    __m256 rms_vec = _mm256_set1_ps(rms);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 vma = _mm256_setzero_ps();
    for (i = 0; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, rms_vec), vw);
        _mm256_storeu_ps(out + i, res);
        vma = _mm256_max_ps(vma, _mm256_andnot_ps(sign_mask, res));
    }
    _mm256_storeu_ps(temp, vma);
    float max_abs = 0.0f;
    for (int k = 0; k < 8; k++)
        if (temp[k] > max_abs)
            max_abs = temp[k];
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
 * Process rows [r_start, r_end) for AVX2 bitlinear forward.
 * Extracted helper for use by both static-partition and work-stealing paths.
 *
 * 8-row primary unrolling amortizes activation deinterleave across 8 rows.
 * Falls through to 4-row and 1-row remainders.
 * X-macros (ROWS_8/ROWS_4) eliminate per-row boilerplate duplication.
 *
 * Register budget (16 YMM): 8 accumulators + x_u + 6 decode constants
 * + 3 temporaries = 18, but deint_shuf/deint_perm are dead during
 * the decode sequence (only live during activation load), so the compiler
 * reclaims those 2 registers. Peak = 16 = exact fit.
 * Verify no accumulator spills: objdump -d | grep 'vmovdqa.*\[rsp'
 */
static void avx2_row_block(bitlinear_row_work_t *w, int r_start, int r_end)
{
    if (r_start >= r_end)
        return;

    const int8_t *x_quant_buf = w->x_quant;
    const uint8_t *packed_ptr = w->packed_data;
    float inv_scale = w->inv_scale;
    int cols = w->cols;
    int packed_stride = w->packed_stride;

    __m256i ones = _mm256_set1_epi16(1);
    __m256i offset_val = _mm256_set1_epi8(-128);
    __m256i shifts = _mm256_setr_epi64x(0, 2, 4, 6);
    __m256i mask03 = _mm256_set1_epi8(0x03);
    __m256i two_vec = _mm256_set1_epi8(2);
    __m256i one_vec = _mm256_set1_epi8(1);
    __m256i deint_shuf =
        _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                         0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m256i deint_perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    /*
     * Decode 32 ternary weights from 8 packed bytes via shift-decode,
     * multiply against deinterleaved activations (x_u), and accumulate.
     * Requires x_u, pack_off, and SIMD constants in the calling scope.
     */
#define DECODE_ROW_W(row_off, acc)                                             \
    do {                                                                       \
        __m128i raw = _mm_loadl_epi64(                                         \
            (const __m128i *) (packed_ptr + (row_off) + pack_off));            \
        __m256i bc = _mm256_broadcastq_epi64(raw);                             \
        __m256i wv = _mm256_sub_epi8(                                          \
            _mm256_min_epu8(                                                   \
                _mm256_and_si256(_mm256_srlv_epi64(bc, shifts), mask03),       \
                two_vec),                                                      \
            one_vec);                                                          \
        __m256i prod = _mm256_sub_epi16(_mm256_maddubs_epi16(x_u, wv),         \
                                        _mm256_maddubs_epi16(offset_val, wv)); \
        (acc) = _mm256_add_epi32((acc), _mm256_madd_epi16(prod, ones));        \
    } while (0)

    /* Load 32 quantized activations + deinterleave for shift-decode layout */
#define LOAD_DEINTERLEAVE_ACT(c)                                               \
    __m256i x_vec = _mm256_loadu_si256((const __m256i *) (x_quant_buf + (c))); \
    __m256i x_d = _mm256_permutevar8x32_epi32(                                 \
        _mm256_shuffle_epi8(x_vec, deint_shuf), deint_perm);                   \
    __m256i x_u = _mm256_xor_si256(x_d, offset_val)

    /* X-macro row expansion: ROWS_N(F) applies F(0) F(1) ... F(N-1) */
#define ROWS_8(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7)
#define ROWS_4(F) F(0) F(1) F(2) F(3)

    /* Per-row step macros (used by ROWS_N to eliminate unroll boilerplate) */
#define DECL_ACC(i) __m256i acc##i = _mm256_setzero_si256();
#define DECL_ROW_OFF(i) int row##i##_off = (r + (i)) * packed_stride;
#define DO_DECODE(i) DECODE_ROW_W(row##i##_off, acc##i);
#define DO_HSUM(i) int32_t total##i = hsum256_epi32(acc##i);
#define DO_TAIL(i) \
    total##i += UNPACK_LUT[packed_ptr[row##i##_off + byte_idx]][sub_idx] * xv;
#define DO_STORE(i) w->out[r + (i)] = (float) total##i * inv_scale;

    int r = r_start;

    /* 8-row unrolling: activation deinterleave amortized across 8 rows */
    for (; r + 8 <= r_end; r += 8) {
        ROWS_8(DECL_ACC)
        ROWS_8(DECL_ROW_OFF)

        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            int pack_off = c / 4;
            LOAD_DEINTERLEAVE_ACT(c);
            ROWS_8(DO_DECODE)
        }

        ROWS_8(DO_HSUM)
        for (; c < cols; c++) {
            int byte_idx = c / 4;
            int sub_idx = c & 3;
            int8_t xv = x_quant_buf[c];
            ROWS_8(DO_TAIL)
        }
        ROWS_8(DO_STORE)
    }

    /* 4-row remainder */
    for (; r + 4 <= r_end; r += 4) {
        ROWS_4(DECL_ACC)
        ROWS_4(DECL_ROW_OFF)

        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            int pack_off = c / 4;
            LOAD_DEINTERLEAVE_ACT(c);
            ROWS_4(DO_DECODE)
        }

        ROWS_4(DO_HSUM)
        for (; c < cols; c++) {
            int byte_idx = c / 4;
            int sub_idx = c & 3;
            int8_t xv = x_quant_buf[c];
            ROWS_4(DO_TAIL)
        }
        ROWS_4(DO_STORE)
    }

    /* 1-row remainder */
    for (; r < r_end; r++) {
        __m256i acc = _mm256_setzero_si256();
        int row_off = r * packed_stride;
        int c = 0;

        for (; c + 32 <= cols; c += 32) {
            int pack_off = c / 4;
            LOAD_DEINTERLEAVE_ACT(c);
            DECODE_ROW_W(row_off, acc);
        }

        int32_t total = hsum256_epi32(acc);
        for (; c < cols; c++) {
            int8_t w_val = UNPACK_LUT[packed_ptr[row_off + c / 4]][c & 3];
            total += w_val * x_quant_buf[c];
        }
        w->out[r] = (float) total * inv_scale;
    }

#undef DECODE_ROW_W
#undef LOAD_DEINTERLEAVE_ACT
#undef ROWS_8
#undef ROWS_4
#undef DECL_ACC
#undef DECL_ROW_OFF
#undef DO_DECODE
#undef DO_HSUM
#undef DO_TAIL
#undef DO_STORE
}

/*
 * Row-parallel worker for AVX2 bitlinear forward.
 * Supports two modes:
 *   - Static partition (chunk_ctr == NULL): bm_work_range for NUMA locality.
 *   - Work-stealing (chunk_ctr != NULL): atomic counter for load balancing.
 */
void avx2_row_worker(int tid, int n_threads, void *arg)
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
            avx2_row_block(w, s, e);
            chunk = atomic_fetch_add_explicit(w->chunk_ctr, 1,
                                              memory_order_relaxed);
        }
    } else {
        /* Static partition */
        int r_start, r_end;
        bm_work_range(tid, n_threads, w->rows, &r_start, &r_end);
        avx2_row_block(w, r_start, r_end);
    }
}

/*
 * Decomposed bitlinear prepare: RMS normalize + quantize to int8.
 * Returns scale_x; caller computes inv_scale = 1/(scale_x * w->scale).
 */
float avx2_bitlinear_prepare(const float *restrict x,
                             int n,
                             const tensor_t *restrict norm_w,
                             float *restrict x_norm_buf,
                             int8_t *restrict x_quant_buf)
{
    float max_abs = avx2_rms_norm(x, n, norm_w->data, x_norm_buf);
    float scale_x = 127.0f / (max_abs + 1e-5f);
    int i;

    /* Quantize to int8 -- 32 values per iteration.
     * packs_epi32/packs_epi16 interleave across 128-bit lanes,
     * so permutevar8x32 restores linear order. */
    __m256 scale_v = _mm256_set1_ps(scale_x);
    __m256 min_v = _mm256_set1_ps(-128.0f);
    __m256 max_v = _mm256_set1_ps(127.0f);
    __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (i = 0; i <= n - 32; i += 32) {
        __m256 v0 = _mm256_loadu_ps(x_norm_buf + i);
        __m256 v1 = _mm256_loadu_ps(x_norm_buf + i + 8);
        __m256 v2 = _mm256_loadu_ps(x_norm_buf + i + 16);
        __m256 v3 = _mm256_loadu_ps(x_norm_buf + i + 24);

        v0 = _mm256_round_ps(
            _mm256_max_ps(min_v,
                          _mm256_min_ps(_mm256_mul_ps(v0, scale_v), max_v)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v1 = _mm256_round_ps(
            _mm256_max_ps(min_v,
                          _mm256_min_ps(_mm256_mul_ps(v1, scale_v), max_v)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v2 = _mm256_round_ps(
            _mm256_max_ps(min_v,
                          _mm256_min_ps(_mm256_mul_ps(v2, scale_v), max_v)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v3 = _mm256_round_ps(
            _mm256_max_ps(min_v,
                          _mm256_min_ps(_mm256_mul_ps(v3, scale_v), max_v)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        /* int32 -> int16 -> int8, then fix lane interleave */
        __m256i b = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1),
                                       _mm256_packs_epi32(i2, i3));
        _mm256_storeu_si256((__m256i *) (x_quant_buf + i),
                            _mm256_permutevar8x32_epi32(b, perm_mask));
    }
    for (; i < n; i++) {
        float val = x_norm_buf[i] * scale_x;
        int ival = (int) roundf(val);
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
} avx2_batch_row_work_t;

static void avx2_batch_row_worker(int tid, int n_threads, void *arg)
{
    avx2_batch_row_work_t *bw = (avx2_batch_row_work_t *) arg;
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
        avx2_row_worker(0, 1, &single);
    }
}

/*
 * Batched BitLinear forward pass for prefill (AVX2).
 *
 * Phase 1: normalize + quantize all tokens upfront.
 * Phase 2: single-dispatch weight-stationary row work across all tokens.
 * One fork-join for all tokens instead of per-token dispatch.
 */
void avx2_bitlinear_forward_batch(const float *restrict x_batch,
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
            avx2_bitlinear_prepare(x_batch + t * n, n, norm_w, x_norm_buf, xq);
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
        avx2_batch_row_work_t bw = {
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
        bm_parallel_for(out_dim, avx2_batch_row_worker, &bw);
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
            avx2_row_worker(0, 1, &work);
        }
    }

    free(scale_x_heap);
}

/*
 * Vectorized expf for AVX2+FMA (8 floats at once).
 * Polynomial approximation ported from llama.cpp ggml.c.
 * Max relative error ~1.5e-5 over [-87, 88].
 */
static inline __m256 bm_v_expf_avx2(__m256 x)
{
    /* Clamp to prevent overflow/underflow */
    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0f)),
                      _mm256_set1_ps(-87.33654475f));

    /* exp(x) = 2^(x / ln2) = 2^(n + r) */
    __m256 log2e = _mm256_set1_ps(1.44269504089f);
    __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);

    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 n_f = _mm256_round_ps(t, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    /* r = x - n * ln2 (FMA for precision) */
    __m256 r = _mm256_fnmadd_ps(n_f, ln2, x); /* x - n*ln2 */

    /* Horner's: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120)))) */
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(0.5f);
    __m256 c3 = _mm256_set1_ps(0.16666666666f);
    __m256 c4 = _mm256_set1_ps(0.04166666666f);
    __m256 c5 = _mm256_set1_ps(0.00833333333f);

    __m256 p = _mm256_fmadd_ps(c5, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c1);

    /* Multiply by 2^n: construct IEEE float with exponent = n+127 */
    __m256i n_i = _mm256_cvtps_epi32(n_f);
    __m256i exp_bits =
        _mm256_slli_epi32(_mm256_add_epi32(n_i, _mm256_set1_epi32(127)), 23);
    __m256 pow2n = _mm256_castsi256_ps(exp_bits);

    return _mm256_mul_ps(p, pow2n);
}

/*
 * Vectorized silu(x) = x / (1 + exp(-x)) for AVX2+FMA.
 * Uses rcp + NR refinement for division.
 */
static inline __m256 bm_v_silu_avx2(__m256 x)
{
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 exp_neg = bm_v_expf_avx2(_mm256_sub_ps(_mm256_setzero_ps(), x));
    __m256 denom = _mm256_add_ps(one, exp_neg);
    /* Newton-Raphson reciprocal: r = rcp(d), r = r * (2 - d*r) */
    __m256 rcp = _mm256_rcp_ps(denom);
    rcp =
        _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, _mm256_set1_ps(2.0f)));
    return _mm256_mul_ps(x, rcp);
}

/*
 * AVX2 vectorized SSM inner loop.
 * Double-pumped: 16 elements per iteration (two independent __m256 chains)
 * to hide FMA latency (~5 cycles on Skylake/Zen) via ILP.
 * Falls back to 8-float and scalar remainders.
 */
void avx2_ssm_inner(const ssm_inner_args_t *a, int idx_start, int idx_end)
{
    int head_dim = a->head_dim;
    int d_inner = a->d_inner;

    const float *cw_tap0 = a->conv1d_w_t;
    const float *cw_tap1 = a->conv1d_w_t + d_inner;
    const float *cw_tap2 = a->conv1d_w_t + 2 * d_inner;
    const float *cw_tap3 = a->conv1d_w_t + 3 * d_inner;

    int idx = idx_start;

    while (idx < idx_end) {
        int h = idx / head_dim;
        int head_end = (h + 1) * head_dim;
        if (head_end > idx_end)
            head_end = idx_end;

        /* Per-head scalars */
        float dt_val = softplus(a->ptr_dt[h] + a->dt_bias[h]);
        float decay_s = expf(a->A_precomp[h] * dt_val);
        float C_s = a->ptr_C[h];
        float Bdt_s = a->ptr_B[h] * dt_val;

        __m256 v_decay = _mm256_set1_ps(decay_s);
        __m256 v_Bdt = _mm256_set1_ps(Bdt_s);
        __m256 v_C = _mm256_set1_ps(C_s);

        int d = idx;

        /* Double-pumped: 16 floats/iter (two independent 8-wide chains)
         * to hide FMA latency via ILP */
        for (; d + 16 <= head_end; d += 16) {
            __m256 conv_a = _mm256_loadu_ps(a->conv1d_b + d);
            __m256 conv_b = _mm256_loadu_ps(a->conv1d_b + d + 8);

            conv_a = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs0 + d),
                                     _mm256_loadu_ps(cw_tap0 + d), conv_a);
            conv_b = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs0 + d + 8),
                                     _mm256_loadu_ps(cw_tap0 + d + 8), conv_b);
            conv_a = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs1 + d),
                                     _mm256_loadu_ps(cw_tap1 + d), conv_a);
            conv_b = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs1 + d + 8),
                                     _mm256_loadu_ps(cw_tap1 + d + 8), conv_b);
            conv_a = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs2 + d),
                                     _mm256_loadu_ps(cw_tap2 + d), conv_a);
            conv_b = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs2 + d + 8),
                                     _mm256_loadu_ps(cw_tap2 + d + 8), conv_b);

            __m256 in_a = _mm256_loadu_ps(a->ptr_x + d);
            __m256 in_b = _mm256_loadu_ps(a->ptr_x + d + 8);
            conv_a =
                _mm256_fmadd_ps(in_a, _mm256_loadu_ps(cw_tap3 + d), conv_a);
            conv_b =
                _mm256_fmadd_ps(in_b, _mm256_loadu_ps(cw_tap3 + d + 8), conv_b);

            _mm256_storeu_ps(a->cs_write + d, in_a);
            _mm256_storeu_ps(a->cs_write + d + 8, in_b);

            __m256 xa_a = bm_v_silu_avx2(conv_a);
            __m256 xa_b = bm_v_silu_avx2(conv_b);

            __m256 st_a = _mm256_loadu_ps(a->ssm_state + d);
            __m256 st_b = _mm256_loadu_ps(a->ssm_state + d + 8);
            st_a = _mm256_fmadd_ps(st_a, v_decay, _mm256_mul_ps(xa_a, v_Bdt));
            st_b = _mm256_fmadd_ps(st_b, v_decay, _mm256_mul_ps(xa_b, v_Bdt));
            _mm256_storeu_ps(a->ssm_state + d, st_a);
            _mm256_storeu_ps(a->ssm_state + d + 8, st_b);

            __m256 za_a = bm_v_silu_avx2(_mm256_loadu_ps(a->ptr_z + d));
            __m256 za_b = bm_v_silu_avx2(_mm256_loadu_ps(a->ptr_z + d + 8));
            _mm256_storeu_ps(
                a->y + d,
                _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        st_a, v_C,
                        _mm256_mul_ps(xa_a, _mm256_loadu_ps(a->D + d))),
                    za_a));
            _mm256_storeu_ps(
                a->y + d + 8,
                _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        st_b, v_C,
                        _mm256_mul_ps(xa_b, _mm256_loadu_ps(a->D + d + 8))),
                    za_b));
        }

        /* 8-float remainder (head_dim not divisible by 16) */
        for (; d + 8 <= head_end; d += 8) {
            __m256 conv = _mm256_loadu_ps(a->conv1d_b + d);
            conv = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs0 + d),
                                   _mm256_loadu_ps(cw_tap0 + d), conv);
            conv = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs1 + d),
                                   _mm256_loadu_ps(cw_tap1 + d), conv);
            conv = _mm256_fmadd_ps(_mm256_loadu_ps(a->cs2 + d),
                                   _mm256_loadu_ps(cw_tap2 + d), conv);
            __m256 v_input = _mm256_loadu_ps(a->ptr_x + d);
            conv = _mm256_fmadd_ps(v_input, _mm256_loadu_ps(cw_tap3 + d), conv);
            _mm256_storeu_ps(a->cs_write + d, v_input);

            __m256 x_act = bm_v_silu_avx2(conv);
            __m256 state = _mm256_loadu_ps(a->ssm_state + d);
            state =
                _mm256_fmadd_ps(state, v_decay, _mm256_mul_ps(x_act, v_Bdt));
            _mm256_storeu_ps(a->ssm_state + d, state);

            __m256 z_act = bm_v_silu_avx2(_mm256_loadu_ps(a->ptr_z + d));
            _mm256_storeu_ps(
                a->y + d,
                _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        state, v_C,
                        _mm256_mul_ps(x_act, _mm256_loadu_ps(a->D + d))),
                    z_act));
        }

        /* Scalar remainder */
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
}

/*
 * BitLinear forward pass (AVX2).
 *
 * Key optimizations:
 * 1. In-register shift-decode: broadcastq + srlv_epi64 + mask decodes
 *    32 ternary weights without LUT gather (~6 vs ~12 cycles).
 * 2. Activation deinterleave: shuffle_epi8 + permutevar8x32 reorders
 *    activations to match the transposed weight layout from shift-decode.
 * 3. maddubs with offset correction: unsigned*signed dot product avoids
 *    the 128-bit split (cvtepi8_epi16 lo/hi).
 * 4. 8-row unrolling: activation deinterleave shared across 8 rows.
 *    X-macros (ROWS_8/ROWS_4) eliminate per-row boilerplate.
 * 5. 32-wide quantization: packs + permute writes 32 int8 per iteration.
 */
void avx2_bitlinear_forward(const float *restrict x,
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
        avx2_bitlinear_prepare(x, n, norm_w, x_norm_buf, x_quant_buf);

    /* Step 3: AVX2 ternary dot product -- dispatch rows in parallel. */
    int cols = w->cols;
    int rows = w->rows;
    int packed_stride = (cols + 3) / 4;
    const uint8_t *packed_ptr = w->packed_data;
    float inv_scale = 1.0f / (scale_x * w->scale);

    /* Work-stealing: enabled on single-NUMA to balance stragglers.
     * Disabled on multi-NUMA where static partition preserves locality.
     * Chunk counter starts at n_threads (chunks 0..n-1 pre-assigned by tid). */
    int n_threads = bm_get_threads();
    int cs = rows / (n_threads * 4);
    if (cs < 8)
        cs = 8;
    _Atomic int steal_ctr = n_threads;
    bool use_steal =
        (bm_get_numa_nodes() <= 1 && rows >= BM_PAR_THRESHOLD && n_threads > 1);

    bitlinear_row_work_t work = {
        .x_quant = x_quant_buf,
        .packed_data = packed_ptr,
        .out = out,
        .inv_scale = inv_scale,
        .rows = rows,
        .cols = cols,
        .packed_stride = packed_stride,
        .chunk_ctr = use_steal ? &steal_ctr : NULL,
        .chunk_size = cs,
    };

    if (rows >= BM_PAR_THRESHOLD && n_threads > 1)
        bm_parallel_for(rows, avx2_row_worker, &work);
    else
        avx2_row_worker(0, 1, &work);
}
