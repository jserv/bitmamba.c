/*
 * Metal compute shaders for BitMamba GPU acceleration.
 *
 * Kernel 1: rms_norm_quantize -- fused RMSNorm + max_abs + quantize
 * Kernel 2: bitlinear_batch   -- direct ternary dot product (no LUT)
 * Kernel 3: residual_add      -- vectorized elementwise x[i] += y[i]
 */

#include <metal_stdlib>
using namespace metal;

/* Kernel 1: Fused RMSNorm + Quantize */

/*
 * One threadgroup per token.
 * Uses simd_sum/simd_max for intra-SIMD-group reduction (hardware-accelerated),
 * then one shared memory pass for cross-SIMD-group reduction.
 * Output: x_quant[n_tokens][quant_stride] (int8) + scale_x[n_tokens]
 */
kernel void rms_norm_quantize(
    device const float *x_batch          [[buffer(0)]],
    device const float *norm_weight      [[buffer(1)]],
    device char *x_quant_batch           [[buffer(2)]],
    device float *scale_x_out            [[buffer(3)]],
    device float *x_norm_buf             [[buffer(4)]],
    constant int &n                      [[buffer(5)]],
    constant int &quant_stride           [[buffer(6)]],
    threadgroup float *shared_mem        [[threadgroup(0)]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint tg_size                         [[threads_per_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]],
    uint gid                             [[threadgroup_position_in_grid]])
{
    (void)x_norm_buf; /* unused: recomputed in phase 3 */

    int token_idx = gid;
    device const float *x = x_batch + token_idx * n;
    device char *xq = x_quant_batch + token_idx * quant_stride;
    uint n_simd_groups = (tg_size + 31) / 32;

    /* Phase 1: parallel sum of squares with simd_sum */
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += tg_size)
        local_sum += x[i] * x[i];

    /* Hardware SIMD reduction (32 lanes -> 1 value, no shared memory) */
    local_sum = simd_sum(local_sum);

    /* Cross-SIMD-group reduction via shared memory (one barrier) */
    if (simd_lane == 0)
        shared_mem[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < n_simd_groups; s++)
            total += shared_mem[s];
        shared_mem[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = 1.0f / sqrt(shared_mem[0] / float(n) + 1e-6f);

    /* Phase 2: scale + track max_abs with simd_max */
    float local_max = 0.0f;
    for (int i = tid; i < n; i += tg_size) {
        float v = x[i] * rms * norm_weight[i];
        float av = abs(v);
        local_max = max(local_max, av);
    }

    local_max = simd_max(local_max);

    if (simd_lane == 0)
        shared_mem[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gmax = 0.0f;
        for (uint s = 0; s < n_simd_groups; s++)
            gmax = max(gmax, shared_mem[s]);
        shared_mem[0] = gmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_abs = shared_mem[0];
    float scale_x = 127.0f / (max_abs + 1e-5f);

    if (tid == 0)
        scale_x_out[token_idx] = scale_x;

    /* Phase 3: quantize to int8 (no barrier needed -- scale_x is broadcast) */
    for (int i = tid; i < n; i += tg_size) {
        float v = x[i] * rms * norm_weight[i];
        float val = v * scale_x;
        int ival = int(round(val));
        ival = clamp(ival, -128, 127);
        xq[i] = char(ival);
    }
}

/* Kernel 2: Direct Ternary BitLinear Dot Product */

/*
 * Optimized for Apple Silicon coalesced memory access:
 * - One SIMD group (32 threads) cooperatively processes ONE row
 * - Adjacent lanes read adjacent uint values (4 packed bytes = 16 weights)
 *   giving 128-byte coalesced reads per SIMD group per iteration
 * - rows_per_tg = tg_size / 32 (derived dynamically, not hardcoded)
 * - simd_sum reduction (hardware-accelerated, no shared memory needed)
 * - Shared memory only for activation cache (one barrier)
 *
 * Grid: (ceil(rows/rows_per_tg), n_tokens, 1)
 * TG:   (rows_per_tg * 32, 1, 1) -- clamped to pipeline max
 *
 * Requires: cols % 16 == 0 (validated at dispatch time).
 * Requires: packed_weights 4-byte aligned per row (model format guarantees).
 */
kernel void bitlinear_batch(
    device const char *x_quant_batch     [[buffer(0)]],
    device const float *scale_x_arr      [[buffer(1)]],
    device const uchar *packed_weights   [[buffer(2)]],
    device float *out_batch              [[buffer(3)]],
    constant int &rows                   [[buffer(4)]],
    constant int &cols                   [[buffer(5)]],
    constant float &w_scale              [[buffer(6)]],
    constant int &quant_stride           [[buffer(7)]],
    constant int &out_dim                [[buffer(8)]],
    threadgroup char *shared_xq          [[threadgroup(0)]],
    uint3 tid3                           [[thread_position_in_threadgroup]],
    uint3 tg_size3                       [[threads_per_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]])
{
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint rows_per_tg = tg_size / 32;
    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    int token_idx = int(tgid.y);
    int row = int(tgid.x) * int(rows_per_tg) + int(simd_id);

    /* Cooperatively load activation vector into shared memory */
    device const char *xq = x_quant_batch + token_idx * quant_stride;
    for (int i = int(tid); i < cols; i += int(tg_size))
        shared_xq[i] = xq[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= rows)
        return;

    /*
     * Coalesced weight reads: each lane loads a uint (4 packed bytes = 16 weights).
     * Adjacent lanes read adjacent uints, so 32 lanes read 128 contiguous bytes.
     */
    int packed_stride = (cols + 3) / 4;
    device const uint *wp = (device const uint *)(packed_weights + row * packed_stride);

    int total = 0;
    int n_uints = cols / 16;

    for (int k = int(lane_id); k < n_uints; k += 32) {
        uint pv = wp[k];
        int base = k * 16;

        uchar v0 = uchar(pv & 0xFFu);
        total += (min(int(v0 & 3u), 2) - 1) * int(shared_xq[base])
               + (min(int((v0 >> 2) & 3u), 2) - 1) * int(shared_xq[base + 1])
               + (min(int((v0 >> 4) & 3u), 2) - 1) * int(shared_xq[base + 2])
               + (min(int(v0 >> 6), 2) - 1) * int(shared_xq[base + 3]);

        uchar v1 = uchar((pv >> 8) & 0xFFu);
        total += (min(int(v1 & 3u), 2) - 1) * int(shared_xq[base + 4])
               + (min(int((v1 >> 2) & 3u), 2) - 1) * int(shared_xq[base + 5])
               + (min(int((v1 >> 4) & 3u), 2) - 1) * int(shared_xq[base + 6])
               + (min(int(v1 >> 6), 2) - 1) * int(shared_xq[base + 7]);

        uchar v2 = uchar((pv >> 16) & 0xFFu);
        total += (min(int(v2 & 3u), 2) - 1) * int(shared_xq[base + 8])
               + (min(int((v2 >> 2) & 3u), 2) - 1) * int(shared_xq[base + 9])
               + (min(int((v2 >> 4) & 3u), 2) - 1) * int(shared_xq[base + 10])
               + (min(int(v2 >> 6), 2) - 1) * int(shared_xq[base + 11]);

        uchar v3 = uchar((pv >> 24) & 0xFFu);
        total += (min(int(v3 & 3u), 2) - 1) * int(shared_xq[base + 12])
               + (min(int((v3 >> 2) & 3u), 2) - 1) * int(shared_xq[base + 13])
               + (min(int((v3 >> 4) & 3u), 2) - 1) * int(shared_xq[base + 14])
               + (min(int(v3 >> 6), 2) - 1) * int(shared_xq[base + 15]);
    }

    /* Hardware-accelerated SIMD reduction -- no shared memory needed */
    total = simd_sum(total);

    if (lane_id == 0) {
        float inv_scale = 1.0f / (scale_x_arr[token_idx] * w_scale);
        out_batch[token_idx * out_dim + row] = float(total) * inv_scale;
    }
}

/* Kernel 3: Vectorized Residual Addition */

/*
 * Elementwise x[i] += y[i] using float4 for 4x throughput.
 * Keeps residual on GPU between layers for cross-layer batching.
 *
 * Grid: (ceil(count/1024), 1, 1)   TG: (256, 1, 1)
 * Each thread processes 4 elements via float4.
 */
kernel void residual_add(
    device float *x            [[buffer(0)]],
    device const float *y      [[buffer(1)]],
    constant int &count        [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]])
{
    /* float4 vectorized path: each thread handles 4 contiguous floats */
    int base = int(tid) * 4;
    if (base + 3 < count) {
        device float4 *x4 = (device float4 *)(x + base);
        device const float4 *y4 = (device const float4 *)(y + base);
        *x4 += *y4;
    } else {
        /* Scalar tail for remaining elements */
        for (int i = base; i < min(base + 4, count); i++)
            x[i] += y[i];
    }
}
