/*
 * Runtime CPU feature detection and kernel dispatch.
 *
 * Detects available SIMD features at startup and populates the global
 * dispatch table g_kernels. The scalar fallback is always available.
 */

#include <stdio.h>

#include "bitmamba.h"
#include "dispatch.h"

kernel_dispatch_t g_kernels;

void dispatch_init(void)
{
    /* Default: scalar fallback (always linked) */
    g_kernels.rms_norm = scalar_rms_norm;
    g_kernels.bitlinear_forward = scalar_bitlinear_forward;
    g_kernels.bitlinear_forward_batch = scalar_bitlinear_forward_batch;
    g_kernels.bitlinear_prepare = NULL;
    g_kernels.bitlinear_row_worker = NULL;
    g_kernels.ssm_inner = scalar_ssm_inner;
    g_kernels.name = "scalar";

#if defined(__x86_64__)
    /* GCC requires explicit cpu_init before cpu_supports (Clang does not) */
#if defined(__GNUC__) && !defined(__clang__)
    __builtin_cpu_init();
#endif
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        g_kernels.rms_norm = avx2_rms_norm;
        g_kernels.bitlinear_forward = avx2_bitlinear_forward;
        g_kernels.bitlinear_forward_batch = avx2_bitlinear_forward_batch;
        g_kernels.bitlinear_prepare = avx2_bitlinear_prepare;
        g_kernels.bitlinear_row_worker = avx2_row_worker;
        g_kernels.ssm_inner = avx2_ssm_inner;
        g_kernels.name = "avx2";
    }
#elif defined(__aarch64__)
    /* ARMv8-A baseline guarantees NEON; unconditional select */
    g_kernels.rms_norm = neon_rms_norm;
    g_kernels.bitlinear_forward = neon_bitlinear_forward;
    g_kernels.bitlinear_forward_batch = neon_bitlinear_forward_batch;
    g_kernels.bitlinear_prepare = neon_bitlinear_prepare;
    g_kernels.bitlinear_row_worker = neon_row_worker;
    g_kernels.ssm_inner = neon_ssm_inner;
    g_kernels.name = "neon";
#endif

    fprintf(stderr, "[INFO] Kernel: %s\n", g_kernels.name);
}
