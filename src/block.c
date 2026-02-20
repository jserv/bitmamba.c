#include <stdatomic.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "bitmamba.h"
#include "threadpool.h"
#ifdef HAS_METAL
#include "metal/bm_metal.h"
#endif

void block_init_cache(bitmamba_block_t *b, int d_model, int n_heads)
{
    b->d_model = d_model;
    b->n_heads = n_heads;
    b->d_inner = d_model * BM_EXPAND_FACTOR;
    b->head_dim = b->d_inner / n_heads;
    b->d_conv = BM_CONV_KERNEL;

    /* SoA layout: conv_state[tap][channel] for contiguous channel access */
    b->conv_state = xcalloc_aligned((BM_CONV_KERNEL - 1) * b->d_inner,
                                    sizeof(float), BM_BUF_ALIGN);
    b->ssm_state = xcalloc_aligned(b->d_inner, sizeof(float), BM_BUF_ALIGN);
    b->A_precomp = NULL;  /* populated after model tensors are loaded */
    b->conv1d_w_t = NULL; /* populated after model tensors are loaded */
    b->conv_pos = 0;      /* ring-buffer write position */
}

/*
 * SSM head-parallel worker: each thread processes elements [idx_start,
 * idx_end). Each thread's range is disjoint, so no conflicts.
 */
static void ssm_head_worker(int tid, int n_threads, void *arg)
{
    ssm_inner_args_t *a = (ssm_inner_args_t *) arg;
    int idx_start, idx_end;
    bm_work_range(tid, n_threads, a->d_inner, &idx_start, &idx_end);
    g_kernels.ssm_inner(a, idx_start, idx_end);
}

/*
 * Populate ssm_inner_args_t from block state and proj_out buffer.
 * Helper used by ssm_scan_token and the batch worker.
 */
static inline void ssm_args_init(ssm_inner_args_t *a,
                                 bitmamba_block_t *b,
                                 float *proj_out,
                                 float *y,
                                 float *cs0,
                                 float *cs1,
                                 float *cs2,
                                 float *cs_write)
{
    int d_inner = b->d_inner;
    a->ptr_z = proj_out;
    a->ptr_x = proj_out + d_inner;
    a->ptr_B = proj_out + 2 * d_inner;
    a->ptr_C = proj_out + 2 * d_inner + b->n_heads;
    a->ptr_dt = proj_out + 2 * d_inner + 2 * b->n_heads;
    a->dt_bias = b->dt_bias.data;
    a->A_precomp = b->A_precomp;
    a->conv1d_w_t = b->conv1d_w_t;
    a->conv1d_b = b->conv1d_b.data;
    a->cs0 = cs0;
    a->cs1 = cs1;
    a->cs2 = cs2;
    a->cs_write = cs_write;
    a->ssm_state = b->ssm_state;
    a->D = b->D.data;
    a->y = y;
    a->head_dim = b->head_dim;
    a->d_inner = d_inner;
    a->n_heads = b->n_heads;
}

/*
 * Single-token SSM scan: Conv1d + selective state space update.
 * proj_out layout: [z(d_inner), x(d_inner), B(n_heads), C(n_heads),
 * dt(n_heads)] y receives the gated output [d_inner].
 *
 * conv_state uses SoA layout [tap][channel] with ring-buffer indexing
 * to eliminate per-token shift copies.
 */
static void ssm_scan_token(bitmamba_block_t *b, float *proj_out, float *y)
{
    int d_inner = b->d_inner;
    int d_conv_m1 = b->d_conv - 1;

    /* Ring-buffer: pos points to the oldest tap (will be overwritten) */
    int pos = b->conv_pos;
    int tap0 = pos;
    int tap1 = (pos + 1 < d_conv_m1) ? pos + 1 : pos + 1 - d_conv_m1;
    int tap2 = (pos + 2 < d_conv_m1) ? pos + 2 : pos + 2 - d_conv_m1;

    float *cs0 = b->conv_state + tap0 * d_inner;
    float *cs1 = b->conv_state + tap1 * d_inner;
    float *cs2 = b->conv_state + tap2 * d_inner;

    ssm_inner_args_t ssm_args;
    ssm_args_init(&ssm_args, b, proj_out, y, cs0, cs1, cs2, cs0);

    if (b->d_inner >= BM_PAR_THRESHOLD && bm_get_threads() > 1)
        bm_parallel_for(b->d_inner, ssm_head_worker, &ssm_args);
    else
        ssm_head_worker(0, 1, &ssm_args);

    /* Advance ring-buffer position */
    b->conv_pos = (pos + 1 < d_conv_m1) ? pos + 1 : 0;
}

/*
 * Batched SSM scan: single dispatch over d_inner, workers loop tokens inside.
 * Each thread owns channels [idx_start, idx_end) across ALL tokens.
 * No barriers needed: conv_state and ssm_state writes/reads are per-element,
 * and each thread's index range is disjoint.
 * Ring-buffer positions are precomputed per token (pure function of initial
 * pos).
 */
typedef struct {
    bitmamba_block_t *b;
    float *proj_batch; /* [n_tokens * proj_stride] */
    float *y_batch;    /* [n_tokens * d_inner] */
    int n_tokens;
    int proj_stride;
    int initial_conv_pos;
} ssm_batch_work_t;

static void ssm_batch_worker(int tid, int n_threads, void *arg)
{
    ssm_batch_work_t *bw = (ssm_batch_work_t *) arg;
    bitmamba_block_t *b = bw->b;
    int d_inner = b->d_inner;
    int d_conv_m1 = b->d_conv - 1;

    int idx_start, idx_end;
    bm_work_range(tid, n_threads, d_inner, &idx_start, &idx_end);
    if (idx_start >= idx_end)
        return;

    int pos = bw->initial_conv_pos;

    for (int t = 0; t < bw->n_tokens; t++) {
        /* Compute ring-buffer tap indices for this token */
        int tap0 = pos;
        int tap1 = (pos + 1 < d_conv_m1) ? pos + 1 : pos + 1 - d_conv_m1;
        int tap2 = (pos + 2 < d_conv_m1) ? pos + 2 : pos + 2 - d_conv_m1;
        float *cs0 = b->conv_state + tap0 * d_inner;
        float *cs1 = b->conv_state + tap1 * d_inner;
        float *cs2 = b->conv_state + tap2 * d_inner;

        float *proj_out = bw->proj_batch + t * bw->proj_stride;
        float *y = bw->y_batch + t * d_inner;

        ssm_inner_args_t a;
        ssm_args_init(&a, b, proj_out, y, cs0, cs1, cs2, cs0);
        g_kernels.ssm_inner(&a, idx_start, idx_end);

        /* Advance ring buffer for next token (local computation, no sync) */
        pos = (pos + 1 < d_conv_m1) ? pos + 1 : 0;
    }
}

/*
 * Fused block worker: executes in_proj rows, SSM heads, and out_proj rows
 * within a single bm_parallel_for dispatch, using in-dispatch barriers
 * instead of 3 separate fork-join round-trips per layer.
 */
typedef struct {
    bitmamba_block_t *b;
    bitlinear_row_work_t in_proj_rw;
    float *proj_out, *y;
    float *cs0, *cs1, *cs2, *cs_write;
    bitlinear_row_work_t out_proj_rw;
    float *out_norm_buf;
    int8_t *out_quant_buf;
    int out_n;
    bm_barrier_t barrier;
    int n_threads;
    /* NUMA weight replicas (NULL entries -> use mmap'd original) */
    uint8_t *const *in_proj_packed_numa;
    uint8_t *const *out_proj_packed_numa;
    /* Per-phase work-stealing counters */
    _Atomic int in_proj_steal_ctr;
    int in_proj_chunk_size;
    _Atomic int out_proj_steal_ctr;
    int out_proj_chunk_size;
} fused_block_work_t;

static void fused_block_worker(int tid, int n_threads, void *arg)
{
    fused_block_work_t *fw = (fused_block_work_t *) arg;
    bitmamba_block_t *b = fw->b;
    (void) n_threads; /* fw->n_threads used consistently instead */

    /* Phase 1: in_proj row work (NUMA-local weights, optional stealing) */
    {
        bitlinear_row_work_t in_local = fw->in_proj_rw;
        if (fw->in_proj_packed_numa) {
            int node = bm_tid_node(tid);
            if (fw->in_proj_packed_numa[node])
                in_local.packed_data = fw->in_proj_packed_numa[node];
        }
        /* Enable work-stealing via shared counter (single-NUMA only) */
        if (fw->in_proj_chunk_size > 0) {
            in_local.chunk_ctr = &fw->in_proj_steal_ctr;
            in_local.chunk_size = fw->in_proj_chunk_size;
        }
        g_kernels.bitlinear_row_worker(tid, fw->n_threads, &in_local);
    }

    /* Barrier: ensure proj_out is fully written */
    bm_barrier_wait(&fw->barrier, tid, fw->n_threads);

    /* Phase 2: SSM scan, parallelized over d_inner (element-level).
     * Distributing by elements (4096) instead of heads (64) eliminates the
     * 50% work imbalance at high thread counts (64 heads / 48 threads).
     * Per-head values (dt, A, B, C) are recomputed at head boundaries;
     * at most 2 boundary heads per thread (softplus + expf), negligible.
     */
    {
        int d_inner = b->d_inner;
        int idx_start, idx_end;
        bm_work_range(tid, fw->n_threads, d_inner, &idx_start, &idx_end);

        ssm_inner_args_t a;
        ssm_args_init(&a, b, fw->proj_out, fw->y, fw->cs0, fw->cs1, fw->cs2,
                      fw->cs_write);
        g_kernels.ssm_inner(&a, idx_start, idx_end);
    }

    /* Barrier: ensure y is fully written */
    bm_barrier_wait(&fw->barrier, tid, fw->n_threads);

    /* Phase 3: out_proj prepare (tid 0 only -- serial norm+quantize) */
    if (tid == 0) {
        float scale_x =
            g_kernels.bitlinear_prepare(fw->y, fw->out_n, &fw->b->out_proj_norm,
                                        fw->out_norm_buf, fw->out_quant_buf);
        fw->out_proj_rw.x_quant = fw->out_quant_buf;
        fw->out_proj_rw.inv_scale = 1.0f / (scale_x * fw->b->out_proj.scale);
    }

    /* Barrier: ensure inv_scale and quantized input visible to all */
    bm_barrier_wait(&fw->barrier, tid, fw->n_threads);

    /* Phase 4: out_proj row work (NUMA-local weights, optional stealing) */
    {
        bitlinear_row_work_t out_local = fw->out_proj_rw;
        if (fw->out_proj_packed_numa) {
            int node = bm_tid_node(tid);
            if (fw->out_proj_packed_numa[node])
                out_local.packed_data = fw->out_proj_packed_numa[node];
        }
        /* Enable work-stealing via shared counter (single-NUMA only) */
        if (fw->out_proj_chunk_size > 0) {
            out_local.chunk_ctr = &fw->out_proj_steal_ctr;
            out_local.chunk_size = fw->out_proj_chunk_size;
        }
        g_kernels.bitlinear_row_worker(tid, fw->n_threads, &out_local);
    }
}

/*
 * Single-token forward pass for one BitMamba block.
 *
 * proj_out, y, bl_x_norm, bl_x_quant are pre-allocated scratch buffers
 * passed from the model to avoid per-call allocation.
 *
 * When decomposed bitlinear is available (AVX2/NEON) and threading is active,
 * fuses the three per-layer operations into a single parallel dispatch with
 * in-dispatch barriers: 1 fork-join + 3 barriers instead of 3 fork-joins.
 */
void block_step(bitmamba_block_t *b,
                const float *u,
                float *out_buffer,
                float *proj_out,
                float *y,
                float *bl_x_norm,
                int8_t *bl_x_quant)
{
    if (g_kernels.bitlinear_prepare && bm_get_threads() > 1) {
        /* Fused path: 1 fork-join + 3 in-dispatch barriers per layer */
        int d_model = b->d_model;
        int d_inner = b->d_inner;
        int d_conv_m1 = b->d_conv - 1;

        /* Prepare in_proj input (serial: norm + quantize) */
        float in_scale = g_kernels.bitlinear_prepare(
            u, d_model, &b->in_proj_norm, bl_x_norm, bl_x_quant);

        /* Ring-buffer setup (same as ssm_scan_token) */
        int pos = b->conv_pos;
        int tap0 = pos;
        int tap1 = (pos + 1 < d_conv_m1) ? pos + 1 : pos + 1 - d_conv_m1;
        int tap2 = (pos + 2 < d_conv_m1) ? pos + 2 : pos + 2 - d_conv_m1;
        float *cs0 = b->conv_state + tap0 * d_inner;
        float *cs1 = b->conv_state + tap1 * d_inner;
        float *cs2 = b->conv_state + tap2 * d_inner;

        int in_proj_cols = b->in_proj.cols;
        int out_proj_cols = b->out_proj.cols;
        int effective = bm_get_threads();

        /* Work-stealing chunk setup: single-NUMA only (preserves locality
         * on multi-NUMA). chunk_size aligned to 8 for NEON / 4 for AVX2. */
        bool use_steal = (bm_get_numa_nodes() <= 1);
        int in_rows = b->in_proj.rows;
        int out_rows = b->out_proj.rows;
        int in_cs = in_rows / (effective * 4);
        if (in_cs < 8)
            in_cs = 8;
        int out_cs = out_rows / (effective * 4);
        if (out_cs < 8)
            out_cs = 8;

        fused_block_work_t work = {
            .b = b,
            .in_proj_rw =
                {
                    .x_quant = bl_x_quant,
                    .packed_data = b->in_proj.packed_data,
                    .out = proj_out,
                    .inv_scale = 1.0f / (in_scale * b->in_proj.scale),
                    .rows = in_rows,
                    .cols = in_proj_cols,
                    .packed_stride = (in_proj_cols + 3) / 4,
                },
            .proj_out = proj_out,
            .y = y,
            .cs0 = cs0,
            .cs1 = cs1,
            .cs2 = cs2,
            .cs_write = cs0, /* overwrite oldest tap */
            .out_proj_rw =
                {
                    .x_quant = NULL, /* filled by tid 0 after SSM */
                    .packed_data = b->out_proj.packed_data,
                    .out = out_buffer,
                    .inv_scale = 0.0f, /* filled by tid 0 */
                    .rows = out_rows,
                    .cols = out_proj_cols,
                    .packed_stride = (out_proj_cols + 3) / 4,
                },
            .out_norm_buf = bl_x_norm,
            .out_quant_buf = bl_x_quant,
            .out_n = d_inner,
            .n_threads = effective,
            .in_proj_packed_numa = b->in_proj_packed_numa,
            .out_proj_packed_numa = b->out_proj_packed_numa,
            .in_proj_steal_ctr = effective,
            .in_proj_chunk_size = use_steal ? in_cs : 0,
            .out_proj_steal_ctr = effective,
            .out_proj_chunk_size = use_steal ? out_cs : 0,
        };

        bm_barrier_init(&work.barrier, effective);
        bm_parallel_for(effective, fused_block_worker, &work);

        /* Advance ring-buffer position */
        b->conv_pos = (pos + 1 < d_conv_m1) ? pos + 1 : 0;
    } else {
        /* Unfused path (scalar or single-threaded) */
        g_kernels.bitlinear_forward(u, b->d_model, &b->in_proj,
                                    &b->in_proj_norm, proj_out, bl_x_norm,
                                    bl_x_quant);

        ssm_scan_token(b, proj_out, y);

        g_kernels.bitlinear_forward(y, b->d_inner, &b->out_proj,
                                    &b->out_proj_norm, out_buffer, bl_x_norm,
                                    bl_x_quant);
    }
}

/*
 * Batched forward pass for prefill: in_proj and out_proj are batched,
 * SSM scan remains sequential (state updates are serial by nature).
 */
void block_step_batch(bitmamba_block_t *b,
                      const float *u_batch,
                      int n_tokens,
                      float *out_batch,
                      float *proj_batch,
                      float *y_batch,
                      float *bl_x_norm,
                      int8_t *bl_x_quant_batch,
                      int quant_stride,
                      bool use_gpu)
{
    int proj_dim = b->in_proj.rows; /* 2*d_inner + 3*n_heads */

    /* Phase 1: Batched input projection */
#ifdef HAS_METAL
    if (use_gpu && bm_metal_available() && n_tokens >= 2) {
        bm_metal_bitlinear_forward_batch(
            u_batch, n_tokens, b->d_model, b->in_proj.packed_data,
            b->in_proj.rows, b->in_proj.cols, b->in_proj.scale,
            b->in_proj_norm.data, proj_batch, bl_x_norm, bl_x_quant_batch,
            quant_stride);
    } else
#endif
    {
        g_kernels.bitlinear_forward_batch(
            u_batch, n_tokens, b->d_model, &b->in_proj, &b->in_proj_norm,
            proj_batch, bl_x_norm, bl_x_quant_batch, quant_stride);
    }

    /* Phase 2: Batched SSM scan.
     * Single parallel dispatch over d_inner, workers loop tokens inside.
     * Eliminates (n_tokens - 1) fork-join round-trips and keeps each
     * thread's channel slice hot in L1 across all tokens.
     */
    if (b->d_inner >= BM_PAR_THRESHOLD && bm_get_threads() > 1 &&
        n_tokens > 1) {
        ssm_batch_work_t bw = {
            .b = b,
            .proj_batch = proj_batch,
            .y_batch = y_batch,
            .n_tokens = n_tokens,
            .proj_stride = proj_dim,
            .initial_conv_pos = b->conv_pos,
        };
        bm_parallel_for(b->d_inner, ssm_batch_worker, &bw);
        /* Advance ring-buffer position past all tokens */
        int d_conv_m1 = b->d_conv - 1;
        b->conv_pos = (b->conv_pos + n_tokens) % d_conv_m1;
    } else {
        for (int t = 0; t < n_tokens; t++)
            ssm_scan_token(b, proj_batch + t * proj_dim,
                           y_batch + t * b->d_inner);
    }

    /* Phase 3: Batched output projection */
#ifdef HAS_METAL
    if (use_gpu && bm_metal_available() && n_tokens >= 2) {
        bm_metal_bitlinear_forward_batch(
            y_batch, n_tokens, b->d_inner, b->out_proj.packed_data,
            b->out_proj.rows, b->out_proj.cols, b->out_proj.scale,
            b->out_proj_norm.data, out_batch, bl_x_norm, bl_x_quant_batch,
            quant_stride);
    } else
#endif
    {
        g_kernels.bitlinear_forward_batch(
            y_batch, n_tokens, b->d_inner, &b->out_proj, &b->out_proj_norm,
            out_batch, bl_x_norm, bl_x_quant_batch, quant_stride);
    }
}

void block_free(bitmamba_block_t *b)
{
    /* Free NUMA weight replicas (mmap'd separately, not from model file) */
    for (int i = 0; i < 8; i++) {
        if (b->in_proj_packed_numa[i]) {
            int stride = (b->in_proj.cols + 3) / 4;
            munmap(b->in_proj_packed_numa[i],
                   (size_t) b->in_proj.rows * stride);
            b->in_proj_packed_numa[i] = NULL;
        }
        if (b->out_proj_packed_numa[i]) {
            int stride = (b->out_proj.cols + 3) / 4;
            munmap(b->out_proj_packed_numa[i],
                   (size_t) b->out_proj.rows * stride);
            b->out_proj_packed_numa[i] = NULL;
        }
    }

    /* conv_state, ssm_state, A_precomp, conv1d_w_t are heap-allocated */
    free(b->conv1d_w_t);
    free(b->conv_state);
    free(b->ssm_state);
    free(b->A_precomp);

    /* Tensor data points into mmap'd region - don't free it.
     * Just null out pointers to avoid dangling references. */
    b->in_proj_norm.data = NULL;
    b->in_proj.packed_data = NULL;
    b->conv1d_w.data = NULL;
    b->conv1d_b.data = NULL;
    b->dt_bias.data = NULL;
    b->A_log.data = NULL;
    b->D.data = NULL;
    b->out_proj_norm.data = NULL;
    b->out_proj.packed_data = NULL;
}

void block_ssm_scan(bitmamba_block_t *b, float *proj_out, float *y)
{
    ssm_scan_token(b, proj_out, y);
}
