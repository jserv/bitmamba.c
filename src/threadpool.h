#pragma once

/*
 * Cache line size for alignment and false-sharing prevention.
 * Apple Silicon P-cores use 128-byte lines; x86 and most ARM use 64.
 * Override at build time with -DBM_CACHE_LINE=N.
 */
#ifndef BM_CACHE_LINE
#if defined(__APPLE__) && defined(__aarch64__)
#define BM_CACHE_LINE 128
#else
#define BM_CACHE_LINE 64
#endif
#endif

#define BM_PAR_THRESHOLD 64
#define BM_MAX_NUMA 8

typedef void (*bm_task_fn_t)(int tid, int n_threads, void *arg);

/*
 * Balanced work distribution: thread tid gets tasks [*start, *end).
 * For N tasks across T threads, threads 0..extra-1 get (base+1),
 * the rest get base. All threads do useful work when N >= T.
 */
static inline void bm_work_range(int tid,
                                 int n_threads,
                                 int n_tasks,
                                 int *start,
                                 int *end)
{
    /* Guard: workers with tid >= n_threads get an empty range.
     * This happens when bm_parallel_for caps n_active < pool size
     * but all pool workers still wake and call the task function.
     */
    if (tid >= n_threads) {
        *start = *end = n_tasks;
        return;
    }
    int base = n_tasks / n_threads;
    int extra = n_tasks % n_threads;
    if (tid < extra) {
        *start = tid * (base + 1);
        *end = *start + base + 1;
    } else {
        *start = extra * (base + 1) + (tid - extra) * base;
        *end = *start + base;
    }
}

void bm_set_threads(int n); /* 0=auto, 1=no pool, N=N threads */
int bm_get_threads(void);
int bm_get_numa_nodes(void);
int bm_get_node_threads(int node);
int bm_tid_node(int tid);
void bm_parallel_for(int n_tasks, bm_task_fn_t fn, void *arg);
void bm_thread_pool_free(void);

/* Poll intensity: 0=pure condvar, 100=pure spin (default). */
void bm_set_poll(int intensity);

/*
 * 2-level NUMA-aware in-dispatch barrier.
 *
 * Single-NUMA: flat phase-counting barrier (same cost as before).
 * Multi-NUMA: per-node local barriers + global leader barrier.
 *   - Workers only touch their node's cache line (no cross-socket traffic)
 *   - Node leaders (last to arrive locally) enter a tiny global barrier
 *   - Leaders release local waiters after global sync completes
 *
 * Call bm_barrier_init() before first use to set per-node thread counts.
 */
typedef struct {
    _Atomic unsigned int arrived;
    _Atomic unsigned int phase;
} __attribute__((aligned(BM_CACHE_LINE))) bm_phase_ctr_t;

typedef struct {
    bm_phase_ctr_t local[BM_MAX_NUMA];
    bm_phase_ctr_t global;
    int n_active_nodes;
    int node_n[BM_MAX_NUMA]; /* active threads per node */
} __attribute__((aligned(BM_CACHE_LINE))) bm_barrier_t;

void bm_barrier_init(bm_barrier_t *b, int n_threads);
void bm_barrier_wait(bm_barrier_t *b, int tid, int n_threads);
