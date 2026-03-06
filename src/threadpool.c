#ifdef __linux__
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include "threadpool.h"

/* Spin-wait hint: reduce power and improve SMT throughput */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define SPIN_PAUSE() __asm__ volatile("yield")
#else
#define SPIN_PAUSE() ((void) 0)
#endif

/* Use BM_CACHE_LINE from threadpool.h (128 Apple Silicon, 64 otherwise) */
#define CACHE_LINE BM_CACHE_LINE
#define BM_MAX_CPUS 256

/* Per-worker condvar sidecar (kept off worker_t cache line -- slow path only)
 */
typedef struct {
    pthread_mutex_t mtx;
    pthread_cond_t cond;
} __attribute__((aligned(CACHE_LINE))) worker_condvar_t;

/*
 * Per-worker struct, cache-line aligned.
 *
 * wake_gen: per-worker mailbox. Main writes the current generation to
 * each active worker's wake_gen (release store). Workers spin on their
 * own wake_gen (acquire load) -- zero cross-worker contention.
 *
 * last_done_gen: completion flag. Worker writes after fn() completes
 * (release store). Main reads during completion scan (acquire load).
 *
 * Both fields share the worker's private cache line. No false sharing
 * because each worker owns its 64-byte-aligned struct exclusively.
 */
typedef struct {
    pthread_t handle;
    int tid;
    int node; /* NUMA node this worker is pinned to */
    _Atomic uint64_t wake_gen;
    _Atomic uint64_t last_done_gen;
} __attribute__((aligned(CACHE_LINE))) worker_t;

/*
 * Thread pool with NUMA-aware layout and cache-line-separated hot fields.
 *
 * Structural fields (set once at pool creation, rarely read).
 * Shutdown flag on its own cache line (polled by all workers during spin).
 * Dispatch metadata on a third line (written by main, read after wake).
 *
 * Not thread-safe across callers: bm_parallel_for / bm_set_threads /
 * bm_thread_pool_free must be called from a single thread.
 */
typedef struct {
    /* Structural (set once, read-mostly) */
    worker_t *workers;
    worker_condvar_t *worker_cvs; /* [n_workers], NULL when poll=100 */
    int n_threads;                /* total threads including main */
    int n_workers;                /* n_threads - 1 (spawned threads) */
    int poll_intensity;           /* 0-100, default 100 (pure spin) */

    /* NUMA topology */
    int n_nodes;
    int node_threads[BM_MAX_NUMA]; /* threads per node (main counted on node 0)
                                    */
    int node_first_tid[BM_MAX_NUMA]; /* first tid assigned to each node */
    int tid_node[BM_MAX_CPUS];       /* NUMA node for each tid */

    char _pad1[CACHE_LINE];

    /* Shutdown flag (polled by all workers during spin-wait) */
    _Atomic int shutdown;

    char _pad2[CACHE_LINE - sizeof(_Atomic int)];

    /* Dispatch metadata (written by main, read by workers after wake) */
    bm_task_fn_t fn;
    void *arg;
    int n_active;     /* effective thread count for current dispatch */
    uint64_t cur_gen; /* monotonic generation counter (main-thread only) */
} thread_pool_t;

static thread_pool_t g_pool = {.poll_intensity = -1};

/* NUMA topology detection (Linux only) */

#ifdef __linux__
/* Parse a CPU list file like "0,2,4-8,10" into an array of CPU IDs. */
static int parse_cpulist(const char *path, int *cpus, int max_cpus)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    int count = 0;
    int lo, hi;
    while (count < max_cpus) {
        if (fscanf(f, "%d", &lo) != 1)
            break;
        int ch = fgetc(f);
        if (ch == '-') {
            if (fscanf(f, "%d", &hi) != 1)
                break;
            for (int c = lo; c <= hi && count < max_cpus; c++)
                cpus[count++] = c;
            ch = fgetc(f);
        } else {
            cpus[count++] = lo;
        }
        if (ch != ',')
            break;
    }
    fclose(f);
    return count;
}

typedef struct {
    int n_nodes;
    int node_cpu_count[BM_MAX_NUMA];
    int node_cpus[BM_MAX_NUMA][BM_MAX_CPUS];
} numa_topo_t;

static void detect_numa(numa_topo_t *topo)
{
    memset(topo, 0, sizeof(*topo));
    for (int node = 0; node < BM_MAX_NUMA; node++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist",
                 node);
        int cpus[BM_MAX_CPUS];
        int count = parse_cpulist(path, cpus, BM_MAX_CPUS);
        if (count == 0)
            break;
        topo->node_cpu_count[node] = count;
        memcpy(topo->node_cpus[node], cpus, (size_t) count * sizeof(int));
        topo->n_nodes++;
    }
    if (topo->n_nodes == 0)
        topo->n_nodes = 1;
}

static void pin_to_cpu(pthread_t thread, int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}
#endif /* __linux__ */

/* Worker entry point */

static void *worker_entry(void *arg)
{
    worker_t *w = (worker_t *) arg;
    uint64_t my_gen = 0;
    int worker_idx = w->tid - 1; /* workers[0] has tid=1, etc. */

    for (;;) {
        /* Hybrid spin/condvar wait on per-worker mailbox.
         * poll_intensity controls spin budget:
         *   100 = pure spin (current behavior, zero condvar overhead)
         *   0   = pure condvar (no spin, minimum power)
         *   1-99 = spin first, then condvar
         */
        int poll = g_pool.poll_intensity;
        int spin_budget = poll * 1280; /* 0=no spin, 100=128K spins */
        int spins = 0;

        while (atomic_load_explicit(&w->wake_gen, memory_order_relaxed) ==
               my_gen) {
            if (atomic_load_explicit(&g_pool.shutdown, memory_order_relaxed))
                return NULL;

            if (spins < spin_budget) {
                SPIN_PAUSE();
                spins++;
            } else if (poll < 100 && g_pool.worker_cvs) {
                /* Slow path: condvar wait */
                worker_condvar_t *cv = &g_pool.worker_cvs[worker_idx];
                pthread_mutex_lock(&cv->mtx);
                /* Re-check under lock to avoid lost wakeup */
                if (atomic_load_explicit(&w->wake_gen, memory_order_relaxed) ==
                        my_gen &&
                    !atomic_load_explicit(&g_pool.shutdown,
                                          memory_order_relaxed))
                    pthread_cond_wait(&cv->cond, &cv->mtx);
                pthread_mutex_unlock(&cv->mtx);
                /* Don't reset spins -- re-enter spin for quick re-check */
            } else {
                /* poll=100 without condvar: yield and retry (legacy behavior)
                 */
                sched_yield();
                spins = 0;
            }
        }
        my_gen = atomic_load_explicit(&w->wake_gen, memory_order_acquire);

        if (atomic_load_explicit(&g_pool.shutdown, memory_order_relaxed))
            return NULL;

        /* Execute work. All woken workers are active -- inactive workers
         * were never signaled and remain spinning on stale wake_gen.
         */
        g_pool.fn(w->tid, g_pool.n_active, g_pool.arg);

        /* Signal completion: write to own cache line, no contention.
         * Release ensures all fn() stores are visible before the flag.
         */
        atomic_store_explicit(&w->last_done_gen, my_gen, memory_order_release);
    }
}

/* Pool creation with NUMA-aware thread placement */

static void pool_create(int n_threads)
{
    if (g_pool.workers)
        return; /* already created */

    g_pool.n_threads = n_threads;
    g_pool.n_workers = n_threads - 1;
    atomic_store(&g_pool.shutdown, 0);
    g_pool.cur_gen = 0;
    if (g_pool.poll_intensity < 0)
        g_pool.poll_intensity = 100; /* default: pure spin */

    /* Aligned allocation: each worker_t occupies its own cache line */
    void *mem;
    if (posix_memalign(&mem, CACHE_LINE,
                       (size_t) g_pool.n_workers * sizeof(worker_t)) != 0) {
        g_pool.n_threads = 1;
        g_pool.n_workers = 0;
        return;
    }
    memset(mem, 0, (size_t) g_pool.n_workers * sizeof(worker_t));
    g_pool.workers = (worker_t *) mem;

    /* Allocate condvar sidecar (used when poll < 100) */
    void *cv_mem;
    if (posix_memalign(&cv_mem, CACHE_LINE,
                       (size_t) g_pool.n_workers * sizeof(worker_condvar_t)) ==
        0) {
        g_pool.worker_cvs = (worker_condvar_t *) cv_mem;
        for (int i = 0; i < g_pool.n_workers; i++) {
            pthread_mutex_init(&g_pool.worker_cvs[i].mtx, NULL);
            pthread_cond_init(&g_pool.worker_cvs[i].cond, NULL);
        }
    }

    /* Detect NUMA topology and plan thread placement */

#ifdef __linux__
    numa_topo_t topo;
    detect_numa(&topo);
    g_pool.n_nodes = topo.n_nodes;

    if (topo.n_nodes > 1) {
        /* Fill nodes sequentially: pack node 0 up to its CPU count,
         * then overflow to node 1, etc. This way --threads 24 on a
         * 2x24 system puts all 24 on node 0 (zero cross-socket traffic).
         * --threads 48 fills both nodes (24 each). The per-socket
         * fused-path cap in block_step then uses node 0 for layers
         * and all threads for the barrier-free LM head.
         */
        int remaining = n_threads;
        int tid_offset = 0;
        for (int node = 0; node < topo.n_nodes && remaining > 0; node++) {
            int capacity = topo.node_cpu_count[node];
            int per_node = remaining < capacity ? remaining : capacity;
            g_pool.node_threads[node] = per_node;
            g_pool.node_first_tid[node] = tid_offset;
            for (int t = 0; t < per_node && (tid_offset + t) < BM_MAX_CPUS; t++)
                g_pool.tid_node[tid_offset + t] = node;
            tid_offset += per_node;
            remaining -= per_node;
        }
    } else
#endif
    {
        g_pool.n_nodes = 1;
        g_pool.node_threads[0] = n_threads;
        g_pool.node_first_tid[0] = 0;
        for (int t = 0; t < n_threads && t < BM_MAX_CPUS; t++)
            g_pool.tid_node[t] = 0;
    }

    /* Create worker threads with NUMA-aware pinning */

    int created = 0;
    for (int i = 0; i < g_pool.n_workers; i++) {
        int tid = i + 1; /* main is tid 0 */
        g_pool.workers[i].tid = tid;
        g_pool.workers[i].node = g_pool.tid_node[tid];
        if (pthread_create(&g_pool.workers[i].handle, NULL, worker_entry,
                           &g_pool.workers[i]) != 0)
            break;
        created++;

#ifdef __linux__
        /* Pin worker to a CPU on its assigned NUMA node.
         * local_idx maps tid to the CPU list index within the node.
         * Main (tid 0) takes slot 0 on node 0 conceptually but is not pinned.
         */
        if (topo.n_nodes > 1) {
            int node = g_pool.tid_node[tid];
            int local_idx = tid - g_pool.node_first_tid[node];
            if (local_idx >= 0 && local_idx < topo.node_cpu_count[node])
                pin_to_cpu(g_pool.workers[i].handle,
                           topo.node_cpus[node][local_idx]);
        }
#endif
    }

    /* If some threads failed to start, adjust counts to avoid
     * barrier deadlock (main would wait forever for missing workers).
     */
    if (created < g_pool.n_workers) {
        /* Destroy condvars for workers that were never started */
        if (g_pool.worker_cvs) {
            for (int i = created; i < g_pool.n_workers; i++) {
                pthread_mutex_destroy(&g_pool.worker_cvs[i].mtx);
                pthread_cond_destroy(&g_pool.worker_cvs[i].cond);
            }
        }
        g_pool.n_workers = created;
        g_pool.n_threads = created + 1;
    }
    if (g_pool.n_workers == 0) {
        free(g_pool.workers);
        g_pool.workers = NULL;
        /* Clean up condvar sidecar if allocated */
        if (g_pool.worker_cvs) {
            free(g_pool.worker_cvs);
            g_pool.worker_cvs = NULL;
        }
        g_pool.n_threads = 1;
    }

    /* Recalculate per-node thread counts if creation was partial */
#ifdef __linux__
    if (g_pool.n_threads < n_threads && g_pool.n_nodes > 1) {
        int remaining = g_pool.n_threads;
        int tid_offset = 0;
        for (int node = 0; node < g_pool.n_nodes && remaining > 0; node++) {
            int capacity = topo.node_cpu_count[node];
            int per_node = remaining < capacity ? remaining : capacity;
            g_pool.node_threads[node] = per_node;
            g_pool.node_first_tid[node] = tid_offset;
            for (int t = 0; t < per_node && (tid_offset + t) < BM_MAX_CPUS; t++)
                g_pool.tid_node[tid_offset + t] = node;
            tid_offset += per_node;
            remaining -= per_node;
        }
    }
#endif

#ifdef __linux__
    if (g_pool.n_nodes > 1 && g_pool.n_threads > 1) {
        /* Pin main thread to first CPU on node 0 to prevent migration */
        pin_to_cpu(pthread_self(), topo.node_cpus[0][0]);
    }
#endif
}

/* Public API */

void bm_set_threads(int n)
{
    if (n <= 0) {
#ifdef __APPLE__
        int ncpu = 0;
        size_t len = sizeof(ncpu);
        /* Prefer performance cores only on big.LITTLE Apple Silicon.
         * E-cores bottleneck spin-wait barriers at half the P-core frequency.
         */
        if (sysctlbyname("hw.perflevel0.logicalcpu", &ncpu, &len, NULL, 0) !=
                0 ||
            ncpu <= 0) {
            len = sizeof(ncpu);
            if (sysctlbyname("hw.logicalcpu", &ncpu, &len, NULL, 0) != 0 ||
                ncpu <= 0)
                ncpu = 1;
        }
        n = ncpu;
#else
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        n = (ncpu > 0) ? (int) ncpu : 1;
#endif
    }
    if (n > BM_MAX_CPUS)
        n = BM_MAX_CPUS;
    if (n == 1) {
        /* Tear down existing workers if switching to single-thread */
        bm_thread_pool_free();
        g_pool.n_threads = 1;
        return;
    }
    pool_create(n);
}

int bm_get_threads(void)
{
    return g_pool.n_threads > 0 ? g_pool.n_threads : 1;
}

int bm_get_numa_nodes(void)
{
    return g_pool.n_nodes > 0 ? g_pool.n_nodes : 1;
}

int bm_get_node_threads(int node)
{
    if (node < 0 || node >= g_pool.n_nodes)
        return 0;
    return g_pool.node_threads[node];
}

int bm_tid_node(int tid)
{
    if (g_pool.n_nodes <= 1 || tid < 0 || tid >= g_pool.n_threads)
        return 0;
    if (tid >= BM_MAX_CPUS)
        return 0;
    return g_pool.tid_node[tid];
}

void bm_parallel_for(int n_tasks, bm_task_fn_t fn, void *arg)
{
    if (g_pool.n_threads <= 1 || !g_pool.workers) {
        fn(0, 1, arg);
        return;
    }

    /* Cap active threads to n_tasks: no point waking 48 threads for
     * 64 heads of SSM work. Workers not woken stay spinning on stale
     * wake_gen with zero overhead to the dispatch.
     */
    int effective = g_pool.n_threads;
    if (n_tasks > 0 && n_tasks < effective)
        effective = n_tasks;

    /* Fast path: if only main thread is needed, skip pool round-trip */
    if (effective <= 1) {
        fn(0, 1, arg);
        return;
    }

    /* Publish work (plain stores, ordered by release on wake_gen below) */
    g_pool.fn = fn;
    g_pool.arg = arg;
    g_pool.n_active = effective;

    /* Bump generation and wake only active workers via per-worker
     * mailboxes. Each release store ensures fn/arg/n_active are visible
     * to the worker that does the matching acquire load on wake_gen.
     * Inactive workers are never signaled -- zero thundering-herd.
     */
    uint64_t gen = ++g_pool.cur_gen;
    for (int i = 0; i < effective - 1; i++)
        atomic_store_explicit(&g_pool.workers[i].wake_gen, gen,
                              memory_order_release);

    /* Signal condvars for workers that may be sleeping (poll < 100).
     * Per-worker signal avoids thundering herd. When poll=100, worker_cvs
     * is allocated but never waited on, so skip the overhead entirely. */
    if (g_pool.poll_intensity < 100 && g_pool.worker_cvs) {
        for (int i = 0; i < effective - 1; i++) {
            pthread_mutex_lock(&g_pool.worker_cvs[i].mtx);
            pthread_cond_signal(&g_pool.worker_cvs[i].cond);
            pthread_mutex_unlock(&g_pool.worker_cvs[i].mtx);
        }
    }

    /* Main thread does tid=0 */
    fn(0, effective, arg);

    /* Wait only for workers that were actually woken.
     * Each worker writes its own cache-line-aligned last_done_gen,
     * so this loop only reads -- no write contention at all.
     * Relaxed loads in spin body: main doesn't touch output buffers
     * until ALL workers finish. Single acquire fence after the loop
     * synchronizes-with every worker's release store.
     */
    for (int i = 0; i < effective - 1; i++) {
        while (atomic_load_explicit(&g_pool.workers[i].last_done_gen,
                                    memory_order_relaxed) != gen)
            SPIN_PAUSE();
    }
    atomic_thread_fence(memory_order_acquire);
}

void bm_set_poll(int intensity)
{
    if (intensity < 0)
        intensity = 0;
    if (intensity > 100)
        intensity = 100;
    g_pool.poll_intensity = intensity;
}

void bm_thread_pool_free(void)
{
    if (!g_pool.workers)
        return;

    atomic_store_explicit(&g_pool.shutdown, 1, memory_order_relaxed);
    /* Wake ALL workers so they can see the shutdown flag */
    uint64_t gen = ++g_pool.cur_gen;
    for (int i = 0; i < g_pool.n_workers; i++)
        atomic_store_explicit(&g_pool.workers[i].wake_gen, gen,
                              memory_order_release);

    /* Signal condvars so sleeping workers wake up to see shutdown */
    if (g_pool.worker_cvs) {
        for (int i = 0; i < g_pool.n_workers; i++) {
            pthread_mutex_lock(&g_pool.worker_cvs[i].mtx);
            pthread_cond_signal(&g_pool.worker_cvs[i].cond);
            pthread_mutex_unlock(&g_pool.worker_cvs[i].mtx);
        }
    }

    for (int i = 0; i < g_pool.n_workers; i++)
        pthread_join(g_pool.workers[i].handle, NULL);

    /* Destroy condvar resources */
    if (g_pool.worker_cvs) {
        for (int i = 0; i < g_pool.n_workers; i++) {
            pthread_mutex_destroy(&g_pool.worker_cvs[i].mtx);
            pthread_cond_destroy(&g_pool.worker_cvs[i].cond);
        }
        free(g_pool.worker_cvs);
        g_pool.worker_cvs = NULL;
    }

    free(g_pool.workers);
    g_pool.workers = NULL;
    g_pool.n_threads = 0;
    g_pool.n_workers = 0;
}

/* 2-level NUMA-aware in-dispatch barrier */

/*
 * Initialize barrier for n_threads active threads.
 * Computes per-node thread counts from the pool's NUMA mapping.
 * Must be called before first bm_barrier_wait on this barrier.
 */
void bm_barrier_init(bm_barrier_t *b, int n_threads)
{
    memset(b, 0, sizeof(*b));

    if (g_pool.n_nodes <= 1) {
        /* Single-NUMA: flat barrier */
        b->n_active_nodes = 1;
        b->node_n[0] = n_threads;
        return;
    }

    /* Count how many of the active tids [0, n_threads) fall on each node.
     * With contiguous assignment, node K owns tids [first_tid[K],
     * first_tid[K]+node_threads[K]).
     */
    b->n_active_nodes = 0;
    for (int node = 0; node < g_pool.n_nodes; node++) {
        int first = g_pool.node_first_tid[node];
        int last = first + g_pool.node_threads[node];
        int lo = first > 0 ? first : 0;
        int hi = last < n_threads ? last : n_threads;
        int count = (hi > lo) ? hi - lo : 0;
        b->node_n[node] = count;
        if (count > 0)
            b->n_active_nodes++;
    }
}

void bm_barrier_wait(bm_barrier_t *b, int tid, int n_threads)
{
    if (b->n_active_nodes <= 1) {
        /* Flat barrier: all threads on same NUMA node.
         * Identical to the original phase-counting barrier.
         */
        unsigned int cur =
            atomic_load_explicit(&b->local[0].phase, memory_order_relaxed);
        unsigned int idx = atomic_fetch_add_explicit(&b->local[0].arrived, 1,
                                                     memory_order_acq_rel);

        if (idx == (unsigned int) (n_threads - 1)) {
            atomic_store_explicit(&b->local[0].arrived, 0,
                                  memory_order_relaxed);
            atomic_store_explicit(&b->local[0].phase, cur + 1,
                                  memory_order_release);
        } else {
            while (atomic_load_explicit(&b->local[0].phase,
                                        memory_order_relaxed) == cur)
                SPIN_PAUSE();
            atomic_thread_fence(memory_order_acquire);
        }
        return;
    }

    /* 2-level NUMA barrier:
     * 1. Workers barrier on their node's local counter (node-local cache line)
     * 2. Last thread on each node (leader) enters global barrier (2 leaders)
     * 3. Leaders release their local waiters after global sync
     *
     * Cross-socket atomics: 2 fetch_add + 2 phase loads (leaders only)
     * vs flat: N fetch_add + N phase loads (all threads, all cross-socket)
     */
    int node = g_pool.tid_node[tid];
    int local_n = b->node_n[node];

    /* Local barrier: arrive on node-local counter */
    unsigned int local_cur =
        atomic_load_explicit(&b->local[node].phase, memory_order_relaxed);
    unsigned int local_idx = atomic_fetch_add_explicit(&b->local[node].arrived,
                                                       1, memory_order_acq_rel);

    if (local_idx == (unsigned int) (local_n - 1)) {
        /* Node leader: reset local counter, enter global barrier */
        atomic_store_explicit(&b->local[node].arrived, 0, memory_order_relaxed);

        unsigned int global_cur =
            atomic_load_explicit(&b->global.phase, memory_order_relaxed);
        unsigned int global_idx = atomic_fetch_add_explicit(
            &b->global.arrived, 1, memory_order_acq_rel);

        if (global_idx == (unsigned int) (b->n_active_nodes - 1)) {
            /* Last leader: release all */
            atomic_store_explicit(&b->global.arrived, 0, memory_order_relaxed);
            atomic_store_explicit(&b->global.phase, global_cur + 1,
                                  memory_order_release);
        } else {
            while (atomic_load_explicit(&b->global.phase,
                                        memory_order_relaxed) == global_cur)
                SPIN_PAUSE();
            atomic_thread_fence(memory_order_acquire);
        }

        /* Release local waiters */
        atomic_store_explicit(&b->local[node].phase, local_cur + 1,
                              memory_order_release);
    } else {
        /* Spin on local phase (node-local cache line, no cross-socket traffic).
         * Relaxed loads in spin body; acquire fence after exit synchronizes
         * with leader's release store to local phase.
         */
        while (atomic_load_explicit(&b->local[node].phase,
                                    memory_order_relaxed) == local_cur)
            SPIN_PAUSE();
        atomic_thread_fence(memory_order_acquire);
    }
}
