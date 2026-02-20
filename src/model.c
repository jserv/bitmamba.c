#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#endif

#include "bitmamba.h"
#include "threadpool.h"
#ifdef HAS_METAL
#include "metal/bm_metal.h"
#endif

/* --- Safe Unaligned Read Helpers ---
 * Avoid UB from casting unaligned uint8_t* to int* or float*.
 * The compiler will optimize memcpy to single load on aligned architectures. */
static inline int read_int_unaligned(const uint8_t *ptr)
{
    int v;
    memcpy(&v, ptr, sizeof(int));
    return v;
}

static inline float read_float_unaligned(const uint8_t *ptr)
{
    float v;
    memcpy(&v, ptr, sizeof(float));
    return v;
}

/* Comparison for qsort (descending) */
static int compare_token_prob_desc(const void *a, const void *b)
{
    float diff =
        ((const token_prob_t *) b)->val - ((const token_prob_t *) a)->val;
    return (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);
}

/*
 * Min-heap helpers for partial top-k selection.
 * Maintains a heap of size k with the k largest elements seen so far.
 * The root (index 0) is the smallest element in the heap.
 */
static inline void heap_sift_down(token_prob_t *h, int n, int i)
{
    while (1) {
        int smallest = i;
        int l = 2 * i + 1, r = 2 * i + 2;
        if (l < n && h[l].val < h[smallest].val)
            smallest = l;
        if (r < n && h[r].val < h[smallest].val)
            smallest = r;
        if (smallest == i)
            break;
        token_prob_t tmp = h[i];
        h[i] = h[smallest];
        h[smallest] = tmp;
        i = smallest;
    }
}

/*
 * Apply min_p + top_p filtering, renormalize, and sample from the
 * sorted probability distribution probs[0..n_probs-1] (descending).
 */
static int filter_and_sample(token_prob_t *probs,
                             int n_probs,
                             float min_p,
                             float top_p,
                             int top_k)
{
    /* min_p filtering */
    if (min_p > 0.0f) {
        float thr = probs[0].val * min_p;
        for (int i = 1; i < n_probs; i++) {
            if (probs[i].val < thr) {
                n_probs = i;
                break;
            }
        }
    }

    /* top_p (nucleus) filtering */
    if (top_p < 1.0f) {
        double cs = 0.0;
        for (int i = 0; i < n_probs; i++) {
            cs += probs[i].val;
            if (cs >= top_p) {
                n_probs = i + 1;
                break;
            }
        }
    }

    /* top_k filtering */
    if (top_k > 0 && top_k < n_probs)
        n_probs = top_k;

    /* Renormalize and CDF sample */
    double new_sum = 0.0;
    for (int i = 0; i < n_probs; i++)
        new_sum += probs[i].val;

    float r = rng_float() * (float) new_sum;
    double cdf = 0.0;
    for (int i = 0; i < n_probs; i++) {
        cdf += probs[i].val;
        if (r < cdf)
            return probs[i].id;
    }
    return probs[n_probs - 1].id;
}

/*
 * Advanced sampler: temperature + min_p + top_p + top_k + CDF sampling.
 * Uses XorShift64 PRNG for high-quality randomness.
 *
 * Optimizations vs naive softmax-then-sort:
 * 1. Fused temperature + max scan: one O(V) pass instead of two.
 * 2. Pre-softmax min_p: threshold = max_logit + logf(min_p) prunes before
 *    sort. Mathematically equivalent to post-softmax filter (monotonicity).
 * 3. Bucket histogram (128 buckets): O(V) pass finds approximate top-k
 *    cutoff, avoiding full sort of 50257 elements.
 * 4. Softmax-after-top-k: expf only on ~k survivors (40 vs 50257 calls).
 */
enum { HIST_BUCKETS = 128 };
static const float HIST_RANGE = 20.0f; /* logit range: [max-20, max] */

static int sample_advanced(float *logits,
                           int vocab_size,
                           float temp,
                           float min_p,
                           float top_p,
                           int top_k,
                           token_prob_t *probs)
{
    /* Greedy if temp is very low */
    if (temp < 0.05f) {
        int best = 0;
        float max_l = -1e9f;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best = i;
            }
        }
        return best;
    }

    /* Step 1: Fused temperature scaling + max logit scan (single O(V) pass) */
    float inv_temp = 1.0f / temp;
    float max_logit = -1e30f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }

    /* Step 2: Pre-softmax min_p threshold (equivalent to post-softmax filter).
     * Proof: p_i >= min_p * p_max iff exp(l_i)/Z >= min_p * exp(l_max)/Z
     *        iff l_i >= l_max + log(min_p).
     * When min_p >= 1.0, only the max-logit token survives. */
    float min_p_thresh = -1e30f;
    if (min_p >= 1.0f)
        min_p_thresh = max_logit;
    else if (min_p > 0.0f)
        min_p_thresh = max_logit + logf(min_p);

    int effective_k = (top_k > 0 && top_k <= 256) ? top_k : 0;

    /* When top_k is active, use bucket histogram to find cutoff */
    if (effective_k > 0) {
        /* Step 3: Bucket histogram over logits passing min_p threshold.
         * 128 buckets spanning [max_logit - 20, max_logit]. */
        int hist[HIST_BUCKETS];
        memset(hist, 0, sizeof(hist));
        float bucket_scale = (float) (HIST_BUCKETS - 1) / HIST_RANGE;

        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] >= min_p_thresh) {
                float delta = max_logit - logits[i];
                int b = (int) (delta * bucket_scale);
                if (b >= HIST_BUCKETS)
                    b = HIST_BUCKETS - 1;
                if (b < 0)
                    b = 0;
                hist[b]++;
            }
        }

        /* Step 4: Scan histogram from top to find cutoff bucket where
         * accumulated count >= effective_k. */
        int accum = 0;
        int cutoff_bucket = HIST_BUCKETS - 1;
        for (int b = 0; b < HIST_BUCKETS; b++) {
            accum += hist[b];
            if (accum >= effective_k) {
                cutoff_bucket = b;
                break;
            }
        }

        /* Convert bucket to logit threshold */
        float bucket_thresh =
            max_logit - (float) (cutoff_bucket + 1) / bucket_scale;
        /* Take the tighter of bucket_thresh and min_p_thresh */
        float extract_thresh =
            bucket_thresh > min_p_thresh ? bucket_thresh : min_p_thresh;

        /* Step 5: Extract candidates above threshold */
        int n_cand = 0;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] >= extract_thresh) {
                probs[n_cand].id = i;
                probs[n_cand].val = logits[i];
                n_cand++;
            }
        }

        /* Guard: if histogram + threshold pruned everything, return argmax */
        if (n_cand == 0) {
            probs[0].id = 0;
            probs[0].val = logits[0];
            for (int i = 1; i < vocab_size; i++) {
                if (logits[i] > probs[0].val) {
                    probs[0].id = i;
                    probs[0].val = logits[i];
                }
            }
            return probs[0].id;
        }

        /* Step 6: If too many candidates, use min-heap to select top-k */
        int k = effective_k;
        if (n_cand > k) {
            /* Build min-heap from first k elements (keyed on logit value) */
            for (int i = k / 2 - 1; i >= 0; i--)
                heap_sift_down(probs, k, i);

            /* Scan remaining, replacing root if larger */
            for (int i = k; i < n_cand; i++) {
                if (probs[i].val > probs[0].val) {
                    probs[0] = probs[i];
                    heap_sift_down(probs, k, 0);
                }
            }
            n_cand = k;
        }

        /* Step 7: Softmax on survivors only (k expf calls, not V) */
        float local_max = -1e30f;
        for (int i = 0; i < n_cand; i++) {
            if (probs[i].val > local_max)
                local_max = probs[i].val;
        }
        double sum_exp = 0.0;
        for (int i = 0; i < n_cand; i++) {
            probs[i].val = expf(probs[i].val - local_max);
            sum_exp += probs[i].val;
        }
        float inv_sum = 1.0f / (float) sum_exp;
        for (int i = 0; i < n_cand; i++)
            probs[i].val *= inv_sum;

        /* Step 8: Sort k candidates descending */
        qsort(probs, n_cand, sizeof(token_prob_t), compare_token_prob_desc);

        /* Step 9: filter_and_sample with min_p=0 (already filtered) */
        return filter_and_sample(probs, n_cand, 0, top_p, 0);
    }

    /* Fallback path: no top_k (or large k).
     * Still benefits from fused scan and min_p pre-filter. */

    /* Extract candidates passing min_p threshold with logit values */
    int n_cand = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] >= min_p_thresh) {
            probs[n_cand].id = i;
            probs[n_cand].val = logits[i];
            n_cand++;
        }
    }

    /* Guard: if threshold pruned everything, return argmax */
    if (n_cand == 0) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best])
                best = i;
        }
        return best;
    }

    /* Softmax on filtered candidates */
    double sum_exp = 0.0;
    for (int i = 0; i < n_cand; i++) {
        probs[i].val = expf(probs[i].val - max_logit);
        sum_exp += probs[i].val;
    }
    float inv_sum = 1.0f / (float) sum_exp;
    for (int i = 0; i < n_cand; i++)
        probs[i].val *= inv_sum;

    /* Full sort of filtered set + sampling */
    qsort(probs, n_cand, sizeof(token_prob_t), compare_token_prob_desc);
    return filter_and_sample(probs, n_cand, 0, top_p, top_k);
}

/* Tensor reading from mmap'd memory */

/* Read tensor from mmap'd memory with bounds checking.
 * Returns bytes consumed, or 0 on error. */
static size_t read_tensor_mmap(const uint8_t *ptr,
                               const uint8_t *end,
                               tensor_t *t)
{
    memset(t, 0, sizeof(tensor_t));
    t->scale = 1.0f;
    const uint8_t *start = ptr;

    /* Check minimum header size */
    if (ptr + sizeof(int) > end)
        return 0;

    int type = read_int_unaligned(ptr);
    ptr += sizeof(int);

    if (type == 2) {
        /* BitNet packed weights */
        if (ptr + 3 * sizeof(int) > end)
            return 0;

        t->is_bitnet = true;
        t->rows = read_int_unaligned(ptr);
        ptr += sizeof(int);
        t->cols = read_int_unaligned(ptr);
        ptr += sizeof(int);
        t->scale = read_float_unaligned(ptr);
        ptr += sizeof(float);

        /* Validate scale: must be finite and positive for inv_scale math */
        if (!isfinite(t->scale) || t->scale <= 0.0f) {
            fprintf(stderr, "Error: Invalid bitnet scale %g\n",
                    (double) t->scale);
            return 0;
        }

        /* Validate tensor dimensions */
        if (t->rows <= 0 || t->rows > MAX_TENSOR_DIM || t->cols <= 0 ||
            t->cols > MAX_TENSOR_DIM) {
            fprintf(stderr, "Error: Invalid tensor dims %dx%d\n", t->rows,
                    t->cols);
            return 0;
        }

        int packed_cols = (t->cols + 3) / 4;

        /* Check for size_t overflow before multiplication */
        if (packed_cols > 0 &&
            (size_t) t->rows > SIZE_MAX / (size_t) packed_cols) {
            fprintf(stderr, "Error: Tensor size overflow %dx%d\n", t->rows,
                    packed_cols);
            return 0;
        }
        size_t total_bytes = (size_t) t->rows * packed_cols;

        if (total_bytes > (size_t) (end - ptr))
            return 0;

        /* Point directly to mmap'd data (zero copy) */
        t->packed_data = ptr;
        t->n_elements = (size_t) t->rows * t->cols;
        ptr += total_bytes;
    } else {
        /* Float32 weights */
        if (ptr + sizeof(int) > end)
            return 0;

        int ndim = read_int_unaligned(ptr);
        ptr += sizeof(int);

        if (ndim < 0 || ndim > 4)
            return 0;
        if (ptr + ndim * sizeof(int) > end)
            return 0;

        size_t total_size = 1;
        t->rows = 0;
        t->cols = 0;
        for (int i = 0; i < ndim; i++) {
            int d = read_int_unaligned(ptr);
            ptr += sizeof(int);
            /* Validate dimension size */
            if (d <= 0 || d > MAX_TENSOR_DIM)
                return 0;
            /* Check for overflow before multiplication */
            if (total_size > SIZE_MAX / (size_t) d) {
                fprintf(stderr, "Error: Float tensor size overflow\n");
                return 0;
            }
            total_size *= (size_t) d;
            if (i == 0)
                t->rows = d;
            if (i == 1)
                t->cols = d;
        }

        /* Check for overflow with sizeof(float) */
        if (total_size > SIZE_MAX / sizeof(float))
            return 0;
        if (ptr + total_size * sizeof(float) > end)
            return 0;

        /* Point directly to mmap'd data (zero copy) */
        t->data = (const float *) ptr;
        t->n_elements = total_size;
        ptr += total_size * sizeof(float);
    }

    return (size_t) (ptr - start);
}

bool model_load(bitmamba_model_t *m, const char *path)
{
    /* Preserve caller-set flags across zero-init */
    bool use_gpu = m->use_gpu;
    memset(m, 0, sizeof(*m));
    m->use_gpu = use_gpu;
    init_lut();

    /* Open file and get size */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open model file: %s\n", path);
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        close(fd);
        return false;
    }
    m->mmap_size = sb.st_size;

    /* Memory map the file.
     * Metal shared mode requires writable pages; MAP_PRIVATE gives COW
     * semantics so no actual extra memory is consumed. */
    int mmap_prot = PROT_READ;
#ifdef HAS_METAL
    if (m->use_gpu)
        mmap_prot |= PROT_WRITE;
#endif
    m->mmap_addr = mmap(NULL, m->mmap_size, mmap_prot, MAP_PRIVATE, fd, 0);
    close(fd);

    if (m->mmap_addr == MAP_FAILED) {
        fprintf(stderr, "Error: mmap failed\n");
        m->mmap_addr = NULL;
        return false;
    }

    /* Parse header with bounds checking */
    const uint8_t *ptr = (const uint8_t *) m->mmap_addr;
    const uint8_t *end = ptr + m->mmap_size;

    /* Minimum header: magic + 4 config ints */
    if (m->mmap_size < 5 * sizeof(int)) {
        fprintf(stderr, "Error: Model file too small\n");
        munmap(m->mmap_addr, m->mmap_size);
        m->mmap_addr = NULL;
        return false;
    }

    int magic = read_int_unaligned(ptr);
    ptr += sizeof(int);

    if (magic != MAGIC_NUMBER) {
        fprintf(stderr, "Error: Wrong format (need Packed .bin)\n");
        munmap(m->mmap_addr, m->mmap_size);
        m->mmap_addr = NULL;
        return false;
    }

    m->config.vocab_size = read_int_unaligned(ptr);
    ptr += sizeof(int);
    m->config.d_model = read_int_unaligned(ptr);
    ptr += sizeof(int);
    m->config.n_layers = read_int_unaligned(ptr);
    ptr += sizeof(int);
    m->config.n_heads = read_int_unaligned(ptr);
    ptr += sizeof(int);

    /* Validate config with security bounds */
    if (m->config.vocab_size <= 0 || m->config.vocab_size > MAX_VOCAB_SIZE ||
        m->config.d_model <= 0 || m->config.d_model > MAX_TENSOR_DIM ||
        m->config.n_layers <= 0 || m->config.n_layers > 256 ||
        m->config.n_heads <= 0 || m->config.n_heads > MAX_TENSOR_DIM) {
        fprintf(stderr,
                "Error: Invalid model config "
                "(vocab=%d, d_model=%d, n_layers=%d, n_heads=%d)\n",
                m->config.vocab_size, m->config.d_model, m->config.n_layers,
                m->config.n_heads);
        munmap(m->mmap_addr, m->mmap_size);
        m->mmap_addr = NULL;
        return false;
    }

    /* Read embedding tensor */
    size_t consumed = read_tensor_mmap(ptr, end, &m->embed);
    if (consumed == 0) {
        fprintf(stderr, "Error: Failed to read embedding tensor\n");
        munmap(m->mmap_addr, m->mmap_size);
        m->mmap_addr = NULL;
        return false;
    }
    ptr += consumed;

    /* Read layer tensors (zero-init so model_free handles partial init) */
    m->layers = xcalloc(m->config.n_layers, sizeof(bitmamba_block_t));

#define READ_TENSOR(tensor)                            \
    do {                                               \
        consumed = read_tensor_mmap(ptr, end, tensor); \
        if (consumed == 0)                             \
            goto parse_error;                          \
        ptr += consumed;                               \
    } while (0)

    for (int i = 0; i < m->config.n_layers; i++) {
        block_init_cache(&m->layers[i], m->config.d_model, m->config.n_heads);

        READ_TENSOR(&m->layers[i].in_proj_norm);
        READ_TENSOR(&m->layers[i].in_proj);
        READ_TENSOR(&m->layers[i].conv1d_w);
        READ_TENSOR(&m->layers[i].conv1d_b);
        READ_TENSOR(&m->layers[i].dt_bias);
        READ_TENSOR(&m->layers[i].A_log);
        READ_TENSOR(&m->layers[i].D);
        READ_TENSOR(&m->layers[i].out_proj_norm);
        READ_TENSOR(&m->layers[i].out_proj);
    }

    /* Read final tensors */
    READ_TENSOR(&m->norm_f);
    READ_TENSOR(&m->lm_head_norm);
    READ_TENSOR(&m->lm_head);

#undef READ_TENSOR

    /* Validate critical tensor shapes */
    int d_model = m->config.d_model;
    int d_inner = d_model * BM_EXPAND_FACTOR;

    /* Norm tensors must be >= d_model */
    if (m->norm_f.n_elements < (size_t) d_model) {
        fprintf(stderr, "Error: norm_f size %zu < d_model %d\n",
                m->norm_f.n_elements, d_model);
        goto parse_error;
    }
    if (m->lm_head_norm.n_elements < (size_t) d_model) {
        fprintf(stderr, "Error: lm_head_norm size %zu < d_model %d\n",
                m->lm_head_norm.n_elements, d_model);
        goto parse_error;
    }
    for (int i = 0; i < m->config.n_layers; i++) {
        if (m->layers[i].in_proj_norm.n_elements < (size_t) d_model) {
            fprintf(stderr,
                    "Error: Layer %d in_proj_norm size %zu < d_model %d\n", i,
                    m->layers[i].in_proj_norm.n_elements, d_model);
            goto parse_error;
        }
    }

    /* Embedding must have vocab_size * d_model elements */
    {
        size_t expected = (size_t) m->config.vocab_size * d_model;
        if (m->embed.n_elements < expected) {
            fprintf(stderr, "Error: embed size %zu < vocab*d_model %zu\n",
                    m->embed.n_elements, expected);
            goto parse_error;
        }
    }

    /* LM head must project d_model -> vocab_size exactly.
     * bitlinear_forward writes w->rows outputs, so rows must equal
     * vocab_size to avoid OOB writes into m->logits[vocab_size]. */
    if (m->lm_head.rows != m->config.vocab_size || m->lm_head.cols != d_model) {
        fprintf(stderr,
                "Error: lm_head shape mismatch "
                "(have %dx%d, need %dx%d)\n",
                m->lm_head.rows, m->lm_head.cols, m->config.vocab_size,
                d_model);
        goto parse_error;
    }

    /* d_inner must be divisible by n_heads */
    if (m->config.n_heads <= 0 || d_inner % m->config.n_heads != 0) {
        fprintf(stderr, "Error: d_inner=%d not divisible by n_heads=%d\n",
                d_inner, m->config.n_heads);
        goto parse_error;
    }

    /* in_proj must have cols == d_model (input dim) and
     * rows == 2*d_inner + 3*n_heads (block_step output slicing) */
    {
        int expected_proj = 2 * d_inner + 3 * m->config.n_heads;
        for (int i = 0; i < m->config.n_layers; i++) {
            if (m->layers[i].in_proj.cols != d_model) {
                fprintf(stderr,
                        "Error: Layer %d in_proj.cols=%d, expected %d\n", i,
                        m->layers[i].in_proj.cols, d_model);
                goto parse_error;
            }
            if (m->layers[i].in_proj.rows != expected_proj) {
                fprintf(stderr,
                        "Error: Layer %d in_proj.rows=%d, expected %d\n", i,
                        m->layers[i].in_proj.rows, expected_proj);
                goto parse_error;
            }
        }
    }

    /* out_proj must project d_inner -> d_model.
     * block_step writes w->rows outputs into d_model-sized buffers. */
    for (int i = 0; i < m->config.n_layers; i++) {
        if (m->layers[i].out_proj.rows != d_model ||
            m->layers[i].out_proj.cols != d_inner) {
            fprintf(stderr,
                    "Error: Layer %d out_proj shape %dx%d, expected %dx%d\n", i,
                    m->layers[i].out_proj.rows, m->layers[i].out_proj.cols,
                    d_model, d_inner);
            goto parse_error;
        }
    }

    /* Per-layer float tensor element counts (n_elements handles ndim>2) */
    {
        int d_conv = BM_CONV_KERNEL;
        int n_heads = m->config.n_heads;
        for (int i = 0; i < m->config.n_layers; i++) {
            bitmamba_block_t *b = &m->layers[i];
            if (b->conv1d_w.n_elements < (size_t) d_inner * d_conv) {
                fprintf(stderr, "Error: Layer %d conv1d_w size %zu < %zu\n", i,
                        b->conv1d_w.n_elements,
                        (size_t) d_inner * d_conv);
                goto parse_error;
            }
            if (b->conv1d_b.n_elements < (size_t) d_inner) {
                fprintf(stderr, "Error: Layer %d conv1d_b size %zu < %d\n", i,
                        b->conv1d_b.n_elements, d_inner);
                goto parse_error;
            }
            if (b->dt_bias.n_elements < (size_t) n_heads) {
                fprintf(stderr, "Error: Layer %d dt_bias size %zu < %d\n", i,
                        b->dt_bias.n_elements, n_heads);
                goto parse_error;
            }
            if (b->A_log.n_elements < (size_t) n_heads) {
                fprintf(stderr, "Error: Layer %d A_log size %zu < %d\n", i,
                        b->A_log.n_elements, n_heads);
                goto parse_error;
            }
            if (b->D.n_elements < (size_t) d_inner) {
                fprintf(stderr, "Error: Layer %d D size %zu < %d\n", i,
                        b->D.n_elements, d_inner);
                goto parse_error;
            }
            if (b->out_proj_norm.n_elements < (size_t) d_inner) {
                fprintf(stderr, "Error: Layer %d out_proj_norm size %zu < %d\n",
                        i, b->out_proj_norm.n_elements, d_inner);
                goto parse_error;
            }
        }
    }

    /* Validate tensor types: float-only tensors must not be bitnet-packed,
     * bitnet-only tensors must be packed. Prevents null dereference of .data
     * on tensors that a malformed model file incorrectly marks as bitnet. */
    if (m->embed.is_bitnet || m->norm_f.is_bitnet ||
        m->lm_head_norm.is_bitnet) {
        fprintf(stderr,
                "Error: embed/norm_f/lm_head_norm must be float, not bitnet\n");
        goto parse_error;
    }
    if (!m->lm_head.is_bitnet) {
        fprintf(stderr, "Error: lm_head must be bitnet-packed\n");
        goto parse_error;
    }
    for (int i = 0; i < m->config.n_layers; i++) {
        bitmamba_block_t *b = &m->layers[i];
        if (!b->in_proj.is_bitnet || !b->out_proj.is_bitnet) {
            fprintf(stderr,
                    "Error: Layer %d in_proj/out_proj must be bitnet-packed\n",
                    i);
            goto parse_error;
        }
        if (b->in_proj_norm.is_bitnet || b->out_proj_norm.is_bitnet ||
            b->conv1d_w.is_bitnet || b->conv1d_b.is_bitnet ||
            b->dt_bias.is_bitnet || b->A_log.is_bitnet || b->D.is_bitnet) {
            fprintf(stderr,
                    "Error: Layer %d float tensors must not be bitnet-packed\n",
                    i);
            goto parse_error;
        }
    }

    /* Precompute per-head A constants: -exp(A_log[h]) */
    for (int i = 0; i < m->config.n_layers; i++) {
        bitmamba_block_t *b = &m->layers[i];
        b->A_precomp = xmalloc(b->n_heads * sizeof(float));
        for (int h = 0; h < b->n_heads; h++)
            b->A_precomp[h] = -expf(b->A_log.data[h]);
    }

    /* Transpose conv1d weights from [channel][tap] to [tap][channel].
     * Enables contiguous vector loads across channels for each tap. */
    for (int i = 0; i < m->config.n_layers; i++) {
        bitmamba_block_t *b = &m->layers[i];
        int dc = b->d_conv;
        int di = b->d_inner;
        b->conv1d_w_t =
            xmalloc_aligned((size_t) dc * di * sizeof(float), BM_BUF_ALIGN);
        for (int tap = 0; tap < dc; tap++)
            for (int ch = 0; ch < di; ch++)
                b->conv1d_w_t[tap * di + ch] = b->conv1d_w.data[ch * dc + tap];
    }

    /* Allocate working buffers (64-byte aligned for SIMD / cache lines) */
    m->current_x = xmalloc_aligned(d_model * sizeof(float), BM_BUF_ALIGN);
    m->next_x = xmalloc_aligned(d_model * sizeof(float), BM_BUF_ALIGN);

    /* Find max in_proj.rows across all layers for proj_out buffer */
    int max_proj = 0;
    for (int i = 0; i < m->config.n_layers; i++) {
        if (m->layers[i].in_proj.rows > max_proj)
            max_proj = m->layers[i].in_proj.rows;
    }
    m->proj_out = xmalloc_aligned(max_proj * sizeof(float), BM_BUF_ALIGN);
    m->y_buffer = xmalloc_aligned(d_inner * sizeof(float), BM_BUF_ALIGN);
    m->final_feat = xmalloc_aligned(d_model * sizeof(float), BM_BUF_ALIGN);
    m->logits =
        xmalloc_aligned(m->config.vocab_size * sizeof(float), BM_BUF_ALIGN);
    m->sampler_probs = xmalloc(m->config.vocab_size * sizeof(token_prob_t));

    /* Repetition penalty dedup bitset (reused across forward steps) */
    {
        size_t bitset_words = ((size_t) m->config.vocab_size + 63) / 64;
        m->penalty_visited = xcalloc(bitset_words, sizeof(uint64_t));
    }

    /* Shared bitlinear buffers: max input dim = d_inner */
    m->bl_max_dim = d_inner;
    m->bl_x_norm = xmalloc_aligned(d_inner * sizeof(float), BM_BUF_ALIGN);
    m->bl_x_quant = xcalloc_aligned(d_inner + 32, sizeof(int8_t), BM_BUF_ALIGN);

    /* Persistent T-MAC LUT for scalar kernel (256 groups * 16 entries) */
    m->scalar_lut_buf =
        xmalloc_aligned(256 * 16 * sizeof(int32_t), BM_BUF_ALIGN);
    scalar_set_lut_buf(m->scalar_lut_buf);

#ifdef HAS_METAL
    if (m->use_gpu) {
        bool metal_ok = bm_metal_init();
        if (metal_ok)
            metal_ok = bm_metal_set_mmap(m->mmap_addr, m->mmap_size);
        if (!metal_ok) {
            fprintf(stderr, "[WARN] Metal init failed, falling back to CPU\n");
            bm_metal_cleanup();
            m->use_gpu = false;
        }
    }
#endif

    /* NUMA weight replication: copy packed weight data to each remote node.
     * The mmap'd weights land on node 0 (first-touch by main thread).
     * Remote-node threads reading cross-socket halve effective bandwidth.
     * Replicating weights lets all nodes read from local memory.
     */
#ifdef __linux__
    if (bm_get_numa_nodes() > 1) {
        int n_nodes = bm_get_numa_nodes();
        size_t total_replicated = 0;

        /* Replicate per-layer in_proj and out_proj packed weights */
        for (int i = 0; i < m->config.n_layers; i++) {
            bitmamba_block_t *b = &m->layers[i];
            struct {
                const uint8_t *src;
                uint8_t **dst;
                int rows, cols;
            } tensors[] = {
                {b->in_proj.packed_data, b->in_proj_packed_numa,
                 b->in_proj.rows, b->in_proj.cols},
                {b->out_proj.packed_data, b->out_proj_packed_numa,
                 b->out_proj.rows, b->out_proj.cols},
            };
            for (int ti = 0; ti < 2; ti++) {
                if (!tensors[ti].src)
                    continue;
                int stride = (tensors[ti].cols + 3) / 4;
                size_t sz = (size_t) tensors[ti].rows * stride;
                for (int node = 1; node < n_nodes && node < 8; node++) {
                    if (bm_get_node_threads(node) == 0)
                        continue;
                    void *r = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                    if (r == MAP_FAILED)
                        continue;
                    unsigned long nodemask = 1UL << node;
                    syscall(__NR_mbind, r, sz, MPOL_BIND, &nodemask,
                            (unsigned long) (node + 2), 0);
                    memcpy(r, tensors[ti].src, sz);
                    tensors[ti].dst[node] = (uint8_t *) r;
                    total_replicated += sz;
                }
            }
        }

        /* Replicate LM head packed weights */
        if (m->lm_head.packed_data) {
            int packed_stride = (m->lm_head.cols + 3) / 4;
            size_t packed_size = (size_t) m->lm_head.rows * packed_stride;
            for (int node = 1; node < n_nodes && node < 8; node++) {
                if (bm_get_node_threads(node) == 0)
                    continue;
                void *replica = mmap(NULL, packed_size, PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                if (replica == MAP_FAILED)
                    continue;
                unsigned long nodemask = 1UL << node;
                syscall(__NR_mbind, replica, packed_size, MPOL_BIND, &nodemask,
                        (unsigned long) (node + 2), 0);
                memcpy(replica, m->lm_head.packed_data, packed_size);
                m->lm_head_packed_numa[node] = (uint8_t *) replica;
                total_replicated += packed_size;
            }
        }

        if (total_replicated > 0)
            fprintf(stderr,
                    "[INFO] NUMA: replicated weights on remote nodes "
                    "(%.1f MB total)\n",
                    (double) total_replicated / (1024.0 * 1024.0));
    }
#endif

    return true;

parse_error:
    fprintf(stderr, "Error: Corrupted model file (parse failed)\n");
    model_free(m);
    return false;
}

#ifdef HAS_METAL
static bool model_prefill_metal(bitmamba_model_t *m,
                                const int32_t *tokens,
                                int n_tokens)
{
    int d_model = m->config.d_model;
    int d_inner = d_model * BM_EXPAND_FACTOR;
    int max_proj = 0;

    for (int i = 0; i < m->config.n_layers; i++) {
        if (m->layers[i].in_proj.rows > max_proj)
            max_proj = m->layers[i].in_proj.rows;
    }

    int quant_stride = d_inner + 32;
    bm_metal_batch_ctx_t *ctx = bm_metal_batch_begin(
        n_tokens, d_model, max_proj, d_inner, quant_stride);
    if (!ctx)
        return false;

    float *x_batch = bm_metal_batch_ptr(ctx, BM_BUF_X);
    if (!x_batch) {
        bm_metal_batch_end(ctx);
        return false;
    }

    /* Embed tokens directly into GPU shared buffer */
    for (int t = 0; t < n_tokens; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= m->config.vocab_size)
            tok = 0;
        memcpy(x_batch + t * d_model, &m->embed.data[tok * d_model],
               d_model * sizeof(float));
    }

    /* Pipelined Execution:
       To hide CPU SSM latency, we batch:
       [Layer i OutProj] + [Layer i Residual] + [Layer i+1 InProj]

       Flow:
       1. Layer 0 InProj -> Sync
       2. Loop i=0..N-2:
          - CPU: SSM (Layer i)
          - GPU: OutProj (Layer i) -> Residual (Layer i) -> InProj (Layer i+1)
          - Sync
       3. Epilogue (Layer N-1):
          - CPU: SSM (Layer N-1)
          - GPU: OutProj (Layer N-1) -> Residual (Layer N-1)
          - Sync
    */

    /* Prologue: Layer 0 InProj */
    bitmamba_block_t *b0 = &m->layers[0];
    if (!bm_metal_encode_bitlinear(ctx, BM_BUF_X, d_model,
                                   b0->in_proj.packed_data, b0->in_proj.rows,
                                   b0->in_proj.cols, b0->in_proj.scale,
                                   b0->in_proj_norm.data, BM_BUF_PROJ)) {
        bm_metal_batch_end(ctx);
        return false;
    }
    bm_metal_submit_and_wait(ctx);

    float *proj_batch = bm_metal_batch_ptr(ctx, BM_BUF_PROJ);
    float *y_batch = bm_metal_batch_ptr(ctx, BM_BUF_Y);

    /* Main Loop */
    for (int i = 0; i < m->config.n_layers; i++) {
        bitmamba_block_t *b = &m->layers[i];
        int proj_dim = b->in_proj.rows;

        /* CPU: SSM Scan (Sequential) */
        for (int t = 0; t < n_tokens; t++)
            block_ssm_scan(b, proj_batch + t * proj_dim, y_batch + t * d_inner);

        /* GPU: OutProj (Layer i) */
        if (!bm_metal_encode_bitlinear(
                ctx, BM_BUF_Y, d_inner, b->out_proj.packed_data,
                b->out_proj.rows, b->out_proj.cols, b->out_proj.scale,
                b->out_proj_norm.data, BM_BUF_OUT)) {
            bm_metal_batch_end(ctx);
            return false;
        }

        /* GPU: Residual (Layer i) */
        bm_metal_encode_residual(ctx, n_tokens * d_model);

        /* GPU: InProj (Layer i+1) - Pipelined! */
        if (i < m->config.n_layers - 1) {
            bitmamba_block_t *next_b = &m->layers[i + 1];
            if (!bm_metal_encode_bitlinear(
                    ctx, BM_BUF_X, d_model, next_b->in_proj.packed_data,
                    next_b->in_proj.rows, next_b->in_proj.cols,
                    next_b->in_proj.scale, next_b->in_proj_norm.data,
                    BM_BUF_PROJ)) {
                bm_metal_batch_end(ctx);
                return false;
            }
        }

        bm_metal_submit_and_wait(ctx);
    }

    if (m->profile_enabled) {
        m->profile.prefill_n_tokens = n_tokens;
    }

    bm_metal_batch_end(ctx);
    return true;
}
#endif

void model_prefill(bitmamba_model_t *m, const int32_t *tokens, int n_tokens)
{
    if (n_tokens <= 0)
        return;

#ifdef HAS_METAL
    if (m->use_gpu && bm_metal_available() && n_tokens >= 2) {
        if (model_prefill_metal(m, tokens, n_tokens))
            return;
    }
#endif

    int d_model = m->config.d_model;
    int d_inner = d_model * BM_EXPAND_FACTOR;

    /* Determine max projection dimension for scratch buffer */
    int max_proj = 0;
    for (int i = 0; i < m->config.n_layers; i++) {
        if (m->layers[i].in_proj.rows > max_proj)
            max_proj = m->layers[i].in_proj.rows;
    }

    /* Quantization stride: input dim padded to +32 for NEON overread safety */
    int quant_stride = d_inner + 32;

    /* Grow-on-demand scratch buffers (reused across prefill calls) */
    if (n_tokens > m->pf_capacity) {
        free(m->pf_x_batch);
        free(m->pf_out_batch);
        free(m->pf_proj_batch);
        free(m->pf_y_batch);
        free(m->pf_quant_batch);
        m->pf_x_batch = xmalloc_aligned(
            (size_t) n_tokens * d_model * sizeof(float), BM_BUF_ALIGN);
        m->pf_out_batch = xmalloc_aligned(
            (size_t) n_tokens * d_model * sizeof(float), BM_BUF_ALIGN);
        m->pf_proj_batch = xmalloc_aligned(
            (size_t) n_tokens * max_proj * sizeof(float), BM_BUF_ALIGN);
        m->pf_y_batch = xmalloc_aligned(
            (size_t) n_tokens * d_inner * sizeof(float), BM_BUF_ALIGN);
        m->pf_quant_batch = xcalloc_aligned((size_t) n_tokens * quant_stride,
                                            sizeof(int8_t), BM_BUF_ALIGN);
        m->pf_capacity = n_tokens;
    }

    float *x_batch = m->pf_x_batch;
    float *out_batch = m->pf_out_batch;
    float *proj_batch = m->pf_proj_batch;
    float *y_batch = m->pf_y_batch;
    int8_t *quant_batch = m->pf_quant_batch;

    double t0 = 0, t1 = 0;
    if (m->profile_enabled)
        t0 = get_time_ms();

    /* Embed all tokens into x_batch */
    for (int t = 0; t < n_tokens; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= m->config.vocab_size) {
            fprintf(stderr, "Error: Invalid token ID %d in prefill\n", tok);
            tok = 0;
        }
        memcpy(x_batch + t * d_model, &m->embed.data[tok * d_model],
               d_model * sizeof(float));
    }

    if (m->profile_enabled) {
        t1 = get_time_ms();
        m->profile.prefill_embed_ms = t1 - t0;
    }

    /* Forward through layers with residual */
    for (int i = 0; i < m->config.n_layers; i++) {
        if (m->profile_enabled)
            t0 = get_time_ms();

        block_step_batch(&m->layers[i], x_batch, n_tokens, out_batch,
                         proj_batch, y_batch, m->bl_x_norm, quant_batch,
                         quant_stride, m->use_gpu);

        /* Residual connection: x_batch += out_batch */
        int total = n_tokens * d_model;
        for (int j = 0; j < total; j++)
            x_batch[j] += out_batch[j];

        if (m->profile_enabled && i < MAX_PROFILE_LAYERS) {
            t1 = get_time_ms();
            m->profile.prefill_layer_ms[i] = t1 - t0;
        }
    }

    /* No final norm, no LM head, no sampling -- just state updates */
    if (m->profile_enabled)
        m->profile.prefill_n_tokens = n_tokens;
}

/*
 * NUMA-aware row worker: wraps the kernel row worker but selects
 * the node-local weight replica for each thread's NUMA node.
 * Only used for the LM head on multi-NUMA systems.
 */
typedef struct {
    bitlinear_row_work_t base;
    uint8_t *const *packed_per_node; /* [BM_MAX_NUMA] array */
} numa_row_work_t;

static void numa_row_worker(int tid, int n_threads, void *arg)
{
    numa_row_work_t *nw = (numa_row_work_t *) arg;
    bitlinear_row_work_t local = nw->base;
    int node = bm_tid_node(tid);
    if (nw->packed_per_node[node])
        local.packed_data = nw->packed_per_node[node];
    g_kernels.bitlinear_row_worker(tid, n_threads, &local);
}

int model_forward_step(bitmamba_model_t *m,
                       int token,
                       const int *history,
                       int history_len,
                       float penalty,
                       float temp,
                       float min_p,
                       float top_p,
                       int top_k)
{
    int d_model = m->config.d_model;
    double t0 = 0, t1 = 0;

    /* Validate token ID is within vocab bounds */
    if (token < 0 || token >= m->config.vocab_size) {
        fprintf(stderr, "Error: Invalid token ID %d (vocab_size=%d)\n", token,
                m->config.vocab_size);
        return 0;
    }

    /* Embedding lookup */
    if (m->profile_enabled)
        t0 = get_time_ms();
    memcpy(m->current_x, &m->embed.data[token * d_model],
           d_model * sizeof(float));
    if (m->profile_enabled) {
        t1 = get_time_ms();
        m->profile.embed_ms += t1 - t0;
    }

    /* Forward through layers with residual */
    for (int i = 0; i < m->config.n_layers; i++) {
        if (m->profile_enabled)
            t0 = get_time_ms();
        block_step(&m->layers[i], m->current_x, m->next_x, m->proj_out,
                   m->y_buffer, m->bl_x_norm, m->bl_x_quant);
        for (int j = 0; j < d_model; j++)
            m->current_x[j] += m->next_x[j];
        if (m->profile_enabled && i < MAX_PROFILE_LAYERS) {
            t1 = get_time_ms();
            m->profile.layer_ms[i] += t1 - t0;
        }
    }

    /* Final RMS norm */
    if (m->profile_enabled)
        t0 = get_time_ms();
    g_kernels.rms_norm(m->current_x, d_model, m->norm_f.data, m->final_feat);
    if (m->profile_enabled) {
        t1 = get_time_ms();
        m->profile.final_norm_ms += t1 - t0;
    }

    /* LM head projection.
     * NUMA-aware path: decompose into prepare + row dispatch with per-node
     * weight replicas so remote-socket threads read from local memory.
     * Falls back to standard bitlinear_forward when replicas don't exist
     * or when running the scalar kernel (bitlinear_prepare == NULL).
     */
    if (m->profile_enabled)
        t0 = get_time_ms();
    if (g_kernels.bitlinear_prepare && m->lm_head_packed_numa[1]) {
        float scale_x = g_kernels.bitlinear_prepare(
            m->final_feat, d_model, &m->lm_head_norm, m->bl_x_norm,
            m->bl_x_quant);
        int cols = m->lm_head.cols;
        numa_row_work_t nw = {
            .base =
                {
                    .x_quant = m->bl_x_quant,
                    .packed_data = m->lm_head.packed_data,
                    .out = m->logits,
                    .inv_scale = 1.0f / (scale_x * m->lm_head.scale),
                    .rows = m->lm_head.rows,
                    .cols = cols,
                    .packed_stride = (cols + 3) / 4,
                },
            .packed_per_node = m->lm_head_packed_numa,
        };
        bm_parallel_for(m->lm_head.rows, numa_row_worker, &nw);
    } else {
        g_kernels.bitlinear_forward(m->final_feat, d_model, &m->lm_head,
                                    &m->lm_head_norm, m->logits, m->bl_x_norm,
                                    m->bl_x_quant);
    }
    if (m->profile_enabled) {
        t1 = get_time_ms();
        m->profile.lm_head_ms += t1 - t0;
    }

    /* Apply repetition penalty (timed as part of sampling stage) */
    if (m->profile_enabled)
        t0 = get_time_ms();

    static const int protected_tokens[] = {TOKEN_EOS, TOKEN_NL, TOKEN_NLNL,
                                           TOKEN_PERIOD};
    static const int n_protected =
        sizeof(protected_tokens) / sizeof(protected_tokens[0]);

    /* Visited bitset: apply penalty once per unique token, not per occurrence.
     * Uses pre-allocated m->penalty_visited, cleared per call. */
    int vocab = m->config.vocab_size;
    size_t bitset_words = ((size_t) vocab + 63) / 64;
    memset(m->penalty_visited, 0, bitset_words * sizeof(uint64_t));

    for (int i = 0; i < history_len; i++) {
        int past = history[i];
        if (past < 0 || past >= vocab)
            continue;
        /* Skip if already penalized */
        if (m->penalty_visited[past / 64] & (1ULL << (past % 64)))
            continue;
        m->penalty_visited[past / 64] |= (1ULL << (past % 64));

        bool is_prot = false;
        for (int p = 0; p < n_protected; p++) {
            if (protected_tokens[p] == past) {
                is_prot = true;
                break;
            }
        }
        if (!is_prot && penalty > 0.0f) {
            if (m->logits[past] > 0)
                m->logits[past] /= penalty;
            else
                m->logits[past] *= penalty;
        }
    }
    int result = sample_advanced(m->logits, m->config.vocab_size, temp, min_p,
                                 top_p, top_k, m->sampler_probs);
    if (m->profile_enabled) {
        t1 = get_time_ms();
        m->profile.sample_ms += t1 - t0;
        m->profile.n_tokens++;
    }
    return result;
}

void model_free(bitmamba_model_t *m)
{
#ifdef HAS_METAL
    if (bm_metal_available())
        bm_metal_cleanup();
#endif

    /* Free NUMA weight replicas (mmap'd, not from model file) */
    for (int i = 0; i < 8; i++) {
        if (m->lm_head_packed_numa[i]) {
            int packed_stride = (m->lm_head.cols + 3) / 4;
            size_t packed_size = (size_t) m->lm_head.rows * packed_stride;
            munmap(m->lm_head_packed_numa[i], packed_size);
            m->lm_head_packed_numa[i] = NULL;
        }
    }

    /* Tensor data points into mmap'd region - just null them */
    m->embed.data = NULL;
    m->norm_f.data = NULL;
    m->lm_head_norm.data = NULL;
    m->lm_head.packed_data = NULL;

    if (m->layers) {
        for (int i = 0; i < m->config.n_layers; i++)
            block_free(&m->layers[i]);
        free(m->layers);
    }

    /* Unmap the model file */
    if (m->mmap_addr && m->mmap_addr != MAP_FAILED)
        munmap(m->mmap_addr, m->mmap_size);
    m->mmap_addr = NULL;

    /* Free working buffers (heap-allocated) */
    free(m->current_x);
    free(m->next_x);
    free(m->proj_out);
    free(m->y_buffer);
    free(m->final_feat);
    free(m->logits);
    free(m->sampler_probs);
    free(m->penalty_visited);
    free(m->bl_x_norm);
    free(m->bl_x_quant);
    free(m->scalar_lut_buf);
    scalar_set_lut_buf(NULL); /* clear global to prevent use-after-free */

    /* Free prefill scratch buffers */
    free(m->pf_x_batch);
    free(m->pf_out_batch);
    free(m->pf_proj_batch);
    free(m->pf_y_batch);
    free(m->pf_quant_batch);
}
