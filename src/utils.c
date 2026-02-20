#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "bitmamba.h"

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p && size > 0) {
        fprintf(stderr, "Error: Out of memory (requested %zu bytes)\n", size);
        exit(1);
    }
    return p;
}

void *xcalloc(size_t nmemb, size_t size)
{
    void *p = calloc(nmemb, size);
    if (!p && nmemb > 0 && size > 0) {
        fprintf(stderr, "Error: Out of memory (requested %zu bytes)\n",
                nmemb * size);
        exit(1);
    }
    return p;
}

void *xmalloc_aligned(size_t size, size_t alignment)
{
    void *p = NULL;
    if (size == 0)
        return NULL;
    if (posix_memalign(&p, alignment, size) != 0 || !p) {
        fprintf(
            stderr,
            "Error: Aligned alloc failed (requested %zu bytes, align %zu)\n",
            size, alignment);
        exit(1);
    }
    return p;
}

void *xcalloc_aligned(size_t nmemb, size_t size, size_t alignment)
{
    if (size && nmemb > SIZE_MAX / size) {
        fprintf(stderr, "Error: Allocation overflow (%zu * %zu bytes)\n", nmemb,
                size);
        exit(1);
    }
    size_t total = nmemb * size;
    void *p = xmalloc_aligned(total, alignment);
    memset(p, 0, total);
    return p;
}

double get_memory_usage_mb(void)
{
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info,
                  &count) == KERN_SUCCESS) {
        return info.resident_size / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp)
        return 0.0;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            long kb = 0;
            sscanf(line + 6, "%ld", &kb);
            fclose(fp);
            return kb / 1024.0;
        }
    }
    fclose(fp);
    return 0.0;
#endif
}

double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

double stats_tokens_per_second(const inference_stats_t *s)
{
    if (s->total_time_ms <= 0)
        return 0;
    return (s->total_tokens * 1000.0) / s->total_time_ms;
}

void stats_print_summary(const inference_stats_t *s)
{
    fprintf(stderr, "\n=== INFERENCE STATISTICS ===\n");
    fprintf(stderr, "Generated tokens: %d\n", s->total_tokens);
    fprintf(stderr, "Total time: %.2f ms\n", s->total_time_ms);
    fprintf(stderr, "Speed: %.2f tokens/sec\n", stats_tokens_per_second(s));
    fprintf(stderr, "Initial RAM: %.2f MB\n", s->initial_memory_mb);
    fprintf(stderr, "Peak RAM: %.2f MB\n", s->peak_memory_mb);
    fprintf(stderr, "RAM used (inference): %.2f MB\n",
            s->peak_memory_mb - s->initial_memory_mb);
    fprintf(stderr, "===================================\n");
}

/* --- XorShift64 PRNG ---
 * Fast, high-quality PRNG with 2^64-1 period. Better than rand(). */
static uint64_t g_rng_state = 0x853c49e6748fea9bULL;

void rng_seed(uint64_t seed)
{
    g_rng_state = seed ? seed : 0x853c49e6748fea9bULL;
}

uint64_t rng_next(void)
{
    uint64_t x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Returns uniform float in [0, 1) */
float rng_float(void)
{
    return (float) (rng_next() >> 11) * (1.0f / 9007199254740992.0f);
}

/* Per-Stage Profiling */

void profile_reset(bitmamba_model_t *m)
{
    int n_layers = m->config.n_layers;
    memset(&m->profile, 0, sizeof(m->profile));
    m->profile.n_layers =
        n_layers < MAX_PROFILE_LAYERS ? n_layers : MAX_PROFILE_LAYERS;
}

void profile_print(const bitmamba_model_t *m)
{
    if (!m->profile_enabled)
        return;

    const token_profile_t *p = &m->profile;

    /* Print prefill profile if we did batched prefill */
    if (p->prefill_n_tokens > 0) {
        double pf_layer_total = 0;
        for (int i = 0; i < p->n_layers; i++)
            pf_layer_total += p->prefill_layer_ms[i];
        double pf_total = p->prefill_embed_ms + pf_layer_total;

        fprintf(stderr, "\n=== PREFILL PROFILE (%d tokens) ===\n",
                p->prefill_n_tokens);
        fprintf(stderr, "Stage           Time (ms)    %%      ms/tok\n");
        fprintf(stderr, "-----------------------------------------------\n");
        fprintf(stderr, "Embedding       %9.2f  %5.1f%%  %7.3f\n",
                p->prefill_embed_ms,
                pf_total > 0 ? 100.0 * p->prefill_embed_ms / pf_total : 0,
                p->prefill_embed_ms / p->prefill_n_tokens);
        fprintf(stderr, "Layers (total)  %9.2f  %5.1f%%  %7.3f\n",
                pf_layer_total,
                pf_total > 0 ? 100.0 * pf_layer_total / pf_total : 0,
                pf_layer_total / p->prefill_n_tokens);
        fprintf(stderr, "-----------------------------------------------\n");
        fprintf(stderr, "TOTAL           %9.2f  100.0%%  %7.3f\n", pf_total,
                pf_total / p->prefill_n_tokens);
        fprintf(stderr, "===========================================\n");
    }

    if (p->n_tokens == 0)
        return;

    double layer_total = 0;
    for (int i = 0; i < p->n_layers; i++)
        layer_total += p->layer_ms[i];

    double total = p->embed_ms + layer_total + p->final_norm_ms +
                   p->lm_head_ms + p->sample_ms;

    fprintf(stderr, "\n=== DECODE PROFILE (%d tokens) ===\n", p->n_tokens);
    fprintf(stderr, "Stage           Time (ms)    %%      ms/tok\n");
    fprintf(stderr, "-----------------------------------------------\n");
    fprintf(stderr, "Embedding       %9.2f  %5.1f%%  %7.3f\n", p->embed_ms,
            100.0 * p->embed_ms / total, p->embed_ms / p->n_tokens);
    fprintf(stderr, "Layers (total)  %9.2f  %5.1f%%  %7.3f\n", layer_total,
            100.0 * layer_total / total, layer_total / p->n_tokens);

    /* Find hottest 3 layers */
    int hottest[3] = {-1, -1, -1};
    for (int i = 0; i < p->n_layers; i++) {
        if (hottest[0] < 0 || p->layer_ms[i] > p->layer_ms[hottest[0]]) {
            hottest[2] = hottest[1];
            hottest[1] = hottest[0];
            hottest[0] = i;
        } else if (hottest[1] < 0 || p->layer_ms[i] > p->layer_ms[hottest[1]]) {
            hottest[2] = hottest[1];
            hottest[1] = i;
        } else if (hottest[2] < 0 || p->layer_ms[i] > p->layer_ms[hottest[2]]) {
            hottest[2] = i;
        }
    }
    if (hottest[0] >= 0) {
        fprintf(stderr, "  Hottest: L%d (%.2fms)", hottest[0],
                p->layer_ms[hottest[0]] / p->n_tokens);
        if (hottest[1] >= 0)
            fprintf(stderr, ", L%d (%.2fms)", hottest[1],
                    p->layer_ms[hottest[1]] / p->n_tokens);
        if (hottest[2] >= 0)
            fprintf(stderr, ", L%d (%.2fms)", hottest[2],
                    p->layer_ms[hottest[2]] / p->n_tokens);
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "Final Norm      %9.2f  %5.1f%%  %7.3f\n", p->final_norm_ms,
            100.0 * p->final_norm_ms / total, p->final_norm_ms / p->n_tokens);
    fprintf(stderr, "LM Head         %9.2f  %5.1f%%  %7.3f\n", p->lm_head_ms,
            100.0 * p->lm_head_ms / total, p->lm_head_ms / p->n_tokens);
    fprintf(stderr, "Sampling        %9.2f  %5.1f%%  %7.3f\n", p->sample_ms,
            100.0 * p->sample_ms / total, p->sample_ms / p->n_tokens);
    fprintf(stderr, "-----------------------------------------------\n");
    fprintf(stderr, "TOTAL           %9.2f  100.0%%  %7.3f\n", total,
            total / p->n_tokens);
    fprintf(stderr, "===========================================\n");
}
