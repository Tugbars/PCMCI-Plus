/**
 * @file benchmark_latency.c
 * @brief Latency benchmarks for PCMCI+ core operations
 *
 * Measures per-operation latency for:
 * - Partial correlation test (varying conditioning set sizes)
 * - Data extraction
 * - Residualization
 * - Full skeleton discovery
 * - Full MCI phase
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_internal.h"
#include "pcmci_tuning.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

/*============================================================================
 * Benchmark Utilities
 *============================================================================*/

typedef struct
{
    double min;
    double max;
    double mean;
    double std;
    double median;
    double p99;
    int64_t count;
} latency_stats_t;

static int cmp_double(const void *a, const void *b)
{
    double diff = *(double *)a - *(double *)b;
    if (diff < 0)
        return -1;
    if (diff > 0)
        return 1;
    return 0;
}

static latency_stats_t compute_stats(double *samples, int64_t n)
{
    latency_stats_t stats = {0};
    if (n == 0)
        return stats;

    stats.count = n;
    stats.min = DBL_MAX;
    stats.max = -DBL_MAX;

    double sum = 0.0;
    for (int64_t i = 0; i < n; i++)
    {
        sum += samples[i];
        if (samples[i] < stats.min)
            stats.min = samples[i];
        if (samples[i] > stats.max)
            stats.max = samples[i];
    }
    stats.mean = sum / n;

    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; i++)
    {
        double d = samples[i] - stats.mean;
        sum_sq += d * d;
    }
    stats.std = sqrt(sum_sq / n);

    /* Sort for percentiles */
    qsort(samples, n, sizeof(double), cmp_double);
    stats.median = samples[n / 2];
    stats.p99 = samples[(int64_t)(n * 0.99)];

    return stats;
}

static void print_stats(const char *name, latency_stats_t *stats, const char *unit)
{
    printf("  %-30s: mean=%8.3f %s, std=%7.3f, min=%7.3f, p50=%7.3f, p99=%7.3f, max=%8.3f (n=%ld)\n",
           name, stats->mean, unit, stats->std, stats->min,
           stats->median, stats->p99, stats->max, (long)stats->count);
}

/*============================================================================
 * Data Generation
 *============================================================================*/

static double *generate_test_data(int32_t n_vars, int32_t T, uint64_t seed)
{
    double *data = (double *)pcmci_malloc((size_t)n_vars * T * sizeof(double));
    if (!data)
        return NULL;

    /* Simple LCG for reproducibility */
    uint64_t state = seed;
    for (int32_t i = 0; i < n_vars; i++)
    {
        for (int32_t t = 0; t < T; t++)
        {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            double u = (double)(state >> 11) / (double)(1ULL << 53);
            /* Box-Muller for Gaussian */
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            double v = (double)(state >> 11) / (double)(1ULL << 53);
            double z = sqrt(-2.0 * log(u + 1e-10)) * cos(2.0 * 3.14159265358979 * v);
            data[i * T + t] = z;
        }
    }

    /* Add autoregression */
    for (int32_t i = 0; i < n_vars; i++)
    {
        for (int32_t t = 1; t < T; t++)
        {
            data[i * T + t] += 0.5 * data[i * T + t - 1];
        }
    }

    return data;
}

/*============================================================================
 * Benchmark: Partial Correlation
 *============================================================================*/

static void benchmark_parcorr(int32_t n_samples, int32_t max_k, int32_t n_iters)
{
    printf("\n=== Partial Correlation Benchmark ===\n");
    printf("Samples: %d, Max conditioning: %d, Iterations per k: %d\n\n",
           n_samples, max_k, n_iters);

    /* Allocate test data */
    double *X = (double *)pcmci_malloc(n_samples * sizeof(double));
    double *Y = (double *)pcmci_malloc(n_samples * sizeof(double));
    double *Z = (double *)pcmci_malloc((size_t)n_samples * max_k * sizeof(double));
    double *samples = (double *)malloc(n_iters * sizeof(double));

    /* Fill with random data */
    uint64_t state = 12345;
    for (int32_t i = 0; i < n_samples; i++)
    {
        state = state * 6364136223846793005ULL + 1;
        X[i] = (double)(state >> 32) / (double)(1ULL << 32) - 0.5;
        state = state * 6364136223846793005ULL + 1;
        Y[i] = (double)(state >> 32) / (double)(1ULL << 32) - 0.5;
    }
    for (int32_t j = 0; j < max_k; j++)
    {
        for (int32_t i = 0; i < n_samples; i++)
        {
            state = state * 6364136223846793005ULL + 1;
            Z[j * n_samples + i] = (double)(state >> 32) / (double)(1ULL << 32) - 0.5;
        }
    }

    pcmci_workspace_t *ws = pcmci_workspace_create(n_samples, max_k);

    /* Warmup */
    for (int i = 0; i < 100; i++)
    {
        pcmci_parcorr_ws(X, Y, Z, n_samples, max_k / 2, ws);
    }

    /* Benchmark each conditioning set size */
    for (int32_t k = 0; k <= max_k; k++)
    {
        for (int32_t iter = 0; iter < n_iters; iter++)
        {
            double t0 = pcmci_get_time();

            pcmci_ci_result_t result = pcmci_parcorr_ws(X, Y,
                                                        k > 0 ? Z : NULL,
                                                        n_samples, k, ws);

            double t1 = pcmci_get_time();
            samples[iter] = (t1 - t0) * 1e6; /* Convert to microseconds */

            /* Prevent optimization */
            if (result.pvalue < -1.0)
                printf("x");
        }

        latency_stats_t stats = compute_stats(samples, n_iters);
        char name[64];
        snprintf(name, sizeof(name), "parcorr (k=%d)", k);
        print_stats(name, &stats, "µs");
    }

    pcmci_workspace_free(ws);
    pcmci_mkl_free(X);
    pcmci_mkl_free(Y);
    pcmci_mkl_free(Z);
    free(samples);
}

/*============================================================================
 * Benchmark: Residualization Only
 *============================================================================*/

static void benchmark_residualize(int32_t n_samples, int32_t max_k, int32_t n_iters)
{
    printf("\n=== Residualization Benchmark ===\n");
    printf("Samples: %d, Max conditioning: %d, Iterations per k: %d\n\n",
           n_samples, max_k, n_iters);

    double *X = (double *)pcmci_malloc(n_samples * sizeof(double));
    double *Z = (double *)pcmci_malloc((size_t)n_samples * max_k * sizeof(double));
    double *resid = (double *)pcmci_malloc(n_samples * sizeof(double));
    double *samples = (double *)malloc(n_iters * sizeof(double));

    uint64_t state = 54321;
    for (int32_t i = 0; i < n_samples; i++)
    {
        state = state * 6364136223846793005ULL + 1;
        X[i] = (double)(state >> 32) / (double)(1ULL << 32) - 0.5;
    }
    for (int32_t j = 0; j < max_k; j++)
    {
        for (int32_t i = 0; i < n_samples; i++)
        {
            state = state * 6364136223846793005ULL + 1;
            Z[j * n_samples + i] = (double)(state >> 32) / (double)(1ULL << 32) - 0.5;
        }
    }

    pcmci_workspace_t *ws = pcmci_workspace_create(n_samples, max_k);

    /* Warmup */
    for (int i = 0; i < 100; i++)
    {
        pcmci_residualize_ws(X, Z, n_samples, max_k / 2, ws, resid);
    }

    for (int32_t k = 1; k <= max_k; k++)
    {
        for (int32_t iter = 0; iter < n_iters; iter++)
        {
            double t0 = pcmci_get_time();

            pcmci_residualize_ws(X, Z, n_samples, k, ws, resid);

            double t1 = pcmci_get_time();
            samples[iter] = (t1 - t0) * 1e6;
        }

        latency_stats_t stats = compute_stats(samples, n_iters);
        char name[64];
        snprintf(name, sizeof(name), "residualize (k=%d)", k);
        print_stats(name, &stats, "µs");
    }

    pcmci_workspace_free(ws);
    pcmci_mkl_free(X);
    pcmci_mkl_free(Z);
    pcmci_mkl_free(resid);
    free(samples);
}

/*============================================================================
 * Benchmark: Data Extraction
 *============================================================================*/

static void benchmark_extraction(int32_t n_vars, int32_t T, int32_t n_iters)
{
    printf("\n=== Data Extraction Benchmark ===\n");
    printf("Variables: %d, Time points: %d, Iterations: %d\n\n", n_vars, T, n_iters);

    int32_t tau_max = 5;
    double *data = generate_test_data(n_vars, T, 99999);
    pcmci_dataframe_t *df = pcmci_dataframe_create(data, n_vars, T, tau_max);

    int32_t n_samples = T - tau_max;
    double *out = (double *)pcmci_malloc(n_samples * sizeof(double));
    double *samples = (double *)malloc(n_iters * sizeof(double));

    /* Benchmark single variable extraction */
    for (int32_t iter = 0; iter < n_iters; iter++)
    {
        double t0 = pcmci_get_time();

        for (int32_t v = 0; v < n_vars; v++)
        {
            for (int32_t tau = 0; tau <= tau_max; tau++)
            {
                pcmci_extract_lagged_into(df, v, tau, out);
            }
        }

        double t1 = pcmci_get_time();
        samples[iter] = (t1 - t0) * 1e6 / (n_vars * (tau_max + 1));
    }

    latency_stats_t stats = compute_stats(samples, n_iters);
    print_stats("extract_lagged_into (per call)", &stats, "µs");

    /* Benchmark conditioning set extraction */
    pcmci_varlag_t *cond_set = (pcmci_varlag_t *)malloc(10 * sizeof(pcmci_varlag_t));
    double *Z_out = (double *)pcmci_malloc((size_t)n_samples * 10 * sizeof(double));

    for (int32_t c = 0; c < 10; c++)
    {
        cond_set[c].var = c % n_vars;
        cond_set[c].tau = (c % tau_max) + 1;
    }

    for (int32_t n_cond = 1; n_cond <= 10; n_cond++)
    {
        for (int32_t iter = 0; iter < n_iters; iter++)
        {
            double t0 = pcmci_get_time();

            pcmci_extract_cond_set_into(df, cond_set, n_cond, Z_out);

            double t1 = pcmci_get_time();
            samples[iter] = (t1 - t0) * 1e6;
        }

        latency_stats_t s = compute_stats(samples, n_iters);
        char name[64];
        snprintf(name, sizeof(name), "extract_cond_set (n=%d)", n_cond);
        print_stats(name, &s, "µs");
    }

    free(cond_set);
    pcmci_mkl_free(Z_out);
    pcmci_mkl_free(out);
    free(samples);
    pcmci_dataframe_free(df);
}

/*============================================================================
 * Benchmark: Full Algorithm Phases
 *============================================================================*/

static void benchmark_full_algorithm(int32_t n_vars, int32_t T, int32_t tau_max,
                                     int32_t n_iters)
{
    printf("\n=== Full Algorithm Benchmark ===\n");
    printf("Variables: %d, Time points: %d, tau_max: %d, Iterations: %d\n\n",
           n_vars, T, tau_max, n_iters);

    double *skeleton_times = (double *)malloc(n_iters * sizeof(double));
    double *mci_times = (double *)malloc(n_iters * sizeof(double));
    double *total_times = (double *)malloc(n_iters * sizeof(double));

    pcmci_config_t config = pcmci_default_config();
    config.tau_max = tau_max;
    config.alpha_level = 0.05;
    config.verbosity = 0; /* Silent */
    config.n_threads = 1; /* Single-threaded for latency measurement */

    for (int32_t iter = 0; iter < n_iters; iter++)
    {
        /* Generate fresh data each iteration */
        double *data = generate_test_data(n_vars, T, 12345 + iter);
        pcmci_dataframe_t *df = pcmci_dataframe_create(data, n_vars, T, tau_max);

        /* Preprocess (winsorize + rank) */
        pcmci_dataframe_t *df_wins = pcmci_dataframe_winsorize(df, config.winsorize_thresh);
        pcmci_dataframe_t *df_ranked = pcmci_dataframe_to_ranks(df_wins ? df_wins : df);
        pcmci_dataframe_t *df_use = df_ranked ? df_ranked : df;

        pcmci_dataframe_t df_mod = *df_use;
        df_mod.tau_max = tau_max;

        double t0 = pcmci_get_time();

        /* Skeleton phase */
        double t_skel_start = pcmci_get_time();
        pcmci_graph_t *skeleton = pcmci_skeleton(&df_mod, &config);
        double t_skel_end = pcmci_get_time();

        /* MCI phase */
        double t_mci_start = pcmci_get_time();
        pcmci_mci(&df_mod, skeleton, &config);
        double t_mci_end = pcmci_get_time();

        double t1 = pcmci_get_time();

        skeleton_times[iter] = (t_skel_end - t_skel_start) * 1000.0; /* ms */
        mci_times[iter] = (t_mci_end - t_mci_start) * 1000.0;
        total_times[iter] = (t1 - t0) * 1000.0;

        pcmci_graph_free(skeleton);
        if (df_ranked)
            pcmci_dataframe_free(df_ranked);
        if (df_wins)
            pcmci_dataframe_free(df_wins);
        pcmci_dataframe_free(df);
    }

    latency_stats_t skel_stats = compute_stats(skeleton_times, n_iters);
    latency_stats_t mci_stats = compute_stats(mci_times, n_iters);
    latency_stats_t total_stats = compute_stats(total_times, n_iters);

    print_stats("Skeleton phase", &skel_stats, "ms");
    print_stats("MCI phase", &mci_stats, "ms");
    print_stats("Total (skel + mci)", &total_stats, "ms");

    free(skeleton_times);
    free(mci_times);
    free(total_times);
}

/*============================================================================
 * Benchmark: Scaling with Problem Size
 *============================================================================*/

static void benchmark_scaling(void)
{
    printf("\n=== Scaling Benchmark ===\n\n");

    int32_t n_iters = 10;
    pcmci_config_t config = pcmci_default_config();
    config.verbosity = 0;
    config.n_threads = 1;

    /* Vary number of variables */
    printf("Scaling with n_vars (T=500, tau_max=3):\n");
    int32_t var_sizes[] = {3, 5, 8, 10, 15, 20};
    int32_t n_var_sizes = sizeof(var_sizes) / sizeof(var_sizes[0]);

    for (int32_t vi = 0; vi < n_var_sizes; vi++)
    {
        int32_t n_vars = var_sizes[vi];
        int32_t T = 500;
        int32_t tau_max = 3;

        double total_time = 0.0;

        for (int32_t iter = 0; iter < n_iters; iter++)
        {
            double *data = generate_test_data(n_vars, T, 12345 + iter);
            pcmci_dataframe_t *df = pcmci_dataframe_create(data, n_vars, T, tau_max);

            config.tau_max = tau_max;

            double t0 = pcmci_get_time();
            pcmci_result_t *result = pcmci_run(df, &config);
            double t1 = pcmci_get_time();

            total_time += (t1 - t0);

            pcmci_result_free(result);
            pcmci_dataframe_free(df);
        }

        printf("  n_vars=%2d: %.3f ms/run\n", n_vars, total_time / n_iters * 1000.0);
    }

    /* Vary time series length */
    printf("\nScaling with T (n_vars=5, tau_max=3):\n");
    int32_t T_sizes[] = {200, 500, 1000, 2000, 5000};
    int32_t n_T_sizes = sizeof(T_sizes) / sizeof(T_sizes[0]);

    for (int32_t ti = 0; ti < n_T_sizes; ti++)
    {
        int32_t n_vars = 5;
        int32_t T = T_sizes[ti];
        int32_t tau_max = 3;

        double total_time = 0.0;

        for (int32_t iter = 0; iter < n_iters; iter++)
        {
            double *data = generate_test_data(n_vars, T, 12345 + iter);
            pcmci_dataframe_t *df = pcmci_dataframe_create(data, n_vars, T, tau_max);

            config.tau_max = tau_max;

            double t0 = pcmci_get_time();
            pcmci_result_t *result = pcmci_run(df, &config);
            double t1 = pcmci_get_time();

            total_time += (t1 - t0);

            pcmci_result_free(result);
            pcmci_dataframe_free(df);
        }

        printf("  T=%5d: %.3f ms/run\n", T, total_time / n_iters * 1000.0);
    }
}

/*============================================================================
 * Main
 *============================================================================*/

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    /* Initialize CPU/BLAS tuning (denormals, threads, MKL config) */
    pcmci_tuning_init(8, 1); /* 0 = auto threads, 1 = verbose */

    printf("PCMCI+ Latency Benchmark\n");
    printf("========================\n");
    printf("Version: %s\n", pcmci_version());
    printf("Threads: %d (max available)\n", omp_get_max_threads());

    /* Core operation benchmarks */
    benchmark_parcorr(500, 10, 1000);
    benchmark_residualize(500, 10, 1000);
    benchmark_extraction(10, 1000, 1000);

    /* Full algorithm benchmarks */
    benchmark_full_algorithm(5, 500, 3, 50);
    benchmark_full_algorithm(10, 1000, 5, 20);

    /* Scaling benchmarks */
    benchmark_scaling();

    printf("\n=== Benchmark Complete ===\n");

    pcmci_tuning_cleanup();
    return 0;
}