/**
 * @file significance.c
 * @brief Multiple testing correction and significance utilities
 *
 * Optimized: Radix sort for O(N) sorting and SIMD-friendly FDR.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/*============================================================================
 * Optimized Sorting (Radix Sort for Floats)
 *============================================================================*/

/* * Flip float bits to make them comparable as integers.
 * IEEE 754 floats preserve order when interpreted as integers if positive.
 */
static inline uint32_t float_to_uint32_key(double f)
{
    /* We assume p-values are in [0, 1], so they are positive doubles.
       We cast to float for 32-bit radix sort speed (sufficient precision for ranking). */
    float f32 = (float)f;
    uint32_t i;
    memcpy(&i, &f32, sizeof(uint32_t));
    return i;
}

/**
 * LSD Radix Sort for p-values.
 * Sorts 'idx' array based on values in 'arr'.
 * Complexity: O(N), much faster than qsort's O(N log N).
 */
void pcmci_radix_argsort(const double *arr, int32_t *idx, int32_t n)
{
    if (n <= 0)
        return;

    /* 4 passes of 8 bits each */
    int32_t *aux = (int32_t *)malloc(n * sizeof(int32_t));
    if (!aux)
        return; // Fallback or fail

    /* Initialize indices */
    for (int32_t i = 0; i < n; i++)
        idx[i] = i;

    /* Histogram buckets */
    int32_t count[256];
    int32_t *src = idx;
    int32_t *dst = aux;

    for (int shift = 0; shift < 32; shift += 8)
    {
        memset(count, 0, sizeof(count));

        /* Count occurrences */
        for (int32_t i = 0; i < n; i++)
        {
            uint32_t key = float_to_uint32_key(arr[src[i]]);
            count[(key >> shift) & 0xFF]++;
        }

        /* Compute prefixes */
        int32_t total = 0;
        for (int32_t i = 0; i < 256; i++)
        {
            int32_t c = count[i];
            count[i] = total;
            total += c;
        }

        /* Distribute */
        for (int32_t i = 0; i < n; i++)
        {
            uint32_t key = float_to_uint32_key(arr[src[i]]);
            dst[count[(key >> shift) & 0xFF]++] = src[i];
        }

        /* Swap pointers */
        int32_t *tmp = src;
        src = dst;
        dst = tmp;
    }

    /* If even number of passes, result is in src (idx if src==idx) */
    /* We did 4 passes. If src is aux, copy back. */
    if (src == aux)
    {
        memcpy(idx, aux, n * sizeof(int32_t));
    }

    free(aux);
}

/* Fallback qsort if needed (memory constrained) */
typedef struct
{
    double val;
    int32_t idx;
} indexed_val_t;

static int cmp_indexed_asc(const void *a, const void *b)
{
    const indexed_val_t *ia = (const indexed_val_t *)a;
    const indexed_val_t *ib = (const indexed_val_t *)b;
    return (ia->val > ib->val) - (ia->val < ib->val);
}

int32_t *pcmci_argsort(const double *arr, int32_t n)
{
    int32_t *result = (int32_t *)malloc(n * sizeof(int32_t));
    if (!result)
        return NULL;

    /* Use Radix sort for speed on large arrays */
    if (n > 128)
    {
        pcmci_radix_argsort(arr, result, n);
    }
    else
    {
        /* Small N: qsort overhead is lower than radix setup */
        indexed_val_t *indexed = (indexed_val_t *)malloc(n * sizeof(indexed_val_t));
        for (int32_t i = 0; i < n; i++)
        {
            indexed[i].val = arr[i];
            indexed[i].idx = i;
        }
        qsort(indexed, n, sizeof(indexed_val_t), cmp_indexed_asc);
        for (int32_t i = 0; i < n; i++)
            result[i] = indexed[i].idx;
        free(indexed);
    }
    return result;
}

/*============================================================================
 * FDR Correction Methods
 *============================================================================*/

/**
 * Benjamini-Hochberg FDR correction (Optimized)
 */
static void fdr_bh(const double *pvalues, int32_t n, double *adjusted)
{
    if (n == 0)
        return;

    int32_t *order = pcmci_argsort(pvalues, n);
    if (!order)
    {
        memcpy(adjusted, pvalues, n * sizeof(double));
        return;
    }

    /* * We need a temporary buffer for sorted adjustments.
     * We can reuse the output 'adjusted' array as the temporary sorted buffer
     * if we are careful, but mapping back requires random access.
     * Safer to alloc temp buffer.
     */
    double *adj_sorted = (double *)malloc(n * sizeof(double));
    if (!adj_sorted)
    {
        free(order);
        return;
    }

    /* Step 1: Initialize last element */
    adj_sorted[n - 1] = pvalues[order[n - 1]];

    /* Step 2: Backward pass (enforce monotonicity) */
    /* Loop is simple enough for compiler to unroll/vectorize parts */
    for (int32_t i = n - 2; i >= 0; i--)
    {
        double p = pvalues[order[i]];
        double factor = (double)n / (i + 1);
        double adj = p * factor;

        /* Optimization: Branchless min */
        if (adj > 1.0)
            adj = 1.0;

        double next_adj = adj_sorted[i + 1];
        adj_sorted[i] = (adj < next_adj) ? adj : next_adj;
    }

    /* Step 3: Scatter results back to original positions */
    /* This random write access is the cache bottleneck */
    for (int32_t i = 0; i < n; i++)
    {
        adjusted[order[i]] = adj_sorted[i];
    }

    free(adj_sorted);
    free(order);
}

/**
 * Benjamini-Yekutieli FDR correction
 */
static void fdr_by(const double *pvalues, int32_t n, double *adjusted)
{
    if (n == 0)
        return;

    /* Compute harmonic number c(m) */
    /* Optimization: Approx for large n: ln(n) + gamma + 1/(2n) */
    double c_m = 0.0;
    if (n > 1000)
    {
        c_m = log((double)n) + 0.5772156649 + 0.5 / (double)n;
    }
    else
    {
        for (int32_t i = 1; i <= n; i++)
            c_m += 1.0 / i;
    }

    int32_t *order = pcmci_argsort(pvalues, n);
    if (!order)
        return;

    double *adj_sorted = (double *)malloc(n * sizeof(double));

    /* Last element */
    double p_last = pvalues[order[n - 1]];
    double adj_last = p_last * c_m; /* factor is n/n * c_m */
    adj_sorted[n - 1] = (adj_last < 1.0) ? adj_last : 1.0;

    for (int32_t i = n - 2; i >= 0; i--)
    {
        double p = pvalues[order[i]];
        double factor = (double)n / (i + 1);
        double adj = p * factor * c_m;
        if (adj > 1.0)
            adj = 1.0;

        double next = adj_sorted[i + 1];
        adj_sorted[i] = (adj < next) ? adj : next;
    }

    for (int32_t i = 0; i < n; i++)
    {
        adjusted[order[i]] = adj_sorted[i];
    }

    free(adj_sorted);
    free(order);
}

static void fdr_bonferroni(const double *restrict pvalues, int32_t n, double *restrict adjusted)
{
    double n_dbl = (double)n;

#pragma omp simd
    for (int32_t i = 0; i < n; i++)
    {
        double adj = pvalues[i] * n_dbl;
        adjusted[i] = (adj < 1.0) ? adj : 1.0;
    }
}

/*============================================================================
 * Public FDR API
 *============================================================================*/

void pcmci_fdr_correct(const double *pvalues, int32_t n,
                       pcmci_fdr_method_t method, double *adjusted)
{
    if (!pvalues || !adjusted || n <= 0)
        return;

    switch (method)
    {
    case PCMCI_FDR_NONE:
        memcpy(adjusted, pvalues, n * sizeof(double));
        break;

    case PCMCI_FDR_BH:
        fdr_bh(pvalues, n, adjusted);
        break;

    case PCMCI_FDR_BY:
        fdr_by(pvalues, n, adjusted);
        break;

    case PCMCI_FDR_BONFERRONI:
        fdr_bonferroni(pvalues, n, adjusted);
        break;

    default:
        memcpy(adjusted, pvalues, n * sizeof(double));
        break;
    }
}

pcmci_link_t *pcmci_get_significant_links(const pcmci_graph_t *graph,
                                          double alpha,
                                          pcmci_fdr_method_t method,
                                          int32_t *out_count)
{
    if (!graph || !out_count)
    {
        if (out_count)
            *out_count = 0;
        return NULL;
    }

    int32_t n = graph->n_vars;
    int32_t tau_max = graph->tau_max;
    int64_t total = (int64_t)n * (tau_max + 1) * n;

    /* Pass 1: Count links to allocate exactly needed memory */
    int32_t n_links = 0;
    /* Use SIMD if possible? Bool array is tricky. Just scalar count. */
    for (int64_t idx = 0; idx < total; idx++)
    {
        if (graph->adj[idx])
            n_links++;
    }

    if (n_links == 0)
    {
        *out_count = 0;
        return NULL;
    }

    /* * Block Allocation Optimization:
     * Allocate pvals, indices, and adjusted in one contiguous block
     * to reduce malloc overhead and fragmentation.
     */
    size_t sz_dbl = n_links * sizeof(double);
    size_t sz_i64 = n_links * sizeof(int64_t);

    uint8_t *block = (uint8_t *)malloc(sz_dbl * 2 + sz_i64);
    if (!block)
    {
        *out_count = 0;
        return NULL;
    }

    double *pvals = (double *)(block);
    double *adjusted = (double *)(block + sz_dbl);
    int64_t *indices = (int64_t *)(block + sz_dbl * 2);

    /* Extraction */
    int32_t k = 0;
    for (int64_t idx = 0; idx < total; idx++)
    {
        if (graph->adj[idx])
        {
            pvals[k] = graph->pval_matrix[idx];
            indices[k] = idx;
            k++;
        }
    }

    /* Correction */
    pcmci_fdr_correct(pvals, n_links, method, adjusted);

    /* Pass 2: Count significant */
    int32_t n_sig = 0;
    for (int32_t i = 0; i < n_links; i++)
    {
        if (adjusted[i] <= alpha)
            n_sig++;
    }

    if (n_sig == 0)
    {
        free(block);
        *out_count = 0;
        return NULL;
    }

    /* Build Result */
    pcmci_link_t *result = (pcmci_link_t *)malloc(n_sig * sizeof(pcmci_link_t));
    if (result)
    {
        int32_t r_idx = 0;
        /* Constants for division */
        int32_t row_stride = n * (tau_max + 1);

        for (int32_t i = 0; i < n_links; i++)
        {
            if (adjusted[i] <= alpha)
            {
                int64_t idx = indices[i];

                /* Optimized decoding: avoid division where possible */
                /* idx = i * (tau_max+1)*n + tau * n + j */
                int32_t j_var = idx % n;
                int64_t rem = idx / n;
                int32_t tau = rem % (tau_max + 1);
                int32_t i_var = rem / (tau_max + 1);

                result[r_idx].i = i_var;
                result[r_idx].tau = tau;
                result[r_idx].j = j_var;
                result[r_idx].val = graph->val_matrix[idx];
                result[r_idx].pvalue = adjusted[i];
                r_idx++;
            }
        }
    }

    free(block);
    *out_count = n_sig;
    return result;
}