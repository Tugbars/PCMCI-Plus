/**
 * @file significance.c
 * @brief Multiple testing correction and significance utilities
 * 
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * Argsort for FDR
 *============================================================================*/

typedef struct {
    double val;
    int32_t idx;
} indexed_val_t;

static int cmp_indexed_asc(const void* a, const void* b) {
    const indexed_val_t* ia = (const indexed_val_t*)a;
    const indexed_val_t* ib = (const indexed_val_t*)b;
    
    if (ia->val < ib->val) return -1;
    if (ia->val > ib->val) return 1;
    return 0;
}

int32_t* pcmci_argsort(const double* arr, int32_t n) {
    indexed_val_t* indexed = (indexed_val_t*)malloc(n * sizeof(indexed_val_t));
    if (!indexed) return NULL;
    
    for (int32_t i = 0; i < n; i++) {
        indexed[i].val = arr[i];
        indexed[i].idx = i;
    }
    
    qsort(indexed, n, sizeof(indexed_val_t), cmp_indexed_asc);
    
    int32_t* result = (int32_t*)malloc(n * sizeof(int32_t));
    if (result) {
        for (int32_t i = 0; i < n; i++) {
            result[i] = indexed[i].idx;
        }
    }
    
    free(indexed);
    return result;
}

/*============================================================================
 * FDR Correction Methods
 *============================================================================*/

/**
 * Benjamini-Hochberg FDR correction
 * 
 * Algorithm:
 * 1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
 * 2. For each i, compute threshold: (i/m) * alpha
 * 3. Find largest k where p_(k) <= (k/m) * alpha
 * 4. Reject all hypotheses with p_(i) <= p_(k)
 * 
 * Adjusted p-value for p_(i) = min(1, m/i * p_(i)) with monotonicity enforced
 */
static void fdr_bh(const double* pvalues, int32_t n, double* adjusted) {
    if (n == 0) return;
    
    /* Get sort indices */
    int32_t* order = pcmci_argsort(pvalues, n);
    if (!order) {
        memcpy(adjusted, pvalues, n * sizeof(double));
        return;
    }
    
    /* Compute adjusted p-values */
    double* adj_sorted = (double*)malloc(n * sizeof(double));
    if (!adj_sorted) {
        free(order);
        memcpy(adjusted, pvalues, n * sizeof(double));
        return;
    }
    
    /* Start from largest p-value */
    adj_sorted[n - 1] = pvalues[order[n - 1]];
    
    for (int32_t i = n - 2; i >= 0; i--) {
        /* Adjusted = m / (i+1) * p_(i+1), clamped to [0, 1] */
        double adj = (double)n / (i + 1) * pvalues[order[i]];
        adj = fmin(adj, 1.0);
        
        /* Enforce monotonicity: adj[i] <= adj[i+1] */
        adj_sorted[i] = fmin(adj, adj_sorted[i + 1]);
    }
    
    /* Map back to original order */
    for (int32_t i = 0; i < n; i++) {
        adjusted[order[i]] = adj_sorted[i];
    }
    
    free(adj_sorted);
    free(order);
}

/**
 * Benjamini-Yekutieli FDR correction
 * More conservative than BH, valid under arbitrary dependence
 * 
 * Uses factor c(m) = sum_{i=1}^{m} 1/i (harmonic number)
 */
static void fdr_by(const double* pvalues, int32_t n, double* adjusted) {
    if (n == 0) return;
    
    /* Compute harmonic number c(m) = 1 + 1/2 + 1/3 + ... + 1/m */
    double c_m = 0.0;
    for (int32_t i = 1; i <= n; i++) {
        c_m += 1.0 / i;
    }
    
    /* Get sort indices */
    int32_t* order = pcmci_argsort(pvalues, n);
    if (!order) {
        memcpy(adjusted, pvalues, n * sizeof(double));
        return;
    }
    
    double* adj_sorted = (double*)malloc(n * sizeof(double));
    if (!adj_sorted) {
        free(order);
        memcpy(adjusted, pvalues, n * sizeof(double));
        return;
    }
    
    /* Start from largest p-value */
    adj_sorted[n - 1] = fmin(1.0, pvalues[order[n - 1]] * c_m);
    
    for (int32_t i = n - 2; i >= 0; i--) {
        /* Adjusted = c(m) * m / (i+1) * p_(i+1) */
        double adj = c_m * (double)n / (i + 1) * pvalues[order[i]];
        adj = fmin(adj, 1.0);
        
        /* Enforce monotonicity */
        adj_sorted[i] = fmin(adj, adj_sorted[i + 1]);
    }
    
    for (int32_t i = 0; i < n; i++) {
        adjusted[order[i]] = adj_sorted[i];
    }
    
    free(adj_sorted);
    free(order);
}

/**
 * Bonferroni correction (most conservative)
 */
static void fdr_bonferroni(const double* pvalues, int32_t n, double* adjusted) {
    for (int32_t i = 0; i < n; i++) {
        adjusted[i] = fmin(1.0, pvalues[i] * n);
    }
}

/*============================================================================
 * Public FDR API
 *============================================================================*/

void pcmci_fdr_correct(const double* pvalues, int32_t n,
                        pcmci_fdr_method_t method, double* adjusted) {
    if (!pvalues || !adjusted || n <= 0) return;
    
    switch (method) {
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

pcmci_link_t* pcmci_get_significant_links(const pcmci_graph_t* graph,
                                           double alpha,
                                           pcmci_fdr_method_t method,
                                           int32_t* out_count) {
    if (!graph || !out_count) {
        if (out_count) *out_count = 0;
        return NULL;
    }
    
    int32_t n = graph->n_vars;
    int32_t tau_max = graph->tau_max;
    
    /* Collect all p-values from existing links */
    int32_t n_links = 0;
    int64_t total = (int64_t)n * (tau_max + 1) * n;
    
    for (int64_t idx = 0; idx < total; idx++) {
        if (graph->adj[idx]) n_links++;
    }
    
    if (n_links == 0) {
        *out_count = 0;
        return NULL;
    }
    
    /* Extract p-values and link info */
    double* pvals = (double*)malloc(n_links * sizeof(double));
    int64_t* indices = (int64_t*)malloc(n_links * sizeof(int64_t));
    
    if (!pvals || !indices) {
        free(pvals);
        free(indices);
        *out_count = 0;
        return NULL;
    }
    
    int32_t link_idx = 0;
    for (int64_t idx = 0; idx < total; idx++) {
        if (graph->adj[idx]) {
            pvals[link_idx] = graph->pval_matrix[idx];
            indices[link_idx] = idx;
            link_idx++;
        }
    }
    
    /* Apply FDR correction */
    double* adjusted = (double*)malloc(n_links * sizeof(double));
    if (!adjusted) {
        free(pvals);
        free(indices);
        *out_count = 0;
        return NULL;
    }
    
    pcmci_fdr_correct(pvals, n_links, method, adjusted);
    
    /* Count significant links */
    int32_t n_sig = 0;
    for (int32_t i = 0; i < n_links; i++) {
        if (adjusted[i] <= alpha) n_sig++;
    }
    
    if (n_sig == 0) {
        free(pvals);
        free(indices);
        free(adjusted);
        *out_count = 0;
        return NULL;
    }
    
    /* Build result array */
    pcmci_link_t* result = (pcmci_link_t*)malloc(n_sig * sizeof(pcmci_link_t));
    if (!result) {
        free(pvals);
        free(indices);
        free(adjusted);
        *out_count = 0;
        return NULL;
    }
    
    int32_t result_idx = 0;
    for (int32_t i = 0; i < n_links; i++) {
        if (adjusted[i] <= alpha) {
            int64_t idx = indices[i];
            
            /* Decode flat index back to (i, tau, j) */
            int32_t j_var = idx % n;
            int32_t tau = (idx / n) % (tau_max + 1);
            int32_t i_var = idx / (n * (tau_max + 1));
            
            result[result_idx].i = i_var;
            result[result_idx].tau = tau;
            result[result_idx].j = j_var;
            result[result_idx].val = graph->val_matrix[idx];
            result[result_idx].pvalue = adjusted[i];  /* Use adjusted p-value */
            result_idx++;
        }
    }
    
    free(pvals);
    free(indices);
    free(adjusted);
    
    *out_count = n_sig;
    return result;
}
