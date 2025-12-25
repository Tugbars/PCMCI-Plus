/**
 * @file mci.c
 * @brief Momentary Conditional Independence (MCI) phase for PCMCI+
 * 
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

/*============================================================================
 * MCI Phase
 *============================================================================*/

/**
 * Get parents of a variable from the skeleton
 * Parents are all (i, tau) with tau > 0 that have a link to j
 * Plus contemporaneous neighbors with i < j (to avoid double-counting)
 */
static pcmci_varlag_t* get_parents(const pcmci_graph_t* g, int32_t j,
                                    int32_t* out_count) {
    if (!g || !out_count) {
        if (out_count) *out_count = 0;
        return NULL;
    }
    
    /* Count parents */
    int32_t count = 0;
    for (int32_t i = 0; i < g->n_vars; i++) {
        for (int32_t tau = 1; tau <= g->tau_max; tau++) {
            if (pcmci_graph_has_link(g, i, tau, j)) {
                count++;
            }
        }
        /* Include contemporaneous with i < j */
        if (i < j && pcmci_graph_has_link(g, i, 0, j)) {
            count++;
        }
    }
    
    *out_count = count;
    if (count == 0) return NULL;
    
    pcmci_varlag_t* parents = (pcmci_varlag_t*)malloc(count * sizeof(pcmci_varlag_t));
    if (!parents) {
        *out_count = 0;
        return NULL;
    }
    
    int32_t idx = 0;
    for (int32_t i = 0; i < g->n_vars; i++) {
        for (int32_t tau = 1; tau <= g->tau_max; tau++) {
            if (pcmci_graph_has_link(g, i, tau, j)) {
                parents[idx].var = i;
                parents[idx].tau = tau;
                idx++;
            }
        }
        if (i < j && pcmci_graph_has_link(g, i, 0, j)) {
            parents[idx].var = i;
            parents[idx].tau = 0;
            idx++;
        }
    }
    
    return parents;
}

/**
 * Merge two varlag arrays, removing duplicates
 * Returns newly allocated array
 */
static pcmci_varlag_t* merge_varlags(const pcmci_varlag_t* a, int32_t n_a,
                                      const pcmci_varlag_t* b, int32_t n_b,
                                      int32_t* out_count) {
    if (!out_count) return NULL;
    
    if (n_a == 0 && n_b == 0) {
        *out_count = 0;
        return NULL;
    }
    
    /* Allocate maximum possible size */
    pcmci_varlag_t* merged = (pcmci_varlag_t*)malloc((n_a + n_b) * sizeof(pcmci_varlag_t));
    if (!merged) {
        *out_count = 0;
        return NULL;
    }
    
    int32_t count = 0;
    
    /* Add all from a */
    for (int32_t i = 0; i < n_a; i++) {
        merged[count++] = a[i];
    }
    
    /* Add from b, checking for duplicates */
    for (int32_t i = 0; i < n_b; i++) {
        bool dup = false;
        for (int32_t j = 0; j < count; j++) {
            if (merged[j].var == b[i].var && merged[j].tau == b[i].tau) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            merged[count++] = b[i];
        }
    }
    
    *out_count = count;
    return merged;
}

void pcmci_mci(const pcmci_dataframe_t* df, pcmci_graph_t* g,
               const pcmci_config_t* config) {
    if (!df || !g || !config) return;
    
    int32_t n = g->n_vars;
    int32_t tau_max = g->tau_max;
    int32_t n_samples = df->T - df->tau_max;
    int32_t verbosity = config->verbosity;
    
    if (verbosity >= 1) {
        printf("MCI: computing causal strengths\n");
    }
    
    /* Set number of threads */
    if (config->n_threads > 0) {
        omp_set_num_threads(config->n_threads);
    }
    
    int64_t n_tests = 0;
    
    /* For each target variable j */
    #pragma omp parallel for schedule(dynamic) reduction(+:n_tests)
    for (int32_t j = 0; j < n; j++) {
        /* Get parents of j (including contemporaneous) */
        int32_t n_parents_j;
        pcmci_varlag_t* parents_j = get_parents(g, j, &n_parents_j);
        
        /* Extract Y = X_j(t) */
        int32_t y_len;
        double* Y = pcmci_extract_lagged(df, j, 0, &y_len);
        if (!Y) {
            free(parents_j);
            continue;
        }
        
        /* Test each link to j */
        for (int32_t i = 0; i < n; i++) {
            for (int32_t tau = (i < j ? 0 : 1); tau <= tau_max; tau++) {
                /* Skip contemporaneous with i >= j (stored at i < j) */
                if (tau == 0 && i >= j) continue;
                
                if (!pcmci_graph_has_link(g, i, tau, j)) continue;
                
                /* Extract X = X_i(t - tau) */
                int32_t x_len;
                double* X = pcmci_extract_lagged(df, i, tau, &x_len);
                if (!X) continue;
                
                /* Get parents of i at time t (we'll shift by tau) */
                int32_t n_parents_i_raw;
                pcmci_varlag_t* parents_i_raw = get_parents(g, i, &n_parents_i_raw);
                
                /* Shift parents of i by tau: parents at t become parents at t-tau
                 * So a parent (k, lag) of i at time t becomes (k, lag+tau) at time t-tau
                 * Only include if lag+tau <= tau_max */
                pcmci_varlag_t* parents_i = NULL;
                int32_t n_parents_i = 0;
                
                if (n_parents_i_raw > 0) {
                    parents_i = (pcmci_varlag_t*)malloc(n_parents_i_raw * sizeof(pcmci_varlag_t));
                    for (int32_t p = 0; p < n_parents_i_raw; p++) {
                        int32_t shifted_tau = parents_i_raw[p].tau + tau;
                        if (shifted_tau <= tau_max) {
                            parents_i[n_parents_i].var = parents_i_raw[p].var;
                            parents_i[n_parents_i].tau = shifted_tau;
                            n_parents_i++;
                        }
                    }
                }
                free(parents_i_raw);
                
                /* Conditioning set: parents(j) \ {(i, tau)} ∪ shifted_parents(i) */
                
                /* First, filter parents_j to exclude (i, tau) */
                pcmci_varlag_t* filtered_parents_j = NULL;
                int32_t n_filtered = 0;
                
                if (n_parents_j > 0) {
                    filtered_parents_j = (pcmci_varlag_t*)malloc(
                        n_parents_j * sizeof(pcmci_varlag_t));
                    
                    for (int32_t p = 0; p < n_parents_j; p++) {
                        if (!(parents_j[p].var == i && parents_j[p].tau == tau)) {
                            filtered_parents_j[n_filtered++] = parents_j[p];
                        }
                    }
                }
                
                /* Merge with shifted parents_i */
                int32_t n_cond;
                pcmci_varlag_t* cond_set = merge_varlags(
                    filtered_parents_j, n_filtered,
                    parents_i, n_parents_i,
                    &n_cond);
                
                free(filtered_parents_j);
                free(parents_i);
                
                /* Extract conditioning matrix */
                int32_t z_samples;
                double* Z = pcmci_extract_cond_set(df, cond_set, n_cond, &z_samples);
                
                /* Run partial correlation test */
                pcmci_ci_result_t res = pcmci_parcorr_test(X, Y, Z, n_samples, n_cond);
                
                /* Store results */
                int64_t idx = pcmci_graph_idx(n, tau_max, i, tau, j);
                g->val_matrix[idx] = res.val;
                g->pval_matrix[idx] = res.pvalue;
                
                n_tests++;
                
                free(cond_set);
                pcmci_mkl_free(Z);
                pcmci_mkl_free(X);
            }
        }
        
        pcmci_mkl_free(Y);
        free(parents_j);
    }
    
    if (verbosity >= 1) {
        printf("  Performed %ld MCI tests\n", (long)n_tests);
    }
}
