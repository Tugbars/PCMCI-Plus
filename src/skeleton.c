/**
 * @file skeleton.c
 * @brief PC-stable skeleton discovery for PCMCI+
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
 * Graph Implementation
 *============================================================================*/

pcmci_graph_t* pcmci_graph_alloc(int32_t n_vars, int32_t tau_max) {
    pcmci_graph_t* g = (pcmci_graph_t*)calloc(1, sizeof(pcmci_graph_t));
    if (!g) return NULL;
    
    g->n_vars = n_vars;
    g->tau_max = tau_max;
    
    int64_t total = (int64_t)n_vars * (tau_max + 1) * n_vars;
    
    g->adj = (bool*)calloc(total, sizeof(bool));
    g->link_types = (pcmci_link_type_t*)calloc(total, sizeof(pcmci_link_type_t));
    g->val_matrix = (double*)pcmci_calloc(total, sizeof(double));
    g->pval_matrix = (double*)pcmci_calloc(total, sizeof(double));
    g->sepsets = (pcmci_sepset_t*)calloc(total, sizeof(pcmci_sepset_t));
    
    if (!g->adj || !g->link_types || !g->val_matrix || !g->pval_matrix || !g->sepsets) {
        pcmci_graph_free(g);
        return NULL;
    }
    
    /* Initialize p-values to 1.0 */
    for (int64_t i = 0; i < total; i++) {
        g->pval_matrix[i] = 1.0;
    }
    
    /* Initialize adjacency: 
     * - All lagged links exist (tau > 0)
     * - Contemporaneous (tau = 0): only i < j to avoid duplicates, no self-loops
     */
    for (int32_t i = 0; i < n_vars; i++) {
        for (int32_t tau = 0; tau <= tau_max; tau++) {
            for (int32_t j = 0; j < n_vars; j++) {
                int64_t idx = pcmci_graph_idx(n_vars, tau_max, i, tau, j);
                
                if (tau == 0) {
                    /* Contemporaneous: undirected, store only i < j */
                    g->adj[idx] = (i < j);
                    g->link_types[idx] = (i < j) ? PCMCI_LINK_UNDIRECTED : PCMCI_LINK_NONE;
                } else {
                    /* Lagged: directed by time */
                    g->adj[idx] = true;
                    g->link_types[idx] = PCMCI_LINK_DIRECTED;
                }
            }
        }
    }
    
    return g;
}

void pcmci_graph_free(pcmci_graph_t* g) {
    if (!g) return;
    
    if (g->sepsets) {
        int64_t total = (int64_t)g->n_vars * (g->tau_max + 1) * g->n_vars;
        for (int64_t i = 0; i < total; i++) {
            if (g->sepsets[i].vars) free(g->sepsets[i].vars);
            if (g->sepsets[i].taus) free(g->sepsets[i].taus);
        }
        free(g->sepsets);
    }
    
    free(g->adj);
    free(g->link_types);
    pcmci_mkl_free(g->val_matrix);
    pcmci_mkl_free(g->pval_matrix);
    free(g);
}

bool pcmci_graph_has_link(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j) {
    if (!g) return false;
    if (i < 0 || i >= g->n_vars || j < 0 || j >= g->n_vars) return false;
    if (tau < 0 || tau > g->tau_max) return false;
    
    return g->adj[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)];
}

void pcmci_graph_set_link(pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j, bool exists) {
    if (!g) return;
    if (i < 0 || i >= g->n_vars || j < 0 || j >= g->n_vars) return;
    if (tau < 0 || tau > g->tau_max) return;
    
    g->adj[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)] = exists;
}

double pcmci_graph_get_val(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j) {
    if (!g) return 0.0;
    return g->val_matrix[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)];
}

double pcmci_graph_get_pval(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j) {
    if (!g) return 1.0;
    return g->pval_matrix[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)];
}

pcmci_varlag_t* pcmci_graph_get_neighbors(const pcmci_graph_t* g, int32_t j,
                                           int32_t* out_count) {
    if (!g || !out_count) {
        if (out_count) *out_count = 0;
        return NULL;
    }
    
    /* Count neighbors */
    int32_t count = 0;
    for (int32_t i = 0; i < g->n_vars; i++) {
        for (int32_t tau = 0; tau <= g->tau_max; tau++) {
            if (pcmci_graph_has_link(g, i, tau, j)) {
                count++;
            }
        }
    }
    
    *out_count = count;
    if (count == 0) return NULL;
    
    pcmci_varlag_t* neighbors = (pcmci_varlag_t*)malloc(count * sizeof(pcmci_varlag_t));
    if (!neighbors) {
        *out_count = 0;
        return NULL;
    }
    
    int32_t idx = 0;
    for (int32_t i = 0; i < g->n_vars; i++) {
        for (int32_t tau = 0; tau <= g->tau_max; tau++) {
            if (pcmci_graph_has_link(g, i, tau, j)) {
                neighbors[idx].var = i;
                neighbors[idx].tau = tau;
                idx++;
            }
        }
    }
    
    return neighbors;
}

void pcmci_graph_print(const pcmci_graph_t* g, const char** var_names) {
    if (!g) {
        printf("Graph: NULL\n");
        return;
    }
    
    printf("Causal Graph (n_vars=%d, tau_max=%d)\n", g->n_vars, g->tau_max);
    printf("========================================\n");
    
    int32_t link_count = 0;
    
    for (int32_t j = 0; j < g->n_vars; j++) {
        const char* target = var_names ? var_names[j] : NULL;
        
        for (int32_t i = 0; i < g->n_vars; i++) {
            for (int32_t tau = 0; tau <= g->tau_max; tau++) {
                if (pcmci_graph_has_link(g, i, tau, j)) {
                    int64_t idx = pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j);
                    double val = g->val_matrix[idx];
                    double pval = g->pval_matrix[idx];
                    
                    const char* source = var_names ? var_names[i] : NULL;
                    
                    if (source && target) {
                        printf("  %s(t-%d) --> %s(t): val=%.4f, pval=%.4e\n",
                               source, tau, target, val, pval);
                    } else {
                        printf("  X%d(t-%d) --> X%d(t): val=%.4f, pval=%.4e\n",
                               i, tau, j, val, pval);
                    }
                    link_count++;
                }
            }
        }
    }
    
    printf("Total links: %d\n", link_count);
}

/*============================================================================
 * Skeleton Discovery (PC-Stable)
 *============================================================================*/

/* Structure to hold a pending removal (for PC-stable) */
typedef struct {
    int32_t i;
    int32_t tau;
    int32_t j;
    pcmci_sepset_t sep;
} removal_t;

pcmci_graph_t* pcmci_skeleton(const pcmci_dataframe_t* df, const pcmci_config_t* config) {
    if (!df || !config) return NULL;
    
    int32_t n = df->n_vars;
    int32_t tau_max = config->tau_max;
    double alpha = config->alpha_level;
    int32_t max_cond = config->max_cond_dim;
    int32_t verbosity = config->verbosity;
    
    if (max_cond < 0) {
        max_cond = n * (tau_max + 1);  /* Unlimited */
    }
    
    /* Allocate graph with all initial links */
    pcmci_graph_t* g = pcmci_graph_alloc(n, tau_max);
    if (!g) return NULL;
    
    /* Set number of threads */
    if (config->n_threads > 0) {
        omp_set_num_threads(config->n_threads);
    }
    
    int32_t n_samples = df->T - tau_max;
    
    /* Iterate over conditioning set sizes */
    for (int32_t cond_dim = 0; cond_dim <= max_cond; cond_dim++) {
        if (verbosity >= 1) {
            printf("Skeleton: testing cond_dim = %d\n", cond_dim);
        }
        
        /* Collect removals for PC-stable */
        removal_t* removals = NULL;
        int32_t n_removals = 0;
        int32_t removals_cap = 0;
        
        /* Lock for thread-safe removal collection */
        omp_lock_t removal_lock;
        omp_init_lock(&removal_lock);
        
        /* Check if any link has enough neighbors for this cond_dim */
        bool any_testable = false;
        for (int32_t j = 0; j < n && !any_testable; j++) {
            int32_t n_neighbors;
            pcmci_varlag_t* neighbors = pcmci_graph_get_neighbors(g, j, &n_neighbors);
            if (n_neighbors > cond_dim) any_testable = true;
            free(neighbors);
        }
        
        if (!any_testable) {
            omp_destroy_lock(&removal_lock);
            if (verbosity >= 1) {
                printf("  No links with enough neighbors, stopping\n");
            }
            break;
        }
        
        /* For each target variable j */
        #pragma omp parallel for schedule(dynamic)
        for (int32_t j = 0; j < n; j++) {
            /* Get current neighbors of j */
            int32_t n_neighbors;
            pcmci_varlag_t* neighbors = pcmci_graph_get_neighbors(g, j, &n_neighbors);
            
            if (n_neighbors <= cond_dim) {
                free(neighbors);
                continue;
            }
            
            /* Extract Y = X_j(t) */
            int32_t y_len;
            double* Y = pcmci_extract_lagged(df, j, 0, &y_len);
            if (!Y) {
                free(neighbors);
                continue;
            }
            
            /* For each neighbor (i, tau) of j */
            for (int32_t nb_idx = 0; nb_idx < n_neighbors; nb_idx++) {
                int32_t i = neighbors[nb_idx].var;
                int32_t tau = neighbors[nb_idx].tau;
                
                /* Extract X = X_i(t - tau) */
                int32_t x_len;
                double* X = pcmci_extract_lagged(df, i, tau, &x_len);
                if (!X) continue;
                
                /* Build neighbor list excluding (i, tau) */
                pcmci_varlag_t* other_neighbors = (pcmci_varlag_t*)malloc(
                    (n_neighbors - 1) * sizeof(pcmci_varlag_t));
                int32_t other_count = 0;
                
                if (other_neighbors) {
                    for (int32_t k = 0; k < n_neighbors; k++) {
                        if (k != nb_idx) {
                            other_neighbors[other_count++] = neighbors[k];
                        }
                    }
                }
                
                /* Iterate over cond_dim-subsets of other neighbors */
                pcmci_comb_iter_t* comb = pcmci_comb_init(other_count, cond_dim);
                
                bool found_independence = false;
                pcmci_sepset_t best_sep = {NULL, NULL, 0};
                
                while (pcmci_comb_next(comb) && !found_independence) {
                    /* Build conditioning set */
                    pcmci_varlag_t* cond_set = NULL;
                    if (cond_dim > 0) {
                        cond_set = (pcmci_varlag_t*)malloc(cond_dim * sizeof(pcmci_varlag_t));
                        for (int32_t c = 0; c < cond_dim; c++) {
                            cond_set[c] = other_neighbors[comb->indices[c]];
                        }
                    }
                    
                    /* Extract conditioning matrix */
                    int32_t z_samples;
                    double* Z = pcmci_extract_cond_set(df, cond_set, cond_dim, &z_samples);
                    
                    /* Run partial correlation test */
                    pcmci_ci_result_t res = pcmci_parcorr_test(X, Y, Z, n_samples, cond_dim);
                    
                    if (res.pvalue > alpha) {
                        /* Found conditional independence! */
                        found_independence = true;
                        
                        /* Store separating set */
                        if (cond_dim > 0) {
                            best_sep.vars = (int32_t*)malloc(cond_dim * sizeof(int32_t));
                            best_sep.taus = (int32_t*)malloc(cond_dim * sizeof(int32_t));
                            best_sep.size = cond_dim;
                            
                            for (int32_t c = 0; c < cond_dim; c++) {
                                best_sep.vars[c] = cond_set[c].var;
                                best_sep.taus[c] = cond_set[c].tau;
                            }
                        }
                    }
                    
                    pcmci_mkl_free(Z);
                    free(cond_set);
                }
                
                pcmci_comb_free(comb);
                free(other_neighbors);
                pcmci_mkl_free(X);
                
                /* Add to removal list if independent */
                if (found_independence) {
                    omp_set_lock(&removal_lock);
                    
                    if (n_removals >= removals_cap) {
                        removals_cap = removals_cap ? removals_cap * 2 : 64;
                        removals = (removal_t*)realloc(removals, 
                            removals_cap * sizeof(removal_t));
                    }
                    
                    removals[n_removals].i = i;
                    removals[n_removals].tau = tau;
                    removals[n_removals].j = j;
                    removals[n_removals].sep = best_sep;
                    n_removals++;
                    
                    omp_unset_lock(&removal_lock);
                }
            }
            
            pcmci_mkl_free(Y);
            free(neighbors);
        }
        
        omp_destroy_lock(&removal_lock);
        
        /* Apply all removals (PC-stable) */
        for (int32_t r = 0; r < n_removals; r++) {
            int32_t i = removals[r].i;
            int32_t tau = removals[r].tau;
            int32_t j = removals[r].j;
            
            int64_t idx = pcmci_graph_idx(n, tau_max, i, tau, j);
            
            g->adj[idx] = false;
            g->link_types[idx] = PCMCI_LINK_NONE;
            
            /* Store separating set */
            g->sepsets[idx] = removals[r].sep;
        }
        
        if (verbosity >= 1) {
            printf("  Removed %d links\n", n_removals);
        }
        
        free(removals);
    }
    
    return g;
}
