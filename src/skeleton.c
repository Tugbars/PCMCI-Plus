/**
 * @file skeleton.c
 * @brief PC-stable skeleton discovery for PCMCI+
 *
 * Optimized: zero allocations in hot paths using per-thread workspaces.
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

pcmci_graph_t *pcmci_graph_alloc(int32_t n_vars, int32_t tau_max)
{
    pcmci_graph_t *g = (pcmci_graph_t *)calloc(1, sizeof(pcmci_graph_t));
    if (!g)
        return NULL;

    g->n_vars = n_vars;
    g->tau_max = tau_max;

    int64_t total = (int64_t)n_vars * (tau_max + 1) * n_vars;

    g->adj = (bool *)calloc(total, sizeof(bool));
    g->link_types = (pcmci_link_type_t *)calloc(total, sizeof(pcmci_link_type_t));
    g->val_matrix = (double *)pcmci_calloc(total, sizeof(double));
    g->pval_matrix = (double *)pcmci_calloc(total, sizeof(double));
    g->sepsets = (pcmci_sepset_t *)calloc(total, sizeof(pcmci_sepset_t));

    if (!g->adj || !g->link_types || !g->val_matrix || !g->pval_matrix || !g->sepsets)
    {
        pcmci_graph_free(g);
        return NULL;
    }

    /* Initialize p-values to 1.0 */
    for (int64_t i = 0; i < total; i++)
    {
        g->pval_matrix[i] = 1.0;
    }

    /* Initialize adjacency */
    for (int32_t i = 0; i < n_vars; i++)
    {
        for (int32_t tau = 0; tau <= tau_max; tau++)
        {
            for (int32_t j = 0; j < n_vars; j++)
            {
                int64_t idx = pcmci_graph_idx(n_vars, tau_max, i, tau, j);
                if (tau == 0)
                {
                    g->adj[idx] = (i < j);
                    g->link_types[idx] = (i < j) ? PCMCI_LINK_UNDIRECTED : PCMCI_LINK_NONE;
                }
                else
                {
                    g->adj[idx] = true;
                    g->link_types[idx] = PCMCI_LINK_DIRECTED;
                }
            }
        }
    }

    return g;
}

void pcmci_graph_free(pcmci_graph_t *g)
{
    if (!g)
        return;

    if (g->sepsets)
    {
        int64_t total = (int64_t)g->n_vars * (g->tau_max + 1) * g->n_vars;
        for (int64_t i = 0; i < total; i++)
        {
            free(g->sepsets[i].vars);
            free(g->sepsets[i].taus);
        }
        free(g->sepsets);
    }

    free(g->adj);
    free(g->link_types);
    pcmci_mkl_free(g->val_matrix);
    pcmci_mkl_free(g->pval_matrix);
    free(g);
}

bool pcmci_graph_has_link(const pcmci_graph_t *g, int32_t i, int32_t tau, int32_t j)
{
    if (!g)
        return false;
    if (i < 0 || i >= g->n_vars || j < 0 || j >= g->n_vars)
        return false;
    if (tau < 0 || tau > g->tau_max)
        return false;
    return g->adj[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)];
}

void pcmci_graph_set_link(pcmci_graph_t *g, int32_t i, int32_t tau, int32_t j, bool exists)
{
    if (!g)
        return;
    if (i < 0 || i >= g->n_vars || j < 0 || j >= g->n_vars)
        return;
    if (tau < 0 || tau > g->tau_max)
        return;
    g->adj[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)] = exists;
}

double pcmci_graph_get_val(const pcmci_graph_t *g, int32_t i, int32_t tau, int32_t j)
{
    if (!g)
        return 0.0;
    return g->val_matrix[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)];
}

double pcmci_graph_get_pval(const pcmci_graph_t *g, int32_t i, int32_t tau, int32_t j)
{
    if (!g)
        return 1.0;
    return g->pval_matrix[pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j)];
}

pcmci_varlag_t *pcmci_graph_get_neighbors(const pcmci_graph_t *g, int32_t j,
                                          int32_t *out_count)
{
    if (!g || !out_count)
    {
        if (out_count)
            *out_count = 0;
        return NULL;
    }

    int32_t count = 0;
    for (int32_t i = 0; i < g->n_vars; i++)
    {
        for (int32_t tau = 0; tau <= g->tau_max; tau++)
        {
            if (pcmci_graph_has_link(g, i, tau, j))
                count++;
        }
    }

    *out_count = count;
    if (count == 0)
        return NULL;

    pcmci_varlag_t *neighbors = (pcmci_varlag_t *)malloc(count * sizeof(pcmci_varlag_t));
    if (!neighbors)
    {
        *out_count = 0;
        return NULL;
    }

    int32_t idx = 0;
    for (int32_t i = 0; i < g->n_vars; i++)
    {
        for (int32_t tau = 0; tau <= g->tau_max; tau++)
        {
            if (pcmci_graph_has_link(g, i, tau, j))
            {
                neighbors[idx].var = i;
                neighbors[idx].tau = tau;
                idx++;
            }
        }
    }

    return neighbors;
}

void pcmci_graph_print(const pcmci_graph_t *g, const char **var_names)
{
    if (!g)
    {
        printf("Graph: NULL\n");
        return;
    }

    printf("Causal Graph (n_vars=%d, tau_max=%d)\n", g->n_vars, g->tau_max);
    printf("========================================\n");

    for (int32_t j = 0; j < g->n_vars; j++)
    {
        for (int32_t i = 0; i < g->n_vars; i++)
        {
            for (int32_t tau = 0; tau <= g->tau_max; tau++)
            {
                if (pcmci_graph_has_link(g, i, tau, j))
                {
                    int64_t idx = pcmci_graph_idx(g->n_vars, g->tau_max, i, tau, j);
                    if (var_names)
                    {
                        printf("  %s(t-%d) --> %s(t): val=%.4f, pval=%.4e\n",
                               var_names[i], tau, var_names[j],
                               g->val_matrix[idx], g->pval_matrix[idx]);
                    }
                    else
                    {
                        printf("  X%d(t-%d) --> X%d(t): val=%.4f, pval=%.4e\n",
                               i, tau, j, g->val_matrix[idx], g->pval_matrix[idx]);
                    }
                }
            }
        }
    }
}

/*============================================================================
 * Helper: get neighbors into pre-allocated buffer
 *============================================================================*/

static int32_t get_neighbors_into(const pcmci_graph_t *g, int32_t j,
                                  pcmci_varlag_t *neighbors)
{
    int32_t count = 0;
    for (int32_t i = 0; i < g->n_vars; i++)
    {
        for (int32_t tau = 0; tau <= g->tau_max; tau++)
        {
            if (pcmci_graph_has_link(g, i, tau, j))
            {
                neighbors[count].var = i;
                neighbors[count].tau = tau;
                count++;
            }
        }
    }
    return count;
}

/*============================================================================
 * Inline combination iterator (no allocation)
 *============================================================================*/

static void comb_init(int32_t *indices, int32_t k)
{
    for (int32_t i = 0; i < k; i++)
        indices[i] = i;
}

static bool comb_next(int32_t *indices, int32_t n, int32_t k)
{
    if (k == 0)
        return false;

    int32_t i = k - 1;
    while (i >= 0 && indices[i] == n - k + i)
        i--;

    if (i < 0)
        return false;

    indices[i]++;
    for (int32_t j = i + 1; j < k; j++)
    {
        indices[j] = indices[j - 1] + 1;
    }
    return true;
}

/*============================================================================
 * Skeleton Discovery (PC-Stable) - Lock-free with thread-local storage
 *============================================================================*/

typedef struct
{
    int32_t i, tau, j;
    int32_t sep_size;
    int32_t sep_vars[32];
    int32_t sep_taus[32];
} removal_t;

/* Thread-local removal buffer */
typedef struct
{
    removal_t *removals;
    int32_t count;
    int32_t capacity;
} thread_removals_t;

static void thread_removals_init(thread_removals_t *tr, int32_t initial_cap)
{
    tr->capacity = initial_cap > 0 ? initial_cap : 64;
    tr->removals = (removal_t *)malloc(tr->capacity * sizeof(removal_t));
    tr->count = 0;
}

static void thread_removals_add(thread_removals_t *tr, const removal_t *r)
{
    if (tr->count >= tr->capacity)
    {
        tr->capacity *= 2;
        tr->removals = (removal_t *)realloc(tr->removals, tr->capacity * sizeof(removal_t));
    }
    tr->removals[tr->count++] = *r;
}

static void thread_removals_free(thread_removals_t *tr)
{
    free(tr->removals);
    tr->removals = NULL;
    tr->count = 0;
    tr->capacity = 0;
}

pcmci_graph_t *pcmci_skeleton(const pcmci_dataframe_t *df, const pcmci_config_t *config)
{
    if (!df || !config)
        return NULL;

    int32_t n = df->n_vars;
    int32_t tau_max = config->tau_max;
    double alpha = config->alpha_level;
    int32_t max_cond = config->max_cond_dim;
    int32_t verbosity = config->verbosity;
    int32_t n_samples = df->T - df->tau_max;
    int32_t max_neighbors = n * (tau_max + 1);

    if (max_cond < 0)
        max_cond = max_neighbors;

    pcmci_graph_t *g = pcmci_graph_alloc(n, tau_max);
    if (!g)
        return NULL;

    int32_t n_threads = config->n_threads > 0 ? config->n_threads : omp_get_max_threads();
    omp_set_num_threads(n_threads);

    /* Allocate per-thread resources ONCE */
    pcmci_workspace_t **workspaces = (pcmci_workspace_t **)malloc(n_threads * sizeof(pcmci_workspace_t *));
    pcmci_varlag_t **neighbor_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));
    pcmci_varlag_t **other_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));
    pcmci_varlag_t **cond_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));
    int32_t **comb_bufs = (int32_t **)malloc(n_threads * sizeof(int32_t *));
    thread_removals_t *thread_removals = (thread_removals_t *)malloc(n_threads * sizeof(thread_removals_t));

    for (int32_t t = 0; t < n_threads; t++)
    {
        workspaces[t] = pcmci_workspace_create(n_samples, max_cond > 0 ? max_cond : 1);
        neighbor_bufs[t] = (pcmci_varlag_t *)malloc(max_neighbors * sizeof(pcmci_varlag_t));
        other_bufs[t] = (pcmci_varlag_t *)malloc(max_neighbors * sizeof(pcmci_varlag_t));
        cond_bufs[t] = (pcmci_varlag_t *)malloc((max_cond > 0 ? max_cond : 1) * sizeof(pcmci_varlag_t));
        comb_bufs[t] = (int32_t *)malloc((max_cond > 0 ? max_cond : 1) * sizeof(int32_t));
    }

    /* Iterate over conditioning set sizes */
    for (int32_t cond_dim = 0; cond_dim <= max_cond; cond_dim++)
    {
        if (verbosity >= 1)
            printf("Skeleton: testing cond_dim = %d\n", cond_dim);

        /* Check if any link has enough neighbors */
        bool any_testable = false;
        for (int32_t j = 0; j < n && !any_testable; j++)
        {
            int32_t cnt = get_neighbors_into(g, j, neighbor_bufs[0]);
            if (cnt > cond_dim)
                any_testable = true;
        }

        if (!any_testable)
        {
            if (verbosity >= 1)
                printf("  No links with enough neighbors, stopping\n");
            break;
        }

        /* Initialize per-thread removal buffers */
        for (int32_t t = 0; t < n_threads; t++)
        {
            thread_removals_init(&thread_removals[t], 64);
        }

/* Parallel skeleton discovery - NO LOCKS */
#pragma omp parallel for schedule(dynamic)
        for (int32_t j = 0; j < n; j++)
        {
            int32_t tid = omp_get_thread_num();
            pcmci_workspace_t *ws = workspaces[tid];
            pcmci_varlag_t *neighbors = neighbor_bufs[tid];
            pcmci_varlag_t *other_neighbors = other_bufs[tid];
            pcmci_varlag_t *cond_set = cond_bufs[tid];
            int32_t *comb_idx = comb_bufs[tid];
            thread_removals_t *my_removals = &thread_removals[tid];

            int32_t n_neighbors = get_neighbors_into(g, j, neighbors);
            if (n_neighbors <= cond_dim)
                continue;

            /* Extract Y into workspace */
            pcmci_extract_lagged_into(df, j, 0, ws->Y_buf);

            for (int32_t nb_idx = 0; nb_idx < n_neighbors; nb_idx++)
            {
                int32_t i = neighbors[nb_idx].var;
                int32_t tau = neighbors[nb_idx].tau;

                /* Extract X into workspace */
                pcmci_extract_lagged_into(df, i, tau, ws->X_buf);

                /* Build other_neighbors (excluding current) */
                int32_t other_count = 0;
                for (int32_t k = 0; k < n_neighbors; k++)
                {
                    if (k != nb_idx)
                        other_neighbors[other_count++] = neighbors[k];
                }

                /* Iterate over cond_dim-subsets */
                bool found_indep = false;
                removal_t removal = {i, tau, j, 0, {0}, {0}};

                if (cond_dim == 0)
                {
                    /* Test unconditional */
                    pcmci_ci_result_t res = pcmci_parcorr_ws(ws->X_buf, ws->Y_buf, NULL, n_samples, 0, ws);
                    if (res.pvalue > alpha)
                        found_indep = true;
                }
                else
                {
                    /* Initialize combination */
                    comb_init(comb_idx, cond_dim);
                    bool first = true;

                    while (!found_indep)
                    {
                        if (!first)
                        {
                            if (!comb_next(comb_idx, other_count, cond_dim))
                                break;
                        }
                        first = false;

                        /* Build conditioning set */
                        for (int32_t c = 0; c < cond_dim; c++)
                        {
                            cond_set[c] = other_neighbors[comb_idx[c]];
                        }

                        /* Extract Z into workspace */
                        pcmci_extract_cond_set_into(df, cond_set, cond_dim, ws->Z_buf);

                        /* Test */
                        pcmci_ci_result_t res = pcmci_parcorr_ws(ws->X_buf, ws->Y_buf, ws->Z_buf, n_samples, cond_dim, ws);

                        if (res.pvalue > alpha)
                        {
                            found_indep = true;
                            removal.sep_size = cond_dim;
                            for (int32_t c = 0; c < cond_dim && c < 32; c++)
                            {
                                removal.sep_vars[c] = cond_set[c].var;
                                removal.sep_taus[c] = cond_set[c].tau;
                            }
                        }
                    }
                }

                if (found_indep)
                {
                    /* Thread-local append - NO LOCK */
                    thread_removals_add(my_removals, &removal);
                }
            }
        }

        /* Batch merge: apply all removals from all threads (sequential, fast) */
        int32_t total_removals = 0;
        for (int32_t t = 0; t < n_threads; t++)
        {
            thread_removals_t *tr = &thread_removals[t];
            for (int32_t r = 0; r < tr->count; r++)
            {
                removal_t *rem = &tr->removals[r];
                int64_t idx = pcmci_graph_idx(n, tau_max, rem->i, rem->tau, rem->j);
                g->adj[idx] = false;
                g->link_types[idx] = PCMCI_LINK_NONE;

                int32_t sep_size = rem->sep_size;
                if (sep_size > 0)
                {
                    g->sepsets[idx].size = sep_size;
                    g->sepsets[idx].vars = (int32_t *)malloc(sep_size * sizeof(int32_t));
                    g->sepsets[idx].taus = (int32_t *)malloc(sep_size * sizeof(int32_t));
                    memcpy(g->sepsets[idx].vars, rem->sep_vars, sep_size * sizeof(int32_t));
                    memcpy(g->sepsets[idx].taus, rem->sep_taus, sep_size * sizeof(int32_t));
                }
            }
            total_removals += tr->count;
            thread_removals_free(tr);
        }

        if (verbosity >= 1)
            printf("  Removed %d links\n", total_removals);
    }

    /* Cleanup per-thread resources */
    for (int32_t t = 0; t < n_threads; t++)
    {
        pcmci_workspace_free(workspaces[t]);
        free(neighbor_bufs[t]);
        free(other_bufs[t]);
        free(cond_bufs[t]);
        free(comb_bufs[t]);
    }
    free(workspaces);
    free(neighbor_bufs);
    free(other_bufs);
    free(cond_bufs);
    free(comb_bufs);
    free(thread_removals);

    return g;
}

/* pcmci_comb_* functions are in parcorr.c */