/**
 * @file mci.c
 * @brief Momentary Conditional Independence (MCI) phase for PCMCI+
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
 * Helper: get parents into pre-allocated buffer
 *============================================================================*/

static int32_t get_parents_into(const pcmci_graph_t *g, int32_t j,
                                pcmci_varlag_t *parents)
{
    int32_t count = 0;
    for (int32_t i = 0; i < g->n_vars; i++)
    {
        for (int32_t tau = 1; tau <= g->tau_max; tau++)
        {
            if (pcmci_graph_has_link(g, i, tau, j))
            {
                parents[count].var = i;
                parents[count].tau = tau;
                count++;
            }
        }
        if (i < j && pcmci_graph_has_link(g, i, 0, j))
        {
            parents[count].var = i;
            parents[count].tau = 0;
            count++;
        }
    }
    return count;
}

/*============================================================================
 * Helper: filter parents excluding (var, tau)
 *============================================================================*/

static int32_t filter_parents_into(const pcmci_varlag_t *parents, int32_t n_parents,
                                   int32_t excl_var, int32_t excl_tau,
                                   pcmci_varlag_t *out)
{
    int32_t count = 0;
    for (int32_t p = 0; p < n_parents; p++)
    {
        if (!(parents[p].var == excl_var && parents[p].tau == excl_tau))
        {
            out[count++] = parents[p];
        }
    }
    return count;
}

/*============================================================================
 * Helper: shift parents by tau offset for source variable conditioning
 *============================================================================*/

static int32_t shift_parents_into(const pcmci_varlag_t *parents, int32_t n_parents,
                                  int32_t tau_offset, int32_t tau_max,
                                  pcmci_varlag_t *out)
{
    int32_t count = 0;
    for (int32_t p = 0; p < n_parents; p++)
    {
        int32_t new_tau = parents[p].tau + tau_offset;
        if (new_tau >= 0 && new_tau <= tau_max)
        {
            out[count].var = parents[p].var;
            out[count].tau = new_tau;
            count++;
        }
    }
    return count;
}

/*============================================================================
 * Helper: merge two varlag arrays (remove duplicates)
 *============================================================================*/

static int32_t merge_varlags_into(const pcmci_varlag_t *a, int32_t n_a,
                                  const pcmci_varlag_t *b, int32_t n_b,
                                  pcmci_varlag_t *out)
{
    int32_t count = 0;

    for (int32_t i = 0; i < n_a; i++)
    {
        out[count++] = a[i];
    }

    for (int32_t i = 0; i < n_b; i++)
    {
        bool dup = false;
        for (int32_t j = 0; j < count; j++)
        {
            if (out[j].var == b[i].var && out[j].tau == b[i].tau)
            {
                dup = true;
                break;
            }
        }
        if (!dup)
            out[count++] = b[i];
    }

    return count;
}

/*============================================================================
 * MCI Phase - Zero allocations in hot path
 *============================================================================*/

void pcmci_mci(const pcmci_dataframe_t *df, pcmci_graph_t *g,
               const pcmci_config_t *config)
{
    if (!df || !g || !config)
        return;

    int32_t n = g->n_vars;
    int32_t tau_max = g->tau_max;
    int32_t n_samples = df->T - df->tau_max;
    int32_t verbosity = config->verbosity;
    int32_t max_parents = n * (tau_max + 1);
    int32_t max_cond = max_parents * 2; /* parents(j) + parents(i) */

    if (verbosity >= 1)
        printf("MCI: computing causal strengths\n");

    int32_t n_threads = config->n_threads > 0 ? config->n_threads : omp_get_max_threads();
    omp_set_num_threads(n_threads);

    /* Per-thread resources */
    pcmci_workspace_t **workspaces = (pcmci_workspace_t **)malloc(n_threads * sizeof(pcmci_workspace_t *));
    pcmci_varlag_t **parents_j_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));
    pcmci_varlag_t **parents_i_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));
    pcmci_varlag_t **filtered_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));
    pcmci_varlag_t **shifted_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));
    pcmci_varlag_t **cond_bufs = (pcmci_varlag_t **)malloc(n_threads * sizeof(pcmci_varlag_t *));

    for (int32_t t = 0; t < n_threads; t++)
    {
        workspaces[t] = pcmci_workspace_create(n_samples, max_cond > 0 ? max_cond : 1);
        parents_j_bufs[t] = (pcmci_varlag_t *)malloc(max_parents * sizeof(pcmci_varlag_t));
        parents_i_bufs[t] = (pcmci_varlag_t *)malloc(max_parents * sizeof(pcmci_varlag_t));
        filtered_bufs[t] = (pcmci_varlag_t *)malloc(max_parents * sizeof(pcmci_varlag_t));
        shifted_bufs[t] = (pcmci_varlag_t *)malloc(max_parents * sizeof(pcmci_varlag_t));
        cond_bufs[t] = (pcmci_varlag_t *)malloc(max_cond * sizeof(pcmci_varlag_t));
    }

    int64_t n_tests = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : n_tests)
    for (int32_t j = 0; j < n; j++)
    {
        int32_t tid = omp_get_thread_num();
        pcmci_workspace_t *ws = workspaces[tid];
        pcmci_varlag_t *parents_j = parents_j_bufs[tid];
        pcmci_varlag_t *parents_i = parents_i_bufs[tid];
        pcmci_varlag_t *filtered = filtered_bufs[tid];
        pcmci_varlag_t *shifted = shifted_bufs[tid];
        pcmci_varlag_t *cond_set = cond_bufs[tid];

        int32_t n_parents_j = get_parents_into(g, j, parents_j);

        /* Extract Y = X_j(t) */
        pcmci_extract_lagged_into(df, j, 0, ws->Y_buf);

        /* Test each link to j */
        for (int32_t i = 0; i < n; i++)
        {
            for (int32_t tau = (i < j ? 0 : 1); tau <= tau_max; tau++)
            {
                if (tau == 0 && i >= j)
                    continue;
                if (!pcmci_graph_has_link(g, i, tau, j))
                    continue;

                /* Extract X = X_i(t - tau) */
                pcmci_extract_lagged_into(df, i, tau, ws->X_buf);

                /* Get parents of i */
                int32_t n_parents_i = get_parents_into(g, i, parents_i);

                /* Filter parents_j to exclude (i, tau) */
                int32_t n_filtered = filter_parents_into(parents_j, n_parents_j, i, tau, filtered);

                /* Shift parents_i by tau */
                int32_t n_shifted = shift_parents_into(parents_i, n_parents_i, tau, tau_max, shifted);

                /* Merge: cond_set = filtered ∪ shifted */
                int32_t n_cond = merge_varlags_into(filtered, n_filtered, shifted, n_shifted, cond_set);

                /* Extract conditioning matrix */
                if (n_cond > 0)
                {
                    pcmci_extract_cond_set_into(df, cond_set, n_cond, ws->Z_buf);
                }

                /* Run partial correlation test */
                pcmci_ci_result_t res = pcmci_parcorr_ws(ws->X_buf, ws->Y_buf,
                                                         n_cond > 0 ? ws->Z_buf : NULL,
                                                         n_samples, n_cond, ws);

                /* Store results */
                int64_t idx = pcmci_graph_idx(n, tau_max, i, tau, j);
                g->val_matrix[idx] = res.val;
                g->pval_matrix[idx] = res.pvalue;

                n_tests++;
            }
        }
    }

    /* Cleanup */
    for (int32_t t = 0; t < n_threads; t++)
    {
        pcmci_workspace_free(workspaces[t]);
        free(parents_j_bufs[t]);
        free(parents_i_bufs[t]);
        free(filtered_bufs[t]);
        free(shifted_bufs[t]);
        free(cond_bufs[t]);
    }
    free(workspaces);
    free(parents_j_bufs);
    free(parents_i_bufs);
    free(filtered_bufs);
    free(shifted_bufs);
    free(cond_bufs);

    if (verbosity >= 1)
        printf("  Performed %ld MCI tests\n", (long)n_tests);
}