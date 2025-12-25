/**
 * @file dataframe.c
 * @brief Time series dataframe implementation
 *
 * Optimized: Aligned allocation and memcpy-based data extraction.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_internal.h"
#include <stdlib.h>
#include <string.h>

/* Windows compatibility for strdup */
#ifdef _WIN32
#define pcmci_strdup _strdup
#else
#define pcmci_strdup strdup
#endif

/*============================================================================
 * Construction / Destruction
 *============================================================================*/

pcmci_dataframe_t *pcmci_dataframe_create(double *data, int32_t n_vars,
                                          int32_t T, int32_t tau_max)
{
    pcmci_dataframe_t *df = (pcmci_dataframe_t *)calloc(1, sizeof(pcmci_dataframe_t));
    if (!df)
        return NULL;

    df->data = data;
    df->n_vars = n_vars;
    df->T = T;
    df->tau_max = tau_max;
    df->var_names = NULL;
    df->owns_data = false;

    return df;
}

pcmci_dataframe_t *pcmci_dataframe_create_copy(const double *data, int32_t n_vars,
                                               int32_t T, int32_t tau_max)
{
    pcmci_dataframe_t *df = pcmci_dataframe_alloc(n_vars, T, tau_max);
    if (!df)
        return NULL;

    /* OPTIMIZATION: Use memcpy for bulk data copy.
     * Since data is contiguous [n_vars * T], this is maximum speed. */
    memcpy(df->data, data, (size_t)n_vars * T * sizeof(double));
    return df;
}

pcmci_dataframe_t *pcmci_dataframe_alloc(int32_t n_vars, int32_t T, int32_t tau_max)
{
    pcmci_dataframe_t *df = (pcmci_dataframe_t *)calloc(1, sizeof(pcmci_dataframe_t));
    if (!df)
        return NULL;

    /* OPTIMIZATION: Use aligned malloc (pcmci_malloc) for vectorization support later. */
    df->data = (double *)pcmci_malloc((size_t)n_vars * T * sizeof(double));
    if (!df->data)
    {
        free(df);
        return NULL;
    }

    /* Initialize to zero to prevent NaN propagation if accessed uninitialized */
    memset(df->data, 0, (size_t)n_vars * T * sizeof(double));

    df->n_vars = n_vars;
    df->T = T;
    df->tau_max = tau_max;
    df->var_names = NULL;
    df->owns_data = true;

    return df;
}

void pcmci_dataframe_set_names(pcmci_dataframe_t *df, const char **names)
{
    if (!df || !names)
        return;

    /* Free existing names */
    if (df->var_names)
    {
        for (int32_t i = 0; i < df->n_vars; i++)
        {
            free(df->var_names[i]);
        }
        free(df->var_names);
    }

    /* Copy new names */
    df->var_names = (char **)malloc(df->n_vars * sizeof(char *));
    for (int32_t i = 0; i < df->n_vars; i++)
    {
        df->var_names[i] = pcmci_strdup(names[i]);
    }
}

void pcmci_dataframe_free(pcmci_dataframe_t *df)
{
    if (!df)
        return;

    if (df->owns_data && df->data)
    {
        pcmci_mkl_free(df->data);
    }

    if (df->var_names)
    {
        for (int32_t i = 0; i < df->n_vars; i++)
        {
            free(df->var_names[i]);
        }
        free(df->var_names);
    }

    free(df);
}

/*============================================================================
 * Data Extraction (Optimized)
 *============================================================================*/

double *pcmci_extract_lagged(const pcmci_dataframe_t *df, int32_t var,
                             int32_t tau, int32_t *out_len)
{
    if (!df || var < 0 || var >= df->n_vars || tau < 0 || tau > df->tau_max)
    {
        if (out_len)
            *out_len = 0;
        return NULL;
    }

    int32_t n_samples = df->T - df->tau_max;
    if (out_len)
        *out_len = n_samples;

    double *out = (double *)pcmci_malloc(n_samples * sizeof(double));
    if (!out)
        return NULL;

    /* OPTIMIZATION:
     * Since data layout is [Var][Time], the row is contiguous.
     * We need [tau_max - tau ... T]. This is a contiguous block.
     * memcpy is significantly faster than a loop here.
     */
    const double *row = df->data + (size_t)var * df->T;
    int32_t offset = df->tau_max - tau;

    memcpy(out, row + offset, n_samples * sizeof(double));

    return out;
}

double *pcmci_extract_cond_set(const pcmci_dataframe_t *df,
                               const pcmci_varlag_t *varlags,
                               int32_t n_cond,
                               int32_t *out_n_samples)
{
    if (!df)
    {
        if (out_n_samples)
            *out_n_samples = 0;
        return NULL;
    }

    int32_t n = df->T - df->tau_max;
    if (out_n_samples)
        *out_n_samples = n;

    if (n_cond == 0 || !varlags)
        return NULL;

    /* Allocate output: [n_samples x n_cond] */
    /* Note: Previous code implied Row-Major output here,
       but parcorr.c expects Column-Major for LAPACK performance.
       This function generates Column-Major output (all samples for cond 0, then cond 1...)
    */
    double *out = (double *)pcmci_malloc((size_t)n * n_cond * sizeof(double));
    if (!out)
        return NULL;

    for (int32_t j = 0; j < n_cond; j++)
    {
        int32_t var = varlags[j].var;
        int32_t tau = varlags[j].tau;

        if (var < 0 || var >= df->n_vars || tau < 0 || tau > df->tau_max)
        {
            pcmci_mkl_free(out);
            return NULL;
        }

        const double *row = df->data + (size_t)var * df->T;

        /* OPTIMIZATION:
         * Writing Column-Major: out[j*n ... j*n+n]
         * The source data is contiguous. The destination is contiguous.
         * Use memcpy.
         */
        double *dest_col = out + (size_t)j * n;
        int32_t offset = df->tau_max - tau;

        memcpy(dest_col, row + offset, n * sizeof(double));
    }

    return out;
}