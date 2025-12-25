/**
 * @file dataframe.c
 * @brief Time series dataframe implementation
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

    memcpy(df->data, data, (size_t)n_vars * T * sizeof(double));
    return df;
}

pcmci_dataframe_t *pcmci_dataframe_alloc(int32_t n_vars, int32_t T, int32_t tau_max)
{
    pcmci_dataframe_t *df = (pcmci_dataframe_t *)calloc(1, sizeof(pcmci_dataframe_t));
    if (!df)
        return NULL;

    df->data = (double *)pcmci_malloc((size_t)n_vars * T * sizeof(double));
    if (!df->data)
    {
        free(df);
        return NULL;
    }

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

    const double *row = df->data + (size_t)var * df->T;

/* Extract X_var(t - tau) for t = tau_max, tau_max+1, ..., T-1 */
#pragma omp simd
    for (int32_t t = 0; t < n_samples; t++)
    {
        int32_t src_t = df->tau_max + t - tau;
        out[t] = row[src_t];
    }

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
    {
        return NULL;
    }

    /* Row-major: [n_samples x n_cond] */
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

#pragma omp simd
        for (int32_t t = 0; t < n; t++)
        {
            out[t * n_cond + j] = row[df->tau_max + t - tau];
        }
    }

    return out;
}