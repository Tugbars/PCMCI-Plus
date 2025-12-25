/**
 * @file parcorr.c
 * @brief Partial correlation test implementation using BLAS/LAPACK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_internal.h"
#include <math.h>
#include <string.h>
#include <omp.h>

/*============================================================================
 * Statistical Utilities
 *============================================================================*/

/**
 * Regularized incomplete beta function I_x(a, b)
 * Using continued fraction representation (more stable than series)
 */
static double betacf(double a, double b, double x)
{
    const int MAXIT = 200;
    const double EPS = 3.0e-12;
    const double FPMIN = 1.0e-30;

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN)
        d = FPMIN;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= MAXIT; m++)
    {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN)
            d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN)
            d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < EPS)
            break;
    }
    return h;
}

/**
 * Regularized incomplete beta function I_x(a, b)
 */
static double betai(double a, double b, double x)
{
    if (x < 0.0 || x > 1.0)
        return 0.0;
    if (x == 0.0)
        return 0.0;
    if (x == 1.0)
        return 1.0;

    double bt;
    if (x == 0.0 || x == 1.0)
    {
        bt = 0.0;
    }
    else
    {
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) +
                 a * log(x) + b * log(1.0 - x));
    }

    if (x < (a + 1.0) / (a + b + 2.0))
    {
        return bt * betacf(a, b, x) / a;
    }
    else
    {
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
    }
}

/**
 * Student's t-distribution CDF
 */
static double t_cdf(double t, int df)
{
    double x = (double)df / (df + t * t);
    double prob = 0.5 * betai(0.5 * df, 0.5, x);
    return t > 0 ? 1.0 - prob : prob;
}

double pcmci_t_pvalue(double t, int32_t df)
{
    if (df < 1)
        df = 1;

    /* Two-tailed p-value */
    double cdf = t_cdf(fabs(t), df);
    double pval = 2.0 * (1.0 - cdf);

    return fmax(0.0, fmin(1.0, pval));
}

double pcmci_z_pvalue(double z)
{
    return erfc(fabs(z) / sqrt(2.0)); /* Two-tailed */
}

double pcmci_std(const double *x, int32_t n)
{
    if (n < 2)
        return 0.0;

    double mean = pcmci_mean(x, n);
    double sum_sq = 0.0;

#pragma omp simd reduction(+ : sum_sq)
    for (int32_t i = 0; i < n; i++)
    {
        double d = x[i] - mean;
        sum_sq += d * d;
    }

    return sqrt(sum_sq / (n - 1));
}

/*============================================================================
 * Residualization
 *============================================================================*/

int pcmci_residualize(const double *X, const double *Z,
                      int32_t n, int32_t k, double *resid)
{
    if (k == 0 || Z == NULL)
    {
        /* No conditioning - just demean */
        memcpy(resid, X, n * sizeof(double));
        pcmci_demean(resid, n);
        return 0;
    }

    /* Copy Z and X since LAPACK overwrites them
     * DGELS requires B to have max(m,n) = max(n,k) rows for the solution */
    int32_t b_rows = (n > k) ? n : k;
    double *Z_work = (double *)pcmci_malloc((size_t)n * k * sizeof(double));
    double *X_work = (double *)pcmci_malloc(b_rows * sizeof(double));

    if (!Z_work || !X_work)
    {
        pcmci_mkl_free(Z_work);
        pcmci_mkl_free(X_work);
        /* Fallback: return demeaned X */
        memcpy(resid, X, n * sizeof(double));
        pcmci_demean(resid, n);
        return -1;
    }

    /* Initialize X_work - copy X and zero-pad if k > n */
    memset(X_work, 0, b_rows * sizeof(double));
    memcpy(Z_work, Z, (size_t)n * k * sizeof(double));
    memcpy(X_work, X, n * sizeof(double));

    /* Solve min ||Z @ beta - X||^2 via QR decomposition
     * dgels: A is n x k, B is n x 1
     * On exit, B[0:k] contains solution beta
     */
    lapack_int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N',
                                    n, k, 1,
                                    Z_work, k,
                                    X_work, 1);

    if (info != 0)
    {
        pcmci_mkl_free(Z_work);
        pcmci_mkl_free(X_work);
        /* Fallback: return demeaned X */
        memcpy(resid, X, n * sizeof(double));
        pcmci_demean(resid, n);
        return -1;
    }

    /* Compute residuals: resid = X - Z @ beta */
    memcpy(resid, X, n * sizeof(double));

    /* resid = resid - Z @ beta (beta is in X_work[0:k]) */
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, k, -1.0, Z, k, X_work, 1, 1.0, resid, 1);

    pcmci_mkl_free(Z_work);
    pcmci_mkl_free(X_work);

    return 0;
}

/*============================================================================
 * Partial Correlation Test
 *============================================================================*/

pcmci_ci_result_t pcmci_parcorr_test(const double *X, const double *Y,
                                     const double *Z, int32_t n, int32_t k)
{
    pcmci_ci_result_t result = {0.0, 1.0, 0.0, 1};

    if (n < k + 3)
    {
        /* Not enough degrees of freedom */
        return result;
    }

    /* Allocate residual arrays */
    double *resid_x = (double *)pcmci_malloc(n * sizeof(double));
    double *resid_y = (double *)pcmci_malloc(n * sizeof(double));

    if (!resid_x || !resid_y)
    {
        pcmci_mkl_free(resid_x);
        pcmci_mkl_free(resid_y);
        return result;
    }

    /* Residualize X and Y on Z */
    pcmci_residualize(X, Z, n, k, resid_x);
    pcmci_residualize(Y, Z, n, k, resid_y);

    /* Compute partial correlation */
    double r = pcmci_corr_centered(resid_x, resid_y, n);

    /* Degrees of freedom: n - k - 2 */
    int32_t df = n - k - 2;
    if (df < 1)
        df = 1;

    /* t-statistic from correlation */
    double r_sq = r * r;
    double denom = 1.0 - r_sq;
    if (denom < 1e-15)
        denom = 1e-15; /* Clamp */

    double t = r * sqrt((double)df / denom);

    result.val = r;
    result.stat = t;
    result.df = df;

    /* Lazy p-value: skip expensive computation for weak correlations */
    double abs_t = fabs(t);
    if (abs_t < 1.5)
    {
        result.pvalue = 1.0;
    }
    else
    {
        result.pvalue = pcmci_t_pvalue(t, df);
    }

    pcmci_mkl_free(resid_x);
    pcmci_mkl_free(resid_y);

    return result;
}

/*============================================================================
 * Batch Partial Correlation (Optimized)
 *============================================================================*/

void pcmci_parcorr_batch(const double *X, const double *Y_batch,
                         const double *Z, int32_t n, int32_t k, int32_t m,
                         pcmci_ci_result_t *results)
{
    if (n < k + 3 || m == 0)
    {
        for (int32_t i = 0; i < m; i++)
        {
            results[i] = (pcmci_ci_result_t){0.0, 1.0, 0.0, 1};
        }
        return;
    }

    /* Residualize X once (shared across all Y tests) */
    double *resid_x = (double *)pcmci_malloc(n * sizeof(double));
    if (!resid_x)
    {
        for (int32_t i = 0; i < m; i++)
        {
            results[i] = (pcmci_ci_result_t){0.0, 1.0, 0.0, 1};
        }
        return;
    }

    pcmci_residualize(X, Z, n, k, resid_x);

    /* Precompute ||resid_x||^2 */
    double norm_x_sq = cblas_ddot(n, resid_x, 1, resid_x, 1);

    int32_t df = n - k - 2;
    if (df < 1)
        df = 1;

/* Process Y candidates in parallel */
#pragma omp parallel
    {
        /* Thread-local residual buffer */
        double *resid_y = (double *)pcmci_malloc(n * sizeof(double));

#pragma omp for schedule(dynamic, 4)
        for (int32_t i = 0; i < m; i++)
        {
            if (!resid_y)
            {
                results[i] = (pcmci_ci_result_t){0.0, 1.0, 0.0, 1};
                continue;
            }

            const double *Yi = Y_batch + (size_t)i * n;

            /* Residualize this Y on Z */
            pcmci_residualize(Yi, Z, n, k, resid_y);

            /* Compute correlation with resid_x */
            double dot_xy = cblas_ddot(n, resid_x, 1, resid_y, 1);
            double norm_y_sq = cblas_ddot(n, resid_y, 1, resid_y, 1);

            double denom = sqrt(norm_x_sq * norm_y_sq);
            double r = (denom > 1e-15) ? dot_xy / denom : 0.0;

            /* t-statistic */
            double r_sq = r * r;
            double t_denom = 1.0 - r_sq;
            if (t_denom < 1e-15)
                t_denom = 1e-15;
            double t = r * sqrt((double)df / t_denom);

            results[i].val = r;
            results[i].stat = t;
            results[i].df = df;

            /* Lazy p-value */
            double abs_t = fabs(t);
            if (abs_t < 1.5)
            {
                results[i].pvalue = 1.0;
            }
            else
            {
                results[i].pvalue = pcmci_t_pvalue(t, df);
            }
        }

        pcmci_mkl_free(resid_y);
    }

    pcmci_mkl_free(resid_x);
}

/*============================================================================
 * Combination Iterator
 *============================================================================*/

pcmci_comb_iter_t *pcmci_comb_init(int32_t n, int32_t k)
{
    pcmci_comb_iter_t *it = (pcmci_comb_iter_t *)calloc(1, sizeof(pcmci_comb_iter_t));
    if (!it)
        return NULL;

    it->n = n;
    it->k = k;
    it->started = false;

    if (k > n || k < 0 || n < 0)
    {
        it->done = true;
        it->indices = NULL;
    }
    else if (k == 0)
    {
        /* Special case: empty combination */
        it->done = false;
        it->indices = NULL;
    }
    else
    {
        it->done = false;
        it->indices = (int32_t *)malloc(k * sizeof(int32_t));
        if (!it->indices)
        {
            it->done = true;
        }
    }

    return it;
}

bool pcmci_comb_next(pcmci_comb_iter_t *it)
{
    if (!it || it->done)
        return false;

    if (it->k == 0)
    {
        /* Empty combination: return once */
        if (!it->started)
        {
            it->started = true;
            return true;
        }
        else
        {
            it->done = true;
            return false;
        }
    }

    if (!it->started)
    {
        /* Initialize to first combination: [0, 1, 2, ..., k-1] */
        for (int32_t i = 0; i < it->k; i++)
        {
            it->indices[i] = i;
        }
        it->started = true;
        return true;
    }

    /* Find rightmost element that can be incremented */
    int32_t i = it->k - 1;
    while (i >= 0 && it->indices[i] == it->n - it->k + i)
    {
        i--;
    }

    if (i < 0)
    {
        it->done = true;
        return false;
    }

    /* Increment and reset following elements */
    it->indices[i]++;
    for (int32_t j = i + 1; j < it->k; j++)
    {
        it->indices[j] = it->indices[j - 1] + 1;
    }

    return true;
}

void pcmci_comb_free(pcmci_comb_iter_t *it)
{
    if (!it)
        return;
    if (it->indices)
        free(it->indices);
    free(it);
}

/*============================================================================
 * Timing
 *============================================================================*/

double pcmci_get_time(void)
{
    return omp_get_wtime();
}

/*============================================================================
 * Workspace Management
 *============================================================================*/

pcmci_workspace_t *pcmci_workspace_create(int32_t n_samples, int32_t max_cond)
{
    pcmci_workspace_t *ws = (pcmci_workspace_t *)calloc(1, sizeof(pcmci_workspace_t));
    if (!ws)
        return NULL;

    ws->n_samples = n_samples;
    ws->max_cond = max_cond > 0 ? max_cond : 1;

    size_t z_size = (size_t)n_samples * ws->max_cond;
    size_t gram_size = (size_t)ws->max_cond * ws->max_cond;

    /* DGELS requires B array to have max(m,n) = max(n_samples, max_cond) rows */
    size_t x_work_size = (n_samples > ws->max_cond) ? n_samples : ws->max_cond;

    ws->X_buf = (double *)pcmci_malloc(n_samples * sizeof(double));
    ws->Y_buf = (double *)pcmci_malloc(n_samples * sizeof(double));
    ws->Z_buf = (double *)pcmci_malloc(z_size * sizeof(double));
    ws->resid_X = (double *)pcmci_malloc(n_samples * sizeof(double));
    ws->resid_Y = (double *)pcmci_malloc(n_samples * sizeof(double));
    ws->Z_work = (double *)pcmci_malloc(z_size * sizeof(double));
    ws->X_work = (double *)pcmci_malloc(x_work_size * sizeof(double));

    /* Cholesky fast-path buffers */
    ws->Gram = (double *)pcmci_malloc(gram_size * sizeof(double));
    ws->Beta = (double *)pcmci_malloc(ws->max_cond * sizeof(double));

    /* Query optimal LAPACK workspace size
     * Use LDB = max(m, n) = max(n_samples, max_cond) for DGELS */
    double work_query;
    int32_t lwork = -1;
    int32_t m = n_samples, n = ws->max_cond, nrhs = 1;
    int32_t ldb = (m > n) ? m : n;

    /* Use column-major layout for better LAPACK performance */
    LAPACKE_dgels_work(LAPACK_COL_MAJOR, 'N', m, n, nrhs,
                       NULL, m, NULL, ldb, &work_query, lwork);

    ws->lapack_lwork = (int32_t)work_query + 64;
    ws->lapack_work = (double *)pcmci_malloc(ws->lapack_lwork * sizeof(double));

    if (!ws->X_buf || !ws->Y_buf || !ws->Z_buf || !ws->resid_X ||
        !ws->resid_Y || !ws->Z_work || !ws->X_work || !ws->lapack_work ||
        !ws->Gram || !ws->Beta)
    {
        pcmci_workspace_free(ws);
        return NULL;
    }

    return ws;
}

void pcmci_workspace_free(pcmci_workspace_t *ws)
{
    if (!ws)
        return;
    pcmci_mkl_free(ws->X_buf);
    pcmci_mkl_free(ws->Y_buf);
    pcmci_mkl_free(ws->Z_buf);
    pcmci_mkl_free(ws->resid_X);
    pcmci_mkl_free(ws->resid_Y);
    pcmci_mkl_free(ws->Z_work);
    pcmci_mkl_free(ws->X_work);
    pcmci_mkl_free(ws->lapack_work);
    pcmci_mkl_free(ws->Gram);
    pcmci_mkl_free(ws->Beta);
    free(ws);
}

/*============================================================================
 * Zero-allocation data extraction
 *============================================================================*/

void pcmci_extract_lagged_into(const pcmci_dataframe_t *df,
                               int32_t var, int32_t tau, double *out)
{
    const double *row = df->data + (size_t)var * df->T;
    int32_t n = df->T - df->tau_max;
    int32_t offset = df->tau_max - tau;

    /* Data is contiguous in memory - memcpy is optimized for this */
    memcpy(out, row + offset, n * sizeof(double));
}

void pcmci_extract_cond_set_into(const pcmci_dataframe_t *df,
                                 const pcmci_varlag_t *varlags,
                                 int32_t n_cond, double *out)
{
    int32_t n = df->T - df->tau_max;

    /* Column-major layout for LAPACK: Z[i,j] = out[i + j*n] */
    for (int32_t j = 0; j < n_cond; j++)
    {
        const double *row = df->data + (size_t)varlags[j].var * df->T;
        double *col = out + (size_t)j * n;
        int32_t offset = df->tau_max - varlags[j].tau;

        /* Each column is contiguous - use memcpy */
        memcpy(col, row + offset, n * sizeof(double));
    }
}

/*============================================================================
 * Zero-allocation partial correlation
 *============================================================================*/

void pcmci_residualize_ws(const double *X, const double *Z,
                          int32_t n, int32_t k,
                          pcmci_workspace_t *ws, double *resid)
{
    if (k == 0 || Z == NULL)
    {
        memcpy(resid, X, n * sizeof(double));
        pcmci_demean(resid, n);
        return;
    }

    /*
     * Cholesky Fast Path:
     * Solve normal equations: (Z'Z) beta = Z'X
     *
     * 1. G = Z'Z (k×k Gram matrix, fits in cache)
     * 2. v = Z'X (k×1 vector)
     * 3. Cholesky: G = L L'
     * 4. Solve: L L' beta = v
     * 5. resid = X - Z beta
     *
     * Complexity: O(nk² + k³/3) vs QR's O(2nk²)
     * For typical PCMCI (n=500, k=5): ~2-4x faster
     */

    /* Step 1: Compute Gram matrix G = Z' * Z (k × k, upper triangle) */
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                k, n, 1.0, Z, n, 0.0, ws->Gram, k);

    /* Step 2: Compute v = Z' * X (k × 1) */
    cblas_dgemv(CblasColMajor, CblasTrans,
                n, k, 1.0, Z, n, X, 1, 0.0, ws->Beta, 1);

    /* Step 3: Cholesky factorization G = U' U (upper triangular) */
    lapack_int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', k, ws->Gram, k);

    if (info == 0)
    {
        /* ===== FAST PATH: Cholesky succeeded ===== */

        /* Step 4: Solve G * beta = v using Cholesky factors */
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', k, 1, ws->Gram, k, ws->Beta, k);

        /* Step 5: resid = X - Z * beta */
        memcpy(resid, X, n * sizeof(double));
        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    n, k, -1.0, Z, n, ws->Beta, 1, 1.0, resid, 1);
    }
    else
    {
        /* ===== SLOW PATH: Z is rank-deficient, fallback to QR ===== */

        /* Copy inputs (dgels destroys them) */
        memcpy(ws->Z_work, Z, (size_t)n * k * sizeof(double));
        memcpy(ws->X_work, X, n * sizeof(double));

        /* Zero-pad X_work if k > n (underdetermined system) */
        if (k > n)
        {
            memset(ws->X_work + n, 0, (k - n) * sizeof(double));
        }

        /* Solve min||Z @ beta - X||_2 via QR
         * DGELS requires LDB >= max(m, n) = max(n, k) */
        int32_t ldb = (n > k) ? n : k;
        info = LAPACKE_dgels_work(LAPACK_COL_MAJOR, 'N', n, k, 1,
                                  ws->Z_work, n, ws->X_work, ldb,
                                  ws->lapack_work, ws->lapack_lwork);

        if (info != 0)
        {
            /* Both Cholesky and QR failed - just demean */
            memcpy(resid, X, n * sizeof(double));
            pcmci_demean(resid, n);
            return;
        }

        /* resid = X - Z @ beta */
        memcpy(resid, X, n * sizeof(double));
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, k,
                    -1.0, Z, n, ws->X_work, 1, 1.0, resid, 1);
    }
}

pcmci_ci_result_t pcmci_parcorr_ws(const double *X, const double *Y,
                                   const double *Z, int32_t n, int32_t k,
                                   pcmci_workspace_t *ws)
{
    pcmci_ci_result_t result = {0.0, 1.0, 0.0, 1};

    if (n < k + 3)
        return result;

    /* Residualize X and Y */
    pcmci_residualize_ws(X, Z, n, k, ws, ws->resid_X);
    pcmci_residualize_ws(Y, Z, n, k, ws, ws->resid_Y);

    /* Compute correlation of residuals */
    double dot_xy = cblas_ddot(n, ws->resid_X, 1, ws->resid_Y, 1);
    double dot_xx = cblas_ddot(n, ws->resid_X, 1, ws->resid_X, 1);
    double dot_yy = cblas_ddot(n, ws->resid_Y, 1, ws->resid_Y, 1);

    double denom = sqrt(dot_xx * dot_yy);
    double r = (denom > 1e-15) ? dot_xy / denom : 0.0;

    if (r > 1.0)
        r = 1.0;
    if (r < -1.0)
        r = -1.0;

    int32_t df = n - k - 2;
    if (df < 1)
        df = 1;

    /* t-statistic */
    double r_sq = r * r;
    double t_denom = 1.0 - r_sq;
    if (t_denom < 1e-15)
        t_denom = 1e-15;
    double t = r * sqrt((double)df / t_denom);

    result.val = r;
    result.stat = t;
    result.df = df;

    /*
     * Lazy p-value: skip expensive computation for obviously non-significant results.
     *
     * For |t| < 1.0: p-value > 0.3 for any df (definitely not significant)
     * For |t| < 1.5: p-value > 0.13 for any df (still not significant at α=0.05)
     *
     * This saves ~30-50% of lgamma/betai calls in skeleton phase where most
     * links are weak and get pruned anyway.
     */
    double abs_t = fabs(t);
    if (abs_t < 1.5)
    {
        result.pvalue = 1.0; /* Definitely not significant, skip expensive math */
    }
    else
    {
        result.pvalue = pcmci_t_pvalue(t, df);
    }

    return result;
}

/*============================================================================
 * Robust Transformations
 *============================================================================*/

/* Comparison function for argsort */
typedef struct
{
    double val;
    int32_t idx;
} rank_pair_t;

static int rank_cmp(const void *a, const void *b)
{
    double diff = ((rank_pair_t *)a)->val - ((rank_pair_t *)b)->val;
    if (diff < 0)
        return -1;
    if (diff > 0)
        return 1;
    return 0;
}

void pcmci_rank_into(const double *x, int32_t n, double *out)
{
    if (n <= 0)
        return;

    /* Sort with original indices */
    rank_pair_t *pairs = (rank_pair_t *)malloc(n * sizeof(rank_pair_t));
    if (!pairs)
    {
        /* Fallback: just copy */
        memcpy(out, x, n * sizeof(double));
        return;
    }

    for (int32_t i = 0; i < n; i++)
    {
        pairs[i].val = x[i];
        pairs[i].idx = i;
    }

    qsort(pairs, n, sizeof(rank_pair_t), rank_cmp);

    /* Assign ranks, handling ties with average rank */
    int32_t i = 0;
    while (i < n)
    {
        int32_t j = i;
        /* Find all tied values */
        while (j < n - 1 && pairs[j + 1].val == pairs[i].val)
        {
            j++;
        }

        /* Average rank for ties (1-based ranks, then convert to 0-based) */
        double avg_rank = (i + j) / 2.0;

        for (int32_t k = i; k <= j; k++)
        {
            out[pairs[k].idx] = avg_rank;
        }

        i = j + 1;
    }

    free(pairs);
}

static int double_cmp(const void *a, const void *b)
{
    double diff = *(double *)a - *(double *)b;
    if (diff < 0)
        return -1;
    if (diff > 0)
        return 1;
    return 0;
}

void pcmci_winsorize_into(const double *x, int32_t n, double thresh, double *out)
{
    if (n <= 0 || thresh <= 0.0 || thresh >= 0.5)
    {
        /* No winsorization, just copy */
        memcpy(out, x, n * sizeof(double));
        return;
    }

    /* Copy and sort to find percentiles */
    double *sorted = (double *)malloc(n * sizeof(double));
    if (!sorted)
    {
        memcpy(out, x, n * sizeof(double));
        return;
    }

    memcpy(sorted, x, n * sizeof(double));
    qsort(sorted, n, sizeof(double), double_cmp);

    /* Find percentile bounds */
    int32_t low_idx = (int32_t)(thresh * n);
    int32_t high_idx = n - 1 - low_idx;

    if (low_idx < 0)
        low_idx = 0;
    if (high_idx >= n)
        high_idx = n - 1;
    if (low_idx >= high_idx)
    {
        /* Edge case: not enough data */
        memcpy(out, x, n * sizeof(double));
        free(sorted);
        return;
    }

    double low_val = sorted[low_idx];
    double high_val = sorted[high_idx];

    free(sorted);

    /* Clip values */
    for (int32_t i = 0; i < n; i++)
    {
        if (x[i] < low_val)
        {
            out[i] = low_val;
        }
        else if (x[i] > high_val)
        {
            out[i] = high_val;
        }
        else
        {
            out[i] = x[i];
        }
    }
}

pcmci_dataframe_t *pcmci_dataframe_to_ranks(const pcmci_dataframe_t *df)
{
    if (!df)
        return NULL;

    /* Allocate new data buffer */
    size_t data_size = (size_t)df->n_vars * df->T;
    double *ranked_data = (double *)pcmci_malloc(data_size * sizeof(double));
    if (!ranked_data)
        return NULL;

    /* Rank each variable independently */
    for (int32_t i = 0; i < df->n_vars; i++)
    {
        const double *row_in = df->data + (size_t)i * df->T;
        double *row_out = ranked_data + (size_t)i * df->T;
        pcmci_rank_into(row_in, df->T, row_out);
    }

    /* Create new dataframe */
    pcmci_dataframe_t *result = (pcmci_dataframe_t *)calloc(1, sizeof(pcmci_dataframe_t));
    if (!result)
    {
        pcmci_mkl_free(ranked_data);
        return NULL;
    }

    result->data = ranked_data;
    result->n_vars = df->n_vars;
    result->T = df->T;
    result->tau_max = df->tau_max;
    result->owns_data = true;
    result->var_names = NULL; /* Don't copy names - user can set if needed */

    return result;
}

pcmci_dataframe_t *pcmci_dataframe_winsorize(const pcmci_dataframe_t *df, double thresh)
{
    if (!df || thresh <= 0.0)
        return NULL;

    /* Allocate new data buffer */
    size_t data_size = (size_t)df->n_vars * df->T;
    double *winsorized_data = (double *)pcmci_malloc(data_size * sizeof(double));
    if (!winsorized_data)
        return NULL;

    /* Winsorize each variable independently */
    for (int32_t i = 0; i < df->n_vars; i++)
    {
        const double *row_in = df->data + (size_t)i * df->T;
        double *row_out = winsorized_data + (size_t)i * df->T;
        pcmci_winsorize_into(row_in, df->T, thresh, row_out);
    }

    /* Create new dataframe */
    pcmci_dataframe_t *result = (pcmci_dataframe_t *)calloc(1, sizeof(pcmci_dataframe_t));
    if (!result)
    {
        pcmci_mkl_free(winsorized_data);
        return NULL;
    }

    result->data = winsorized_data;
    result->n_vars = df->n_vars;
    result->T = df->T;
    result->tau_max = df->tau_max;
    result->owns_data = true;
    result->var_names = NULL;

    return result;
}