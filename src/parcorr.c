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
static double betacf(double a, double b, double x) {
    const int MAXIT = 200;
    const double EPS = 3.0e-12;
    const double FPMIN = 1.0e-30;
    
    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    
    for (int m = 1; m <= MAXIT; m++) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < EPS) break;
    }
    return h;
}

/**
 * Regularized incomplete beta function I_x(a, b)
 */
static double betai(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return 0.0;
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    
    double bt;
    if (x == 0.0 || x == 1.0) {
        bt = 0.0;
    } else {
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + 
                 a * log(x) + b * log(1.0 - x));
    }
    
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(a, b, x) / a;
    } else {
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
    }
}

/**
 * Student's t-distribution CDF
 */
static double t_cdf(double t, int df) {
    double x = (double)df / (df + t * t);
    double prob = 0.5 * betai(0.5 * df, 0.5, x);
    return t > 0 ? 1.0 - prob : prob;
}

double pcmci_t_pvalue(double t, int32_t df) {
    if (df < 1) df = 1;
    
    /* Two-tailed p-value */
    double cdf = t_cdf(fabs(t), df);
    double pval = 2.0 * (1.0 - cdf);
    
    return fmax(0.0, fmin(1.0, pval));
}

double pcmci_z_pvalue(double z) {
    return erfc(fabs(z) / sqrt(2.0));  /* Two-tailed */
}

double pcmci_std(const double* x, int32_t n) {
    if (n < 2) return 0.0;
    
    double mean = pcmci_mean(x, n);
    double sum_sq = 0.0;
    
    #pragma omp simd reduction(+:sum_sq)
    for (int32_t i = 0; i < n; i++) {
        double d = x[i] - mean;
        sum_sq += d * d;
    }
    
    return sqrt(sum_sq / (n - 1));
}

/*============================================================================
 * Residualization
 *============================================================================*/

int pcmci_residualize(const double* X, const double* Z,
                       int32_t n, int32_t k, double* resid) {
    if (k == 0 || Z == NULL) {
        /* No conditioning - just demean */
        memcpy(resid, X, n * sizeof(double));
        pcmci_demean(resid, n);
        return 0;
    }
    
    /* Copy Z and X since LAPACK overwrites them */
    double* Z_work = (double*)pcmci_malloc((size_t)n * k * sizeof(double));
    double* X_work = (double*)pcmci_malloc(n * sizeof(double));
    
    if (!Z_work || !X_work) {
        pcmci_mkl_free(Z_work);
        pcmci_mkl_free(X_work);
        /* Fallback: return demeaned X */
        memcpy(resid, X, n * sizeof(double));
        pcmci_demean(resid, n);
        return -1;
    }
    
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
    
    if (info != 0) {
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

pcmci_ci_result_t pcmci_parcorr_test(const double* X, const double* Y,
                                      const double* Z, int32_t n, int32_t k) {
    pcmci_ci_result_t result = {0.0, 1.0, 0.0, 1};
    
    if (n < k + 3) {
        /* Not enough degrees of freedom */
        return result;
    }
    
    /* Allocate residual arrays */
    double* resid_x = (double*)pcmci_malloc(n * sizeof(double));
    double* resid_y = (double*)pcmci_malloc(n * sizeof(double));
    
    if (!resid_x || !resid_y) {
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
    if (df < 1) df = 1;
    
    /* t-statistic from correlation */
    double r_sq = r * r;
    double denom = 1.0 - r_sq;
    if (denom < 1e-15) denom = 1e-15;  /* Clamp */
    
    double t = r * sqrt((double)df / denom);
    
    result.val = r;
    result.stat = t;
    result.df = df;
    result.pvalue = pcmci_t_pvalue(t, df);
    
    pcmci_mkl_free(resid_x);
    pcmci_mkl_free(resid_y);
    
    return result;
}

/*============================================================================
 * Batch Partial Correlation (Optimized)
 *============================================================================*/

void pcmci_parcorr_batch(const double* X, const double* Y_batch,
                          const double* Z, int32_t n, int32_t k, int32_t m,
                          pcmci_ci_result_t* results) {
    if (n < k + 3 || m == 0) {
        for (int32_t i = 0; i < m; i++) {
            results[i] = (pcmci_ci_result_t){0.0, 1.0, 0.0, 1};
        }
        return;
    }
    
    /* Residualize X once (shared across all Y tests) */
    double* resid_x = (double*)pcmci_malloc(n * sizeof(double));
    if (!resid_x) {
        for (int32_t i = 0; i < m; i++) {
            results[i] = (pcmci_ci_result_t){0.0, 1.0, 0.0, 1};
        }
        return;
    }
    
    pcmci_residualize(X, Z, n, k, resid_x);
    
    /* Precompute ||resid_x||^2 */
    double norm_x_sq = cblas_ddot(n, resid_x, 1, resid_x, 1);
    
    int32_t df = n - k - 2;
    if (df < 1) df = 1;
    
    /* Process Y candidates in parallel */
    #pragma omp parallel
    {
        /* Thread-local residual buffer */
        double* resid_y = (double*)pcmci_malloc(n * sizeof(double));
        
        #pragma omp for schedule(dynamic, 4)
        for (int32_t i = 0; i < m; i++) {
            if (!resid_y) {
                results[i] = (pcmci_ci_result_t){0.0, 1.0, 0.0, 1};
                continue;
            }
            
            const double* Yi = Y_batch + (size_t)i * n;
            
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
            if (t_denom < 1e-15) t_denom = 1e-15;
            double t = r * sqrt((double)df / t_denom);
            
            results[i].val = r;
            results[i].stat = t;
            results[i].df = df;
            results[i].pvalue = pcmci_t_pvalue(t, df);
        }
        
        pcmci_mkl_free(resid_y);
    }
    
    pcmci_mkl_free(resid_x);
}

/*============================================================================
 * Combination Iterator
 *============================================================================*/

pcmci_comb_iter_t* pcmci_comb_init(int32_t n, int32_t k) {
    pcmci_comb_iter_t* it = (pcmci_comb_iter_t*)calloc(1, sizeof(pcmci_comb_iter_t));
    if (!it) return NULL;
    
    it->n = n;
    it->k = k;
    it->started = false;
    
    if (k > n || k < 0 || n < 0) {
        it->done = true;
        it->indices = NULL;
    } else if (k == 0) {
        /* Special case: empty combination */
        it->done = false;
        it->indices = NULL;
    } else {
        it->done = false;
        it->indices = (int32_t*)malloc(k * sizeof(int32_t));
        if (!it->indices) {
            it->done = true;
        }
    }
    
    return it;
}

bool pcmci_comb_next(pcmci_comb_iter_t* it) {
    if (!it || it->done) return false;
    
    if (it->k == 0) {
        /* Empty combination: return once */
        if (!it->started) {
            it->started = true;
            return true;
        } else {
            it->done = true;
            return false;
        }
    }
    
    if (!it->started) {
        /* Initialize to first combination: [0, 1, 2, ..., k-1] */
        for (int32_t i = 0; i < it->k; i++) {
            it->indices[i] = i;
        }
        it->started = true;
        return true;
    }
    
    /* Find rightmost element that can be incremented */
    int32_t i = it->k - 1;
    while (i >= 0 && it->indices[i] == it->n - it->k + i) {
        i--;
    }
    
    if (i < 0) {
        it->done = true;
        return false;
    }
    
    /* Increment and reset following elements */
    it->indices[i]++;
    for (int32_t j = i + 1; j < it->k; j++) {
        it->indices[j] = it->indices[j - 1] + 1;
    }
    
    return true;
}

void pcmci_comb_free(pcmci_comb_iter_t* it) {
    if (!it) return;
    if (it->indices) free(it->indices);
    free(it);
}

/*============================================================================
 * Timing
 *============================================================================*/

double pcmci_get_time(void) {
    return omp_get_wtime();
}
