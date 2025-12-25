/**
 * @file gpdc.c
 * @brief Gaussian Process Distance Correlation (GPDC) implementation
 *
 * Implements:
 * - Distance correlation (Székely et al., 2007)
 * - Gaussian Process regression with RBF kernel
 * - GPDC conditional independence test
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#define _USE_MATH_DEFINES

#include "pcmci_gpdc.h"
#include "pcmci_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*============================================================================
 * BLAS/LAPACK - use LAPACKE/CBLAS interface (portable)
 *============================================================================*/

/* Headers already included via pcmci_internal.h */
/* MKL provides LAPACKE-compatible interface */

/*============================================================================
 * Configuration
 *============================================================================*/

pcmci_gpdc_config_t pcmci_gpdc_default_config(void)
{
    pcmci_gpdc_config_t config = {
        .n_perm = 100,
        .gp_lengthscale = 0.0,  /* Auto */
        .gp_variance = 0.0,     /* Auto */
        .gp_noise = 0.1,
        .seed = 0
    };
    return config;
}

/*============================================================================
 * Distance Matrix Computation
 *============================================================================*/

/**
 * Compute Euclidean distance matrix for 1D data
 * D[i,j] = |X[i] - X[j]|
 */
static void compute_dist_matrix_1d(const double* X, int32_t n, double* D)
{
    for (int32_t i = 0; i < n; i++) {
        D[i * n + i] = 0.0;
        for (int32_t j = i + 1; j < n; j++) {
            double d = fabs(X[i] - X[j]);
            D[i * n + j] = d;
            D[j * n + i] = d;
        }
    }
}

/**
 * Double-center a distance matrix
 * A[i,j] = D[i,j] - row_mean[i] - col_mean[j] + grand_mean
 */
static void double_center(double* D, int32_t n)
{
    /* Compute row means */
    double* row_mean = (double*)malloc(n * sizeof(double));
    if (!row_mean) return;
    
    double grand_mean = 0.0;
    
    for (int32_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (int32_t j = 0; j < n; j++) {
            sum += D[i * n + j];
        }
        row_mean[i] = sum / n;
        grand_mean += sum;
    }
    grand_mean /= (n * n);
    
    /* Double center */
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            D[i * n + j] = D[i * n + j] - row_mean[i] - row_mean[j] + grand_mean;
        }
    }
    
    free(row_mean);
}

/*============================================================================
 * Distance Correlation Implementation
 *============================================================================*/

double pcmci_dcov(const double* X, const double* Y, int32_t n)
{
    if (!X || !Y || n < 2) return 0.0;
    
    /* Allocate distance matrices */
    double* A = (double*)pcmci_malloc(n * n * sizeof(double));
    double* B = (double*)pcmci_malloc(n * n * sizeof(double));
    
    if (!A || !B) {
        pcmci_mkl_free(A);
        pcmci_mkl_free(B);
        return 0.0;
    }
    
    /* Compute distance matrices */
    compute_dist_matrix_1d(X, n, A);
    compute_dist_matrix_1d(Y, n, B);
    
    /* Double center */
    double_center(A, n);
    double_center(B, n);
    
    /* Compute distance covariance: dCov² = (1/n²) Σ A[i,j] * B[i,j] */
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            sum += A[i * n + j] * B[i * n + j];
        }
    }
    
    double dcov_sq = sum / (n * n);
    
    pcmci_mkl_free(A);
    pcmci_mkl_free(B);
    
    /* Return dCov (not squared) */
    return (dcov_sq > 0) ? sqrt(dcov_sq) : 0.0;
}

double pcmci_dvar(const double* X, int32_t n)
{
    if (!X || n < 2) return 0.0;
    
    /* Allocate distance matrix */
    double* A = (double*)pcmci_malloc(n * n * sizeof(double));
    if (!A) return 0.0;
    
    compute_dist_matrix_1d(X, n, A);
    double_center(A, n);
    
    /* dVar² = (1/n²) Σ A[i,j]² */
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            sum += A[i * n + j] * A[i * n + j];
        }
    }
    
    double dvar_sq = sum / (n * n);
    
    pcmci_mkl_free(A);
    
    return (dvar_sq > 0) ? sqrt(dvar_sq) : 0.0;
}

double pcmci_dcor(const double* X, const double* Y, int32_t n)
{
    if (!X || !Y || n < 2) return 0.0;
    
    /* Allocate distance matrices */
    double* A = (double*)pcmci_malloc(n * n * sizeof(double));
    double* B = (double*)pcmci_malloc(n * n * sizeof(double));
    
    if (!A || !B) {
        pcmci_mkl_free(A);
        pcmci_mkl_free(B);
        return 0.0;
    }
    
    /* Compute and double-center distance matrices */
    compute_dist_matrix_1d(X, n, A);
    compute_dist_matrix_1d(Y, n, B);
    double_center(A, n);
    double_center(B, n);
    
    /* Compute dCov², dVarX², dVarY² simultaneously */
    double sum_AB = 0.0, sum_AA = 0.0, sum_BB = 0.0;
    
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            double a = A[i * n + j];
            double b = B[i * n + j];
            sum_AB += a * b;
            sum_AA += a * a;
            sum_BB += b * b;
        }
    }
    
    pcmci_mkl_free(A);
    pcmci_mkl_free(B);
    
    double dcov_sq = sum_AB / (n * n);
    double dvarX_sq = sum_AA / (n * n);
    double dvarY_sq = sum_BB / (n * n);
    
    /* dCor = dCov / sqrt(dVarX * dVarY) */
    double denom = sqrt(dvarX_sq * dvarY_sq);
    
    if (denom < 1e-15) return 0.0;
    
    double dcor_sq = dcov_sq / denom;
    
    return (dcor_sq > 0) ? sqrt(dcor_sq) : 0.0;
}

/*============================================================================
 * Gaussian Process Regression
 *============================================================================*/

/**
 * Compute median of pairwise distances (for lengthscale heuristic)
 */
static double median_distance(const double* X, int32_t n, int32_t dim)
{
    int32_t n_pairs = n * (n - 1) / 2;
    double* dists = (double*)malloc(n_pairs * sizeof(double));
    if (!dists) return 1.0;
    
    int32_t k = 0;
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = i + 1; j < n; j++) {
            double sum_sq = 0.0;
            for (int32_t d = 0; d < dim; d++) {
                double diff = X[i + d * n] - X[j + d * n];
                sum_sq += diff * diff;
            }
            dists[k++] = sqrt(sum_sq);
        }
    }
    
    /* Simple selection for median (partial sort) */
    int32_t mid = n_pairs / 2;
    for (int32_t i = 0; i <= mid; i++) {
        int32_t min_idx = i;
        for (int32_t j = i + 1; j < n_pairs; j++) {
            if (dists[j] < dists[min_idx]) min_idx = j;
        }
        double tmp = dists[i];
        dists[i] = dists[min_idx];
        dists[min_idx] = tmp;
    }
    
    double median = dists[mid];
    free(dists);
    
    return (median > 1e-10) ? median : 1.0;
}

/**
 * Compute RBF kernel matrix: K[i,j] = variance * exp(-||x_i - x_j||² / (2 * lengthscale²))
 */
static void compute_rbf_kernel(const double* X, int32_t n, int32_t dim,
                                double lengthscale, double variance, double* K)
{
    double inv_2l2 = 1.0 / (2.0 * lengthscale * lengthscale);
    
    for (int32_t i = 0; i < n; i++) {
        K[i * n + i] = variance;
        for (int32_t j = i + 1; j < n; j++) {
            double sum_sq = 0.0;
            for (int32_t d = 0; d < dim; d++) {
                double diff = X[i + d * n] - X[j + d * n];
                sum_sq += diff * diff;
            }
            double k = variance * exp(-sum_sq * inv_2l2);
            K[i * n + j] = k;
            K[j * n + i] = k;
        }
    }
}

int pcmci_gp_residuals(const double* Y, const double* X, int32_t n, int32_t dim,
                        double lengthscale, double variance, double noise,
                        double* residuals)
{
    if (!Y || !X || !residuals || n < 2 || dim < 1) return -1;
    
    /* Auto-select hyperparameters if needed */
    if (lengthscale <= 0.0) {
        lengthscale = median_distance(X, n, dim);
    }
    if (variance <= 0.0) {
        /* Use variance of Y */
        double mean_y = 0.0;
        for (int32_t i = 0; i < n; i++) mean_y += Y[i];
        mean_y /= n;
        
        variance = 0.0;
        for (int32_t i = 0; i < n; i++) {
            double d = Y[i] - mean_y;
            variance += d * d;
        }
        variance /= (n - 1);
        if (variance < 1e-10) variance = 1.0;
    }
    
    /* Allocate kernel matrix K + σ²I */
    double* K = (double*)pcmci_malloc(n * n * sizeof(double));
    double* alpha = (double*)pcmci_malloc(n * sizeof(double));
    
    if (!K || !alpha) {
        pcmci_mkl_free(K);
        pcmci_mkl_free(alpha);
        return -1;
    }
    
    /* Compute K */
    compute_rbf_kernel(X, n, dim, lengthscale, variance, K);
    
    /* Add noise to diagonal: K + σ²I */
    for (int32_t i = 0; i < n; i++) {
        K[i * n + i] += noise;
    }
    
    /* Copy Y to alpha (will be overwritten by solution) */
    memcpy(alpha, Y, n * sizeof(double));
    
    /* Solve (K + σ²I) α = Y using Cholesky decomposition */
    /* Use LAPACKE interface which is more portable */
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, K, n);
    
    if (info != 0) {
        /* Cholesky failed - add more regularization and retry */
        compute_rbf_kernel(X, n, dim, lengthscale, variance, K);
        for (int32_t i = 0; i < n; i++) {
            K[i * n + i] += noise + 0.1;
        }
        memcpy(alpha, Y, n * sizeof(double));
        
        info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, K, n);
        
        if (info != 0) {
            /* Still failed - return Y as residuals (no GP fit) */
            memcpy(residuals, Y, n * sizeof(double));
            pcmci_mkl_free(K);
            pcmci_mkl_free(alpha);
            return -1;
        }
    }
    
    /* Solve L L^T x = b for x (alpha) */
    info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', n, 1, K, n, alpha, n);
    
    /* Now alpha = (K + σ²I)^(-1) Y */
    /* GP mean prediction at training points: μ = K_train * α */
    /* But K_train = K (without noise), so we need to recompute */
    
    /* Recompute K without noise for prediction */
    compute_rbf_kernel(X, n, dim, lengthscale, variance, K);
    
    /* μ = K * α using CBLAS */
    double* mu = residuals;  /* Use residuals buffer for mu first */
    
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, K, n, alpha, 1, 0.0, mu, 1);
    
    /* residuals = Y - μ */
    for (int32_t i = 0; i < n; i++) {
        residuals[i] = Y[i] - mu[i];
    }
    
    pcmci_mkl_free(K);
    pcmci_mkl_free(alpha);
    
    return 0;
}

/*============================================================================
 * GPDC Test Implementation
 *============================================================================*/

double pcmci_gpdc_value(const double* X, const double* Y, const double* Z,
                         int32_t n, int32_t dim_z, const pcmci_gpdc_config_t* config)
{
    if (!X || !Y || n < 4) return 0.0;
    
    pcmci_gpdc_config_t cfg = config ? *config : pcmci_gpdc_default_config();
    
    /* If no conditioning, just compute distance correlation */
    if (dim_z == 0 || !Z) {
        return pcmci_dcor(X, Y, n);
    }
    
    /* Allocate residual buffers */
    double* res_X = (double*)pcmci_malloc(n * sizeof(double));
    double* res_Y = (double*)pcmci_malloc(n * sizeof(double));
    
    if (!res_X || !res_Y) {
        pcmci_mkl_free(res_X);
        pcmci_mkl_free(res_Y);
        return 0.0;
    }
    
    /* Fit GP: X ~ Z, get residuals */
    int ret = pcmci_gp_residuals(X, Z, n, dim_z, 
                                  cfg.gp_lengthscale, cfg.gp_variance, cfg.gp_noise,
                                  res_X);
    if (ret != 0) {
        /* GP failed, use original X */
        memcpy(res_X, X, n * sizeof(double));
    }
    
    /* Fit GP: Y ~ Z, get residuals */
    ret = pcmci_gp_residuals(Y, Z, n, dim_z,
                              cfg.gp_lengthscale, cfg.gp_variance, cfg.gp_noise,
                              res_Y);
    if (ret != 0) {
        memcpy(res_Y, Y, n * sizeof(double));
    }
    
    /* Compute distance correlation of residuals */
    double dcor = pcmci_dcor(res_X, res_Y, n);
    
    pcmci_mkl_free(res_X);
    pcmci_mkl_free(res_Y);
    
    return dcor;
}

/* Fisher-Yates shuffle */
static void shuffle_array_gpdc(double* arr, int32_t n, uint64_t* rng_state)
{
    for (int32_t i = n - 1; i > 0; i--) {
        *rng_state ^= *rng_state << 13;
        *rng_state ^= *rng_state >> 7;
        *rng_state ^= *rng_state << 17;
        
        int32_t j = *rng_state % (i + 1);
        
        double tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

pcmci_gpdc_result_t pcmci_gpdc_test(const double* X, const double* Y, const double* Z,
                                     int32_t n, int32_t dim_z, const pcmci_gpdc_config_t* config)
{
    pcmci_gpdc_config_t cfg = config ? *config : pcmci_gpdc_default_config();
    
    pcmci_gpdc_result_t result = {0};
    result.n_perm = cfg.n_perm;
    
    /* Compute actual GPDC value */
    result.dcor = pcmci_gpdc_value(X, Y, Z, n, dim_z, &cfg);
    result.dcov = result.dcor;  /* Simplified: just use dcor */
    
    if (cfg.n_perm <= 0) {
        result.pvalue = -1.0;
        return result;
    }
    
    /* For GPDC, we need residuals for permutation test */
    double* res_X = (double*)pcmci_malloc(n * sizeof(double));
    double* res_Y = (double*)pcmci_malloc(n * sizeof(double));
    double* res_X_perm = (double*)pcmci_malloc(n * sizeof(double));
    
    if (!res_X || !res_Y || !res_X_perm) {
        pcmci_mkl_free(res_X);
        pcmci_mkl_free(res_Y);
        pcmci_mkl_free(res_X_perm);
        result.pvalue = -1.0;
        return result;
    }
    
    /* Get residuals (or original data if no Z) */
    if (dim_z > 0 && Z) {
        pcmci_gp_residuals(X, Z, n, dim_z, cfg.gp_lengthscale, cfg.gp_variance, cfg.gp_noise, res_X);
        pcmci_gp_residuals(Y, Z, n, dim_z, cfg.gp_lengthscale, cfg.gp_variance, cfg.gp_noise, res_Y);
    } else {
        memcpy(res_X, X, n * sizeof(double));
        memcpy(res_Y, Y, n * sizeof(double));
    }
    
    memcpy(res_X_perm, res_X, n * sizeof(double));
    
    uint64_t rng_state = cfg.seed ? (uint64_t)cfg.seed : (uint64_t)time(NULL);
    
    int32_t n_greater = 0;
    
    for (int32_t p = 0; p < cfg.n_perm; p++) {
        shuffle_array_gpdc(res_X_perm, n, &rng_state);
        
        double dcor_null = pcmci_dcor(res_X_perm, res_Y, n);
        
        if (dcor_null >= result.dcor) {
            n_greater++;
        }
    }
    
    result.pvalue = (double)(n_greater + 1) / (cfg.n_perm + 1);
    
    pcmci_mkl_free(res_X);
    pcmci_mkl_free(res_Y);
    pcmci_mkl_free(res_X_perm);
    
    return result;
}

pcmci_gpdc_result_t pcmci_dcor_test(const double* X, const double* Y, int32_t n,
                                     int32_t n_perm, int32_t seed)
{
    pcmci_gpdc_config_t config = pcmci_gpdc_default_config();
    config.n_perm = n_perm;
    config.seed = seed;
    
    return pcmci_gpdc_test(X, Y, NULL, n, 0, &config);
}

/*============================================================================
 * Integration with PCMCI+ (test wrapper matching parcorr interface)
 *============================================================================*/

/**
 * GPDC test wrapper compatible with PCMCI+ test interface
 */
pcmci_ci_result_t pcmci_gpdc_ci_test(const double* X, const double* Y,
                                      const double* Z, int32_t n, int32_t dim_z)
{
    pcmci_gpdc_config_t config = pcmci_gpdc_default_config();
    config.n_perm = 100;
    
    pcmci_gpdc_result_t gpdc_result = pcmci_gpdc_test(X, Y, Z, n, dim_z, &config);
    
    pcmci_ci_result_t result = {
        .val = gpdc_result.dcor,
        .pvalue = gpdc_result.pvalue,
        .stat = gpdc_result.dcor,
        .df = n - dim_z - 2
    };
    
    return result;
}
