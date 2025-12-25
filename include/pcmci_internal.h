/**
 * @file pcmci_internal.h
 * @brief Internal functions and helpers for PCMCI+
 * 
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PCMCI_INTERNAL_H
#define PCMCI_INTERNAL_H

#include "pcmci_types.h"

/* BLAS/LAPACK backend selection */
#ifdef PCMCI_USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Memory Alignment
 *============================================================================*/

#define PCMCI_ALIGNMENT 64  /* AVX-512 alignment */

/* Aligned allocation - works with both MKL and standard C */
static inline void* pcmci_malloc(size_t size) {
#ifdef PCMCI_USE_MKL
    return mkl_malloc(size, PCMCI_ALIGNMENT);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, PCMCI_ALIGNMENT, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static inline void* pcmci_calloc(size_t n, size_t size) {
#ifdef PCMCI_USE_MKL
    return mkl_calloc(n, size, PCMCI_ALIGNMENT);
#else
    size_t total = n * size;
    void* ptr = pcmci_malloc(total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
#endif
}

static inline void pcmci_mkl_free(void* ptr) {
    if (ptr) {
#ifdef PCMCI_USE_MKL
        mkl_free(ptr);
#else
        free(ptr);
#endif
    }
}

/*============================================================================
 * Graph Index Helpers
 *============================================================================*/

/* Flat index into 3D graph arrays: [n_vars x (tau_max+1) x n_vars] */
static inline int64_t pcmci_graph_idx(int32_t n_vars, int32_t tau_max,
                                       int32_t i, int32_t tau, int32_t j) {
    return (int64_t)i * (tau_max + 1) * n_vars + (int64_t)tau * n_vars + j;
}

/*============================================================================
 * Statistical Functions
 *============================================================================*/

/**
 * Fisher z-transform: z = 0.5 * ln((1+r)/(1-r)) = atanh(r)
 */
static inline double pcmci_fisher_z(double r) {
    /* Clamp to avoid infinity */
    if (r >= 0.9999) return 4.0;
    if (r <= -0.9999) return -4.0;
    return 0.5 * log((1.0 + r) / (1.0 - r));
}

/**
 * Inverse Fisher z-transform: r = tanh(z)
 */
static inline double pcmci_fisher_z_inv(double z) {
    return tanh(z);
}

/**
 * Compute p-value from t-statistic (two-tailed)
 * Uses MKL VSL for accurate CDF
 */
double pcmci_t_pvalue(double t, int32_t df);

/**
 * Compute p-value from z-statistic (two-tailed, normal)
 */
double pcmci_z_pvalue(double z);

/**
 * Mean of array
 */
static inline double pcmci_mean(const double* x, int32_t n) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int32_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum / n;
}

/**
 * Standard deviation (sample, ddof=1)
 */
double pcmci_std(const double* x, int32_t n);

/**
 * Demean array in place
 */
static inline void pcmci_demean(double* x, int32_t n) {
    double m = pcmci_mean(x, n);
    #pragma omp simd
    for (int32_t i = 0; i < n; i++) {
        x[i] -= m;
    }
}

/*============================================================================
 * Linear Algebra Helpers
 *============================================================================*/

/**
 * Residualize X on Z: resid = X - Z @ inv(Z'Z) @ Z' @ X
 * Uses MKL LAPACK for QR-based least squares
 * 
 * @param X         Input array, length n
 * @param Z         Conditioning matrix [n x k], row-major
 * @param n         Number of samples
 * @param k         Number of conditioning variables
 * @param resid     Output residuals, length n (pre-allocated)
 * @return          0 on success, -1 on failure
 */
int pcmci_residualize(const double* X, const double* Z, 
                       int32_t n, int32_t k, double* resid);

/**
 * Pearson correlation of two zero-mean arrays
 */
static inline double pcmci_corr_centered(const double* x, const double* y, int32_t n) {
    double dot_xy = cblas_ddot(n, x, 1, y, 1);
    double dot_xx = cblas_ddot(n, x, 1, x, 1);
    double dot_yy = cblas_ddot(n, y, 1, y, 1);
    
    double denom = sqrt(dot_xx * dot_yy);
    if (denom < 1e-15) return 0.0;
    
    return dot_xy / denom;
}

/*============================================================================
 * Combination Iterator
 *============================================================================*/

/**
 * Iterator for k-combinations of n elements
 */
typedef struct {
    int32_t* indices;   /**< Current combination */
    int32_t n;          /**< Total elements */
    int32_t k;          /**< Combination size */
    bool done;          /**< Iteration complete */
    bool started;       /**< Has been advanced at least once */
} pcmci_comb_iter_t;

/**
 * Initialize combination iterator
 * Returns iterator in "before first" state - call _next to get first combo
 */
pcmci_comb_iter_t* pcmci_comb_init(int32_t n, int32_t k);

/**
 * Advance to next combination
 * @return true if valid combination available, false if done
 */
bool pcmci_comb_next(pcmci_comb_iter_t* it);

/**
 * Free combination iterator
 */
void pcmci_comb_free(pcmci_comb_iter_t* it);

/*============================================================================
 * Timing
 *============================================================================*/

/**
 * Get current time in seconds (high resolution)
 */
double pcmci_get_time(void);

/*============================================================================
 * Sorting for FDR
 *============================================================================*/

/**
 * Argsort: return indices that would sort array in ascending order
 */
int32_t* pcmci_argsort(const double* arr, int32_t n);

#ifdef __cplusplus
}
#endif

#endif /* PCMCI_INTERNAL_H */
