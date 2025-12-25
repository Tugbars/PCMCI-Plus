/**
 * @file pcmci_gpdc.h
 * @brief Gaussian Process Distance Correlation (GPDC) test
 *
 * GPDC is a nonparametric conditional independence test that combines:
 * - Distance Correlation (dCor): Detects any type of dependence
 * - Gaussian Process Regression: Removes confounding effects
 *
 * For testing X ⊥ Y | Z:
 *   1. Fit GP: X ~ Z, get residuals ε_X
 *   2. Fit GP: Y ~ Z, get residuals ε_Y  
 *   3. Compute dCor(ε_X, ε_Y)
 *   4. Permutation test for p-value
 *
 * Reference:
 *   Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
 *   Measuring and testing dependence by correlation of distances.
 *   The Annals of Statistics, 35(6), 2769-2794.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PCMCI_GPDC_H
#define PCMCI_GPDC_H

#include "pcmci_types.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * API Export Macro
 *============================================================================*/

#ifndef PCMCI_API
#ifdef _WIN32
    #ifdef PCMCI_EXPORTS
        #define PCMCI_API __declspec(dllexport)
    #else
        #define PCMCI_API __declspec(dllimport)
    #endif
#else
    #define PCMCI_API
#endif
#endif

/*============================================================================
 * Configuration
 *============================================================================*/

/** GPDC test configuration */
typedef struct {
    int32_t n_perm;         /**< Number of permutations for p-value (default: 100) */
    double gp_lengthscale;  /**< GP RBF kernel lengthscale (0 = auto-select) */
    double gp_variance;     /**< GP signal variance (0 = auto-select) */
    double gp_noise;        /**< GP noise variance (default: 0.1) */
    int32_t seed;           /**< Random seed for permutations (0 = time-based) */
} pcmci_gpdc_config_t;

/** Get default GPDC configuration */
PCMCI_API pcmci_gpdc_config_t pcmci_gpdc_default_config(void);

/*============================================================================
 * GPDC Test Result
 *============================================================================*/

/** Result of GPDC test */
typedef struct {
    double dcor;            /**< Distance correlation value [0, 1] */
    double dcov;            /**< Distance covariance */
    double pvalue;          /**< P-value from permutation test */
    int32_t n_perm;         /**< Number of permutations used */
} pcmci_gpdc_result_t;

/*============================================================================
 * Distance Correlation Functions
 *============================================================================*/

/**
 * Compute distance correlation between X and Y
 *
 * Distance correlation is 0 iff X and Y are independent.
 * Unlike Pearson correlation, it detects nonlinear dependencies.
 *
 * @param X         First variable, array of length n
 * @param Y         Second variable, array of length n  
 * @param n         Number of samples
 * @return          Distance correlation in [0, 1]
 */
PCMCI_API double pcmci_dcor(const double* X, const double* Y, int32_t n);

/**
 * Compute distance covariance between X and Y
 *
 * @param X         First variable
 * @param Y         Second variable
 * @param n         Number of samples
 * @return          Distance covariance (>= 0)
 */
PCMCI_API double pcmci_dcov(const double* X, const double* Y, int32_t n);

/**
 * Compute distance variance of X
 *
 * @param X         Variable array
 * @param n         Number of samples
 * @return          Distance variance (>= 0)
 */
PCMCI_API double pcmci_dvar(const double* X, int32_t n);

/*============================================================================
 * Gaussian Process Regression
 *============================================================================*/

/**
 * Fit GP regression and return residuals
 *
 * Fits Y ~ GP(X) using RBF kernel and returns residuals ε = Y - μ(X)
 *
 * @param Y         Response variable [n]
 * @param X         Predictor variables [n x dim], column-major
 * @param n         Number of samples
 * @param dim       Dimension of X
 * @param lengthscale  RBF kernel lengthscale (0 = median heuristic)
 * @param variance     Signal variance (0 = auto from Y)
 * @param noise        Noise variance
 * @param residuals    Output: residuals [n] (pre-allocated)
 * @return          0 on success, -1 on error
 */
PCMCI_API int pcmci_gp_residuals(
    const double* Y,
    const double* X,
    int32_t n,
    int32_t dim,
    double lengthscale,
    double variance,
    double noise,
    double* residuals
);

/*============================================================================
 * Main GPDC Test Functions
 *============================================================================*/

/**
 * GPDC test for conditional independence: X ⊥ Y | Z
 *
 * Uses Gaussian Process regression to remove effect of Z,
 * then tests independence of residuals using distance correlation.
 *
 * @param X         First variable [n]
 * @param Y         Second variable [n]
 * @param Z         Conditioning variables [n x dim_z], column-major, or NULL
 * @param n         Number of samples
 * @param dim_z     Dimension of Z (0 if Z is NULL)
 * @param config    Configuration (NULL for defaults)
 * @return          GPDC result with dCor and p-value
 */
PCMCI_API pcmci_gpdc_result_t pcmci_gpdc_test(
    const double* X,
    const double* Y,
    const double* Z,
    int32_t n,
    int32_t dim_z,
    const pcmci_gpdc_config_t* config
);

/**
 * Compute GPDC value only (no p-value, faster)
 *
 * @param X         First variable
 * @param Y         Second variable
 * @param Z         Conditioning variables or NULL
 * @param n         Number of samples
 * @param dim_z     Dimension of Z
 * @param config    Configuration (NULL for defaults)
 * @return          Distance correlation of GP residuals
 */
PCMCI_API double pcmci_gpdc_value(
    const double* X,
    const double* Y,
    const double* Z,
    int32_t n,
    int32_t dim_z,
    const pcmci_gpdc_config_t* config
);

/**
 * Simple distance correlation test (no conditioning)
 *
 * Tests X ⊥ Y using distance correlation with permutation p-value.
 *
 * @param X         First variable
 * @param Y         Second variable
 * @param n         Number of samples
 * @param n_perm    Number of permutations (0 = no p-value)
 * @param seed      Random seed (0 = time-based)
 * @return          Result with dCor and p-value
 */
PCMCI_API pcmci_gpdc_result_t pcmci_dcor_test(
    const double* X,
    const double* Y,
    int32_t n,
    int32_t n_perm,
    int32_t seed
);

#ifdef __cplusplus
}
#endif

#endif /* PCMCI_GPDC_H */
