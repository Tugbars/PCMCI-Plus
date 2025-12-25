/**
 * @file pcmci_cmi.h
 * @brief Conditional Mutual Information (CMI) test using k-NN KSG estimator
 *
 * Implements the Kraskov-Stögbauer-Grassberger (KSG) estimator for CMI,
 * which can detect nonlinear dependencies between variables.
 *
 * Reference:
 *   Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
 *   Estimating mutual information. Physical Review E, 69(6), 066138.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PCMCI_CMI_H
#define PCMCI_CMI_H

#include "pcmci_types.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
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

    /** CMI test configuration */
    typedef struct
    {
        int32_t k;          /**< Number of nearest neighbors (default: 5) */
        int32_t n_perm;     /**< Number of permutations for p-value (default: 100) */
        int32_t n_threads;  /**< Number of threads (0 = auto) */
        bool use_chebyshev; /**< Use Chebyshev (max) norm instead of Euclidean */
        int32_t seed;       /**< Random seed for permutations (0 = time-based) */
    } pcmci_cmi_config_t;

    /** Get default CMI configuration */
    PCMCI_API pcmci_cmi_config_t pcmci_cmi_default_config(void);

    /*============================================================================
     * CMI Test Result
     *============================================================================*/

    /** Result of CMI test */
    typedef struct
    {
        double cmi;     /**< Conditional mutual information value */
        double pvalue;  /**< P-value from permutation test */
        double stat;    /**< Test statistic (same as cmi) */
        int32_t k;      /**< k used for estimation */
        int32_t n_perm; /**< Number of permutations used */
    } pcmci_cmi_result_t;

    /*============================================================================
     * Main CMI Functions
     *============================================================================*/

    /**
     * Compute CMI(X; Y | Z) using the KSG estimator
     *
     * Estimates conditional mutual information between X and Y given Z.
     * Uses k-nearest neighbors in the joint space to estimate entropies.
     *
     * CMI(X; Y | Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
     *
     * For the unconditional case (Z = NULL), computes MI(X; Y).
     *
     * @param X         First variable, array of length n
     * @param Y         Second variable, array of length n
     * @param Z         Conditioning variables, column-major [n x dim_z], or NULL
     * @param n         Number of samples
     * @param dim_z     Dimension of Z (0 if Z is NULL)
     * @param config    Configuration (use pcmci_cmi_default_config() for defaults)
     * @return          CMI result with value and p-value
     */
    PCMCI_API pcmci_cmi_result_t pcmci_cmi_test(
        const double *X,
        const double *Y,
        const double *Z,
        int32_t n,
        int32_t dim_z,
        const pcmci_cmi_config_t *config);

    /**
     * Compute CMI value only (no p-value, faster)
     *
     * @param X         First variable
     * @param Y         Second variable
     * @param Z         Conditioning variables or NULL
     * @param n         Number of samples
     * @param dim_z     Dimension of Z
     * @param k         Number of nearest neighbors
     * @return          CMI estimate
     */
    PCMCI_API double pcmci_cmi_value(
        const double *X,
        const double *Y,
        const double *Z,
        int32_t n,
        int32_t dim_z,
        int32_t k);

    /**
     * Compute mutual information MI(X; Y)
     *
     * Convenience function for unconditional MI.
     *
     * @param X         First variable
     * @param Y         Second variable
     * @param n         Number of samples
     * @param k         Number of nearest neighbors
     * @return          MI estimate
     */
    PCMCI_API double pcmci_mi_value(
        const double *X,
        const double *Y,
        int32_t n,
        int32_t k);

    /*============================================================================
     * KD-Tree for Fast k-NN Search
     *============================================================================*/

    /** Opaque KD-tree structure */
    typedef struct pcmci_kdtree pcmci_kdtree_t;

    /**
     * Build KD-tree for k-NN queries
     *
     * @param data      Column-major data matrix [n x dim]
     * @param n         Number of points
     * @param dim       Dimensionality
     * @return          KD-tree (call pcmci_kdtree_free when done)
     */
    PCMCI_API pcmci_kdtree_t *pcmci_kdtree_build(
        const double *data,
        int32_t n,
        int32_t dim);

    /**
     * Find k nearest neighbors
     *
     * @param tree      KD-tree
     * @param query     Query point [dim]
     * @param k         Number of neighbors
     * @param indices   Output: neighbor indices [k]
     * @param distances Output: neighbor distances [k]
     */
    PCMCI_API void pcmci_kdtree_knn(
        const pcmci_kdtree_t *tree,
        const double *query,
        int32_t k,
        int32_t *indices,
        double *distances);

    /**
     * Count points within radius (Chebyshev/max norm)
     *
     * @param tree      KD-tree
     * @param query     Query point [dim]
     * @param radius    Search radius (Chebyshev distance)
     * @return          Number of points within radius (excluding query itself)
     */
    PCMCI_API int32_t pcmci_kdtree_count_radius(
        const pcmci_kdtree_t *tree,
        const double *query,
        double radius);

    /**
     * Free KD-tree
     */
    PCMCI_API void pcmci_kdtree_free(pcmci_kdtree_t *tree);

    /*============================================================================
     * Utility Functions
     *============================================================================*/

    /**
     * Digamma function ψ(x) = d/dx ln(Γ(x))
     *
     * Used in KSG entropy estimation.
     */
    PCMCI_API double pcmci_digamma(double x);

    /**
     * Normalize data to [0, 1] range per dimension
     *
     * Important for k-NN based methods to work correctly.
     *
     * @param data      Input/output data [n x dim], column-major
     * @param n         Number of samples
     * @param dim       Number of dimensions
     */
    PCMCI_API void pcmci_normalize_data(double *data, int32_t n, int32_t dim);

#ifdef __cplusplus
}
#endif

#endif /* PCMCI_CMI_H */