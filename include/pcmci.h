/**
 * @file pcmci.h
 * @brief PCMCI+ Causal Discovery Algorithm
 * 
 * C implementation of PCMCI+ (Runge 2020) for time series causal discovery.
 * Uses Intel MKL for linear algebra and OpenMP for parallelization.
 * 
 * Reference:
 *   Runge, J. (2020). Discovering contemporaneous and lagged causal relations
 *   in autocorrelated nonlinear time series datasets. UAI 2020.
 * 
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PCMCI_H
#define PCMCI_H

#include "pcmci_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Version Info
 *============================================================================*/

#define PCMCI_VERSION_MAJOR 0
#define PCMCI_VERSION_MINOR 1
#define PCMCI_VERSION_PATCH 0

const char* pcmci_version(void);

/*============================================================================
 * Configuration
 *============================================================================*/

/**
 * Get default configuration
 * 
 * Defaults:
 *   - test_type: PCMCI_TEST_PARCORR
 *   - alpha_level: 0.05
 *   - tau_min: 0
 *   - tau_max: 1
 *   - max_cond_dim: -1 (unlimited)
 *   - pc_stable: true
 *   - fdr_method: PCMCI_FDR_BH
 *   - n_threads: 0 (auto)
 *   - verbosity: 1
 */
pcmci_config_t pcmci_default_config(void);

/*============================================================================
 * Data Frame API
 *============================================================================*/

/**
 * Create dataframe from existing data (does not copy)
 * 
 * @param data      Row-major matrix [n_vars x T], caller retains ownership
 * @param n_vars    Number of variables
 * @param T         Number of time points
 * @param tau_max   Maximum lag to consider
 * @return          New dataframe (call pcmci_dataframe_free when done)
 */
pcmci_dataframe_t* pcmci_dataframe_create(double* data, int32_t n_vars, 
                                           int32_t T, int32_t tau_max);

/**
 * Create dataframe with owned data (copies input)
 */
pcmci_dataframe_t* pcmci_dataframe_create_copy(const double* data, int32_t n_vars,
                                                int32_t T, int32_t tau_max);

/**
 * Create dataframe and allocate storage
 */
pcmci_dataframe_t* pcmci_dataframe_alloc(int32_t n_vars, int32_t T, int32_t tau_max);

/**
 * Set variable names (optional)
 */
void pcmci_dataframe_set_names(pcmci_dataframe_t* df, const char** names);

/**
 * Free dataframe
 */
void pcmci_dataframe_free(pcmci_dataframe_t* df);

/**
 * Extract lagged variable: X_var(t - tau) for t in [tau_max, T)
 * 
 * @param df        Dataframe
 * @param var       Variable index
 * @param tau       Lag
 * @param out_len   Output: length of returned array
 * @return          Newly allocated array (caller must free with pcmci_free)
 */
double* pcmci_extract_lagged(const pcmci_dataframe_t* df, int32_t var, 
                              int32_t tau, int32_t* out_len);

/**
 * Extract conditioning set as matrix
 * 
 * @param df            Dataframe  
 * @param varlags       Array of (var, tau) pairs
 * @param n_cond        Number of conditioning variables
 * @param out_n_samples Output: number of samples (rows)
 * @return              Row-major matrix [n_samples x n_cond], or NULL if n_cond=0
 */
double* pcmci_extract_cond_set(const pcmci_dataframe_t* df,
                                const pcmci_varlag_t* varlags,
                                int32_t n_cond,
                                int32_t* out_n_samples);

/*============================================================================
 * Conditional Independence Tests
 *============================================================================*/

/**
 * Partial correlation test
 * 
 * Tests H0: corr(X, Y | Z) = 0
 * Uses residualization: resid_X = X - Z @ beta_X, then correlate residuals
 * 
 * @param X     Array of length n
 * @param Y     Array of length n  
 * @param Z     Conditioning matrix [n x k], row-major, or NULL if k=0
 * @param n     Number of samples
 * @param k     Number of conditioning variables
 * @return      Test result with partial correlation and p-value
 */
pcmci_ci_result_t pcmci_parcorr_test(const double* X, const double* Y,
                                      const double* Z, int32_t n, int32_t k);

/**
 * Batch partial correlation: test X against multiple Y's
 * 
 * @param X         Array of length n (single X)
 * @param Y_batch   Matrix [m x n] of Y candidates, row-major
 * @param Z         Conditioning matrix [n x k], or NULL
 * @param n         Number of samples
 * @param k         Number of conditioning variables
 * @param m         Number of Y candidates
 * @param results   Output array of m results (pre-allocated)
 */
void pcmci_parcorr_batch(const double* X, const double* Y_batch,
                          const double* Z, int32_t n, int32_t k, int32_t m,
                          pcmci_ci_result_t* results);

/*============================================================================
 * Graph API
 *============================================================================*/

/**
 * Allocate graph structure
 */
pcmci_graph_t* pcmci_graph_alloc(int32_t n_vars, int32_t tau_max);

/**
 * Free graph
 */
void pcmci_graph_free(pcmci_graph_t* g);

/**
 * Check if link exists
 */
bool pcmci_graph_has_link(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j);

/**
 * Set link existence
 */
void pcmci_graph_set_link(pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j, bool exists);

/**
 * Get link value (partial correlation)
 */
double pcmci_graph_get_val(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j);

/**
 * Get link p-value
 */
double pcmci_graph_get_pval(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j);

/**
 * Get neighbors of target variable j
 * 
 * @param g             Graph
 * @param j             Target variable
 * @param out_count     Output: number of neighbors
 * @return              Array of (var, tau) neighbors (caller frees with pcmci_free)
 */
pcmci_varlag_t* pcmci_graph_get_neighbors(const pcmci_graph_t* g, int32_t j,
                                           int32_t* out_count);

/**
 * Print graph summary
 */
void pcmci_graph_print(const pcmci_graph_t* g, const char** var_names);

/*============================================================================
 * Main Algorithm
 *============================================================================*/

/**
 * Run PCMCI+ algorithm
 * 
 * @param df        Input time series data
 * @param config    Algorithm configuration
 * @return          Results structure (caller frees with pcmci_result_free)
 */
pcmci_result_t* pcmci_run(const pcmci_dataframe_t* df, const pcmci_config_t* config);

/**
 * Run skeleton discovery only (PC-stable phase)
 */
pcmci_graph_t* pcmci_skeleton(const pcmci_dataframe_t* df, const pcmci_config_t* config);

/**
 * Run MCI phase on existing skeleton
 */
void pcmci_mci(const pcmci_dataframe_t* df, pcmci_graph_t* skeleton,
               const pcmci_config_t* config);

/**
 * Free results
 */
void pcmci_result_free(pcmci_result_t* result);

/*============================================================================
 * Significance / Multiple Testing
 *============================================================================*/

/**
 * Apply FDR correction to p-values
 * 
 * @param pvalues       Array of p-values
 * @param n             Number of p-values
 * @param method        Correction method
 * @param adjusted      Output: adjusted p-values (pre-allocated, length n)
 */
void pcmci_fdr_correct(const double* pvalues, int32_t n, 
                        pcmci_fdr_method_t method, double* adjusted);

/**
 * Get significant links after FDR correction
 * 
 * @param graph         Graph with p-values
 * @param alpha         Significance threshold
 * @param method        FDR method
 * @param out_count     Output: number of significant links
 * @return              Array of significant links
 */
pcmci_link_t* pcmci_get_significant_links(const pcmci_graph_t* graph,
                                           double alpha,
                                           pcmci_fdr_method_t method,
                                           int32_t* out_count);

/*============================================================================
 * Utility Functions
 *============================================================================*/

/**
 * Aligned memory allocation (64-byte for AVX-512)
 */
void* pcmci_alloc(size_t size);

/**
 * Free aligned memory
 */
void pcmci_free(void* ptr);

/**
 * Set random seed for any stochastic operations
 */
void pcmci_set_seed(uint64_t seed);

/**
 * Set number of threads
 */
void pcmci_set_threads(int32_t n_threads);

#ifdef __cplusplus
}
#endif

#endif /* PCMCI_H */
