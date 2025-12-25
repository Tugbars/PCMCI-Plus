/**
 * @file pcmci_types.h
 * @brief Type definitions for PCMCI+ causal discovery
 * 
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PCMCI_TYPES_H
#define PCMCI_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Basic Types
 *============================================================================*/

/** Variable-lag pair identifying a node in the time series graph */
typedef struct {
    int32_t var;    /**< Variable index */
    int32_t tau;    /**< Time lag (0 = contemporaneous, >0 = lagged) */
} pcmci_varlag_t;

/** Result of a conditional independence test */
typedef struct {
    double val;         /**< Test statistic value (e.g., partial correlation) */
    double pvalue;      /**< p-value for independence test */
    double stat;        /**< Raw test statistic (e.g., t-statistic) */
    int32_t df;         /**< Degrees of freedom */
} pcmci_ci_result_t;

/** A causal link with test results */
typedef struct {
    int32_t i;          /**< Source variable */
    int32_t tau;        /**< Time lag */
    int32_t j;          /**< Target variable */
    double val;         /**< Causal strength (partial correlation) */
    double pvalue;      /**< p-value after MCI */
} pcmci_link_t;

/** Separating set for a removed link */
typedef struct {
    int32_t* vars;      /**< Variable indices in separating set */
    int32_t* taus;      /**< Corresponding lags */
    int32_t size;       /**< Size of separating set */
} pcmci_sepset_t;

/*============================================================================
 * Time Series Data Frame
 *============================================================================*/

/** 
 * Multivariate time series container (SoA layout)
 * Data stored as [n_vars x T] row-major matrix
 */
typedef struct {
    double* data;           /**< Raw data: data[i*T + t] = X_i(t) */
    char** var_names;       /**< Optional variable names */
    int32_t n_vars;         /**< Number of variables */
    int32_t T;              /**< Number of time points */
    int32_t tau_max;        /**< Maximum lag to consider */
    bool owns_data;         /**< Whether to free data on destroy */
} pcmci_dataframe_t;

/*============================================================================
 * Graph Structures  
 *============================================================================*/

/** Link type for orientation */
typedef enum {
    PCMCI_LINK_NONE = 0,        /**< No link */
    PCMCI_LINK_UNDIRECTED,      /**< o--o (undetermined) */
    PCMCI_LINK_DIRECTED,        /**< --> (causal) */
    PCMCI_LINK_BIDIRECTED,      /**< <-> (confounded) */
    PCMCI_LINK_UNCERTAIN        /**< o-> (partially determined) */
} pcmci_link_type_t;

/**
 * Causal graph from PCMCI+ analysis
 * 
 * Adjacency stored as 3D array: adj[i][tau][j]
 * For tau > 0: always directed (past cannot be caused by future)
 * For tau = 0: may be undirected/bidirected (contemporaneous)
 */
typedef struct {
    /* Adjacency matrix: adj[i * (tau_max+1) * n_vars + tau * n_vars + j] */
    bool* adj;              /**< Flattened adjacency array */
    pcmci_link_type_t* link_types;  /**< Link type for each edge */
    double* val_matrix;     /**< Causal strengths */
    double* pval_matrix;    /**< p-values */
    
    /* Separating sets for removed links */
    pcmci_sepset_t* sepsets;    /**< Flattened sepset array */
    
    int32_t n_vars;
    int32_t tau_max;
} pcmci_graph_t;

/*============================================================================
 * Algorithm Configuration
 *============================================================================*/

/** Conditional independence test type */
typedef enum {
    PCMCI_TEST_PARCORR = 0,     /**< Partial correlation (Gaussian) */
    PCMCI_TEST_CMI_KNN,         /**< CMI via k-NN (nonparametric) - future */
    PCMCI_TEST_GPDC             /**< Gaussian Process DC - future */
} pcmci_test_type_t;

/** Multiple testing correction method */
typedef enum {
    PCMCI_FDR_NONE = 0,         /**< No correction */
    PCMCI_FDR_BH,               /**< Benjamini-Hochberg */
    PCMCI_FDR_BY,               /**< Benjamini-Yekutieli */
    PCMCI_FDR_BONFERRONI        /**< Bonferroni (conservative) */
} pcmci_fdr_method_t;

/** PCMCI+ algorithm configuration */
typedef struct {
    /* Test parameters */
    pcmci_test_type_t test_type;    /**< CI test to use */
    double alpha_level;             /**< Significance level for skeleton */
    double alpha_mci;               /**< Significance for MCI (0 = same as alpha_level) */
    
    /* Algorithm parameters */
    int32_t tau_min;                /**< Minimum lag (usually 0 or 1) */
    int32_t tau_max;                /**< Maximum lag */
    int32_t max_cond_dim;           /**< Max conditioning set size (-1 = unlimited) */
    bool pc_stable;                 /**< Use PC-stable variant (recommended) */
    
    /* Multiple testing */
    pcmci_fdr_method_t fdr_method;  /**< FDR correction method */
    
    /* Parallelization */
    int32_t n_threads;              /**< Number of OpenMP threads (0 = auto) */
    
    /* Verbosity */
    int32_t verbosity;              /**< 0=silent, 1=progress, 2=detailed */
} pcmci_config_t;

/*============================================================================
 * Results Structure
 *============================================================================*/

/** Complete PCMCI+ results */
typedef struct {
    pcmci_graph_t* graph;           /**< Final causal graph */
    pcmci_link_t* links;            /**< Array of significant links */
    int32_t n_links;                /**< Number of significant links */
    
    /* Statistics */
    int64_t n_tests;                /**< Total CI tests performed */
    double runtime_secs;            /**< Total runtime */
    
    /* Intermediate results (if requested) */
    pcmci_graph_t* skeleton;        /**< Skeleton before MCI (optional) */
} pcmci_result_t;

#ifdef __cplusplus
}
#endif

#endif /* PCMCI_TYPES_H */
