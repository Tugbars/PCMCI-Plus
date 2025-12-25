/**
 * @file pcmci.c
 * @brief Main PCMCI+ algorithm driver
 * 
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

/*============================================================================
 * Version
 *============================================================================*/

static const char* VERSION_STRING = "0.1.0";

const char* pcmci_version(void) {
    return VERSION_STRING;
}

/*============================================================================
 * Memory Management
 *============================================================================*/

void* pcmci_alloc(size_t size) {
    return pcmci_malloc(size);
}

void pcmci_free(void* ptr) {
    pcmci_mkl_free(ptr);
}

/*============================================================================
 * Configuration
 *============================================================================*/

pcmci_config_t pcmci_default_config(void) {
    pcmci_config_t config = {
        .test_type = PCMCI_TEST_PARCORR,
        .alpha_level = 0.05,
        .alpha_mci = 0.0,  /* Same as alpha_level */
        .tau_min = 0,
        .tau_max = 1,
        .max_cond_dim = -1,  /* Unlimited */
        .pc_stable = true,
        .fdr_method = PCMCI_FDR_BH,
        .n_threads = 0,  /* Auto */
        .verbosity = 1
    };
    return config;
}

/*============================================================================
 * Thread Control
 *============================================================================*/

void pcmci_set_threads(int32_t n_threads) {
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
}

void pcmci_set_seed(uint64_t seed) {
    /* For potential future stochastic methods */
    (void)seed;
}

/*============================================================================
 * Main Algorithm
 *============================================================================*/

pcmci_result_t* pcmci_run(const pcmci_dataframe_t* df, const pcmci_config_t* config) {
    if (!df || !config) return NULL;
    
    pcmci_result_t* result = (pcmci_result_t*)calloc(1, sizeof(pcmci_result_t));
    if (!result) return NULL;
    
    double start_time = pcmci_get_time();
    
    if (config->verbosity >= 1) {
        printf("PCMCI+ v%s\n", pcmci_version());
        printf("========================================\n");
        printf("Variables: %d, Time points: %d, tau_max: %d\n",
               df->n_vars, df->T, config->tau_max);
        printf("Alpha: %.4f, FDR method: %d\n", 
               config->alpha_level, config->fdr_method);
        printf("========================================\n\n");
    }
    
    /* Create a modified config with tau_max from config */
    pcmci_dataframe_t df_modified = *df;
    df_modified.tau_max = config->tau_max;
    
    /*----------------------------------------------------------------------
     * Phase 1: Skeleton Discovery (PC-stable)
     *----------------------------------------------------------------------*/
    if (config->verbosity >= 1) {
        printf("Phase 1: Skeleton Discovery\n");
        printf("----------------------------------------\n");
    }
    
    pcmci_graph_t* skeleton = pcmci_skeleton(&df_modified, config);
    if (!skeleton) {
        free(result);
        return NULL;
    }
    
    /* Store skeleton as intermediate result */
    result->skeleton = skeleton;
    
    /*----------------------------------------------------------------------
     * Phase 2: MCI (Momentary Conditional Independence)
     *----------------------------------------------------------------------*/
    if (config->verbosity >= 1) {
        printf("\nPhase 2: MCI Testing\n");
        printf("----------------------------------------\n");
    }
    
    pcmci_mci(&df_modified, skeleton, config);
    
    /*----------------------------------------------------------------------
     * Phase 3: Extract Significant Links
     *----------------------------------------------------------------------*/
    if (config->verbosity >= 1) {
        printf("\nPhase 3: Extracting Significant Links\n");
        printf("----------------------------------------\n");
    }
    
    double alpha = config->alpha_mci > 0 ? config->alpha_mci : config->alpha_level;
    
    result->links = pcmci_get_significant_links(skeleton, alpha, 
                                                 config->fdr_method,
                                                 &result->n_links);
    
    /* Point to same graph for final result */
    result->graph = skeleton;
    
    result->runtime_secs = pcmci_get_time() - start_time;
    
    if (config->verbosity >= 1) {
        printf("\n========================================\n");
        printf("Results Summary\n");
        printf("========================================\n");
        printf("Significant links (alpha=%.4f): %d\n", alpha, result->n_links);
        printf("Runtime: %.3f seconds\n", result->runtime_secs);
        printf("\n");
        
        /* Print significant links */
        if (result->n_links > 0) {
            printf("Significant Causal Links:\n");
            for (int32_t i = 0; i < result->n_links; i++) {
                pcmci_link_t* link = &result->links[i];
                const char* src = df->var_names ? df->var_names[link->i] : NULL;
                const char* tgt = df->var_names ? df->var_names[link->j] : NULL;
                
                if (src && tgt) {
                    printf("  %s(t-%d) --> %s(t): val=%.4f, pval=%.4e\n",
                           src, link->tau, tgt, link->val, link->pvalue);
                } else {
                    printf("  X%d(t-%d) --> X%d(t): val=%.4f, pval=%.4e\n",
                           link->i, link->tau, link->j, link->val, link->pvalue);
                }
            }
        }
    }
    
    return result;
}

void pcmci_result_free(pcmci_result_t* result) {
    if (!result) return;
    
    /* Note: result->graph and result->skeleton may point to same object */
    if (result->skeleton && result->skeleton != result->graph) {
        pcmci_graph_free(result->skeleton);
    }
    if (result->graph) {
        pcmci_graph_free(result->graph);
    }
    
    free(result->links);
    free(result);
}
