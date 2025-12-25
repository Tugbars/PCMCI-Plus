/**
 * @file example_basic.c
 * @brief Basic PCMCI+ usage example
 * 
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    printf("PCMCI+ Basic Example\n");
    printf("====================\n\n");
    
    /*----------------------------------------------------------------------
     * Create synthetic data with known causal structure:
     * 
     *   X0(t-1) --> X0(t)      (autoregressive)
     *   X0(t-1) --> X1(t)      (causal link)
     *   X1(t-1) --> X1(t)      (autoregressive)
     *   X1(t-1) --> X2(t)      (causal link)
     *   X2(t-1) --> X2(t)      (autoregressive)
     * 
     * No contemporaneous links.
     *----------------------------------------------------------------------*/
    
    int32_t n_vars = 3;
    int32_t T = 500;
    int32_t tau_max = 2;
    
    /* Allocate data */
    pcmci_dataframe_t* df = pcmci_dataframe_alloc(n_vars, T, tau_max);
    if (!df) {
        fprintf(stderr, "Failed to allocate dataframe\n");
        return 1;
    }
    
    /* Set variable names */
    const char* names[] = {"X0", "X1", "X2"};
    pcmci_dataframe_set_names(df, names);
    
    /* Generate data with known causal structure */
    double* X0 = df->data;
    double* X1 = df->data + T;
    double* X2 = df->data + 2 * T;
    
    /* Initialize with random values */
    srand(42);
    for (int32_t t = 0; t < tau_max; t++) {
        X0[t] = (double)rand() / RAND_MAX - 0.5;
        X1[t] = (double)rand() / RAND_MAX - 0.5;
        X2[t] = (double)rand() / RAND_MAX - 0.5;
    }
    
    /* Generate with causal structure */
    double c_auto = 0.8;   /* Autoregressive coefficient */
    double c_cross = 0.5;  /* Cross-variable coefficient */
    double noise = 0.3;    /* Noise standard deviation */
    
    for (int32_t t = tau_max; t < T; t++) {
        double e0 = noise * ((double)rand() / RAND_MAX - 0.5);
        double e1 = noise * ((double)rand() / RAND_MAX - 0.5);
        double e2 = noise * ((double)rand() / RAND_MAX - 0.5);
        
        /* X0(t) = c_auto * X0(t-1) + noise */
        X0[t] = c_auto * X0[t-1] + e0;
        
        /* X1(t) = c_auto * X1(t-1) + c_cross * X0(t-1) + noise */
        X1[t] = c_auto * X1[t-1] + c_cross * X0[t-1] + e1;
        
        /* X2(t) = c_auto * X2(t-1) + c_cross * X1(t-1) + noise */
        X2[t] = c_auto * X2[t-1] + c_cross * X1[t-1] + e2;
    }
    
    printf("Generated %d time points for %d variables\n", T, n_vars);
    printf("True causal structure:\n");
    printf("  X0(t-1) --> X0(t) [%.2f]\n", c_auto);
    printf("  X0(t-1) --> X1(t) [%.2f]\n", c_cross);
    printf("  X1(t-1) --> X1(t) [%.2f]\n", c_auto);
    printf("  X1(t-1) --> X2(t) [%.2f]\n", c_cross);
    printf("  X2(t-1) --> X2(t) [%.2f]\n", c_auto);
    printf("\n");
    
    /*----------------------------------------------------------------------
     * Run PCMCI+
     *----------------------------------------------------------------------*/
    
    pcmci_config_t config = pcmci_default_config();
    config.tau_max = tau_max;
    config.alpha_level = 0.05;
    config.fdr_method = PCMCI_FDR_BH;
    config.verbosity = 2;
    
    pcmci_result_t* result = pcmci_run(df, &config);
    
    if (!result) {
        fprintf(stderr, "PCMCI+ failed\n");
        pcmci_dataframe_free(df);
        return 1;
    }
    
    /*----------------------------------------------------------------------
     * Evaluate results
     *----------------------------------------------------------------------*/
    
    printf("\n========================================\n");
    printf("Evaluation\n");
    printf("========================================\n");
    
    /* Check if we recovered the true links */
    bool found_x0_x0 = false, found_x0_x1 = false;
    bool found_x1_x1 = false, found_x1_x2 = false;
    bool found_x2_x2 = false;
    int32_t false_positives = 0;
    
    for (int32_t i = 0; i < result->n_links; i++) {
        pcmci_link_t* link = &result->links[i];
        
        if (link->i == 0 && link->j == 0 && link->tau == 1) found_x0_x0 = true;
        else if (link->i == 0 && link->j == 1 && link->tau == 1) found_x0_x1 = true;
        else if (link->i == 1 && link->j == 1 && link->tau == 1) found_x1_x1 = true;
        else if (link->i == 1 && link->j == 2 && link->tau == 1) found_x1_x2 = true;
        else if (link->i == 2 && link->j == 2 && link->tau == 1) found_x2_x2 = true;
        else false_positives++;
    }
    
    int32_t true_positives = (found_x0_x0 + found_x0_x1 + found_x1_x1 + 
                              found_x1_x2 + found_x2_x2);
    int32_t total_true = 5;
    int32_t false_negatives = total_true - true_positives;
    
    printf("True Positives: %d / %d\n", true_positives, total_true);
    printf("False Negatives: %d\n", false_negatives);
    printf("False Positives: %d\n", false_positives);
    
    if (true_positives > 0) {
        double precision = (double)true_positives / (true_positives + false_positives);
        double recall = (double)true_positives / total_true;
        double f1 = 2 * precision * recall / (precision + recall);
        printf("Precision: %.3f\n", precision);
        printf("Recall: %.3f\n", recall);
        printf("F1 Score: %.3f\n", f1);
    }
    
    /*----------------------------------------------------------------------
     * Cleanup
     *----------------------------------------------------------------------*/
    
    pcmci_result_free(result);
    pcmci_dataframe_free(df);
    
    return 0;
}
