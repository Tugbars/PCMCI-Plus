/**
 * @file example_cmi.c
 * @brief Test CMI (Conditional Mutual Information) implementation
 *
 * Demonstrates the KSG k-NN estimator for detecting nonlinear dependencies.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include "pcmci_cmi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*============================================================================
 * Test Data Generation
 *============================================================================*/

/* Linear relationship: Y = 0.8*X + noise */
static void generate_linear(double* X, double* Y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;  /* [-1, 1] */
        Y[i] = 0.8 * X[i] + 0.3 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/* Nonlinear relationship: Y = X^2 + noise */
static void generate_quadratic(double* X, double* Y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Y[i] = X[i] * X[i] + 0.2 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/* Nonlinear relationship: Y = sin(2*pi*X) + noise */
static void generate_sinusoidal(double* X, double* Y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Y[i] = sin(2.0 * M_PI * X[i]) + 0.3 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/* Independent: no relationship */
static void generate_independent(double* X, double* Y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Y[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

/* Conditional dependency: Y = X + Z + noise, test X;Y|Z */
static void generate_conditional(double* X, double* Y, double* Z, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Z[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Y[i] = 0.5 * X[i] + 0.5 * Z[i] + 0.2 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/* Conditional independence: Y = Z + noise, X independent given Z */
static void generate_cond_independent(double* X, double* Y, double* Z, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        Z[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        X[i] = 0.8 * Z[i] + 0.3 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
        Y[i] = 0.8 * Z[i] + 0.3 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/*============================================================================
 * Main Test
 *============================================================================*/

int main(void)
{
    printf("CMI (Conditional Mutual Information) Test\n");
    printf("==========================================\n\n");
    
    int n = 1000;
    int k = 5;
    int n_perm = 100;
    
    double* X = (double*)malloc(n * sizeof(double));
    double* Y = (double*)malloc(n * sizeof(double));
    double* Z = (double*)malloc(n * sizeof(double));
    
    pcmci_cmi_config_t config = pcmci_cmi_default_config();
    config.k = k;
    config.n_perm = n_perm;
    config.seed = 12345;
    
    printf("Settings: n=%d, k=%d, n_perm=%d\n\n", n, k, n_perm);
    
    /* ========== Test 1: MI for Linear Relationship ========== */
    printf("Test 1: Linear relationship (Y = 0.8*X + noise)\n");
    generate_linear(X, Y, n, 42);
    
    clock_t start = clock();
    pcmci_cmi_result_t result = pcmci_cmi_test(X, Y, NULL, n, 0, &config);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  MI(X;Y) = %.4f, p-value = %.4f (%.1f ms)\n", 
           result.cmi, result.pvalue, elapsed * 1000);
    printf("  Expected: High MI, low p-value (significant dependency)\n\n");
    
    /* ========== Test 2: MI for Quadratic Relationship ========== */
    printf("Test 2: Quadratic relationship (Y = X^2 + noise)\n");
    generate_quadratic(X, Y, n, 42);
    
    /* First check partial correlation (linear test) */
    pcmci_ci_result_t parcorr = pcmci_parcorr_test(X, Y, NULL, n, 0);
    printf("  Partial corr: r = %.4f, p-value = %.4f\n", parcorr.val, parcorr.pvalue);
    
    /* Now CMI (nonlinear test) */
    start = clock();
    result = pcmci_cmi_test(X, Y, NULL, n, 0, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  MI(X;Y) = %.4f, p-value = %.4f (%.1f ms)\n", 
           result.cmi, result.pvalue, elapsed * 1000);
    printf("  Expected: Parcorr misses it, CMI detects it!\n\n");
    
    /* ========== Test 3: MI for Sinusoidal Relationship ========== */
    printf("Test 3: Sinusoidal relationship (Y = sin(2*pi*X) + noise)\n");
    generate_sinusoidal(X, Y, n, 42);
    
    parcorr = pcmci_parcorr_test(X, Y, NULL, n, 0);
    printf("  Partial corr: r = %.4f, p-value = %.4f\n", parcorr.val, parcorr.pvalue);
    
    start = clock();
    result = pcmci_cmi_test(X, Y, NULL, n, 0, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  MI(X;Y) = %.4f, p-value = %.4f (%.1f ms)\n", 
           result.cmi, result.pvalue, elapsed * 1000);
    printf("  Expected: Parcorr ~0 (symmetric), CMI detects dependency!\n\n");
    
    /* ========== Test 4: MI for Independent Variables ========== */
    printf("Test 4: Independent variables\n");
    generate_independent(X, Y, n, 42);
    
    start = clock();
    result = pcmci_cmi_test(X, Y, NULL, n, 0, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  MI(X;Y) = %.4f, p-value = %.4f (%.1f ms)\n", 
           result.cmi, result.pvalue, elapsed * 1000);
    printf("  Expected: MI ~0, high p-value (no dependency)\n\n");
    
    /* ========== Test 5: CMI with Conditioning ========== */
    printf("Test 5: Conditional dependency (Y = 0.5*X + 0.5*Z + noise)\n");
    generate_conditional(X, Y, Z, n, 42);
    
    /* Unconditional MI */
    result = pcmci_cmi_test(X, Y, NULL, n, 0, &config);
    printf("  MI(X;Y) = %.4f, p-value = %.4f\n", result.cmi, result.pvalue);
    
    /* Conditional MI */
    start = clock();
    result = pcmci_cmi_test(X, Y, Z, n, 1, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  CMI(X;Y|Z) = %.4f, p-value = %.4f (%.1f ms)\n", 
           result.cmi, result.pvalue, elapsed * 1000);
    printf("  Expected: Both significant (X affects Y even given Z)\n\n");
    
    /* ========== Test 6: Conditional Independence ========== */
    printf("Test 6: Conditional independence (X->Z->Y, X indep Y | Z)\n");
    generate_cond_independent(X, Y, Z, n, 42);
    
    /* Unconditional MI */
    result = pcmci_cmi_test(X, Y, NULL, n, 0, &config);
    printf("  MI(X;Y) = %.4f, p-value = %.4f (spurious!)\n", result.cmi, result.pvalue);
    
    /* Conditional MI */
    start = clock();
    result = pcmci_cmi_test(X, Y, Z, n, 1, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  CMI(X;Y|Z) = %.4f, p-value = %.4f (%.1f ms)\n", 
           result.cmi, result.pvalue, elapsed * 1000);
    printf("  Expected: MI high (confounded), CMI ~0 (independent given Z)\n\n");
    
    /* ========== Benchmark ========== */
    printf("Performance Benchmark\n");
    printf("---------------------\n");
    
    int sizes[] = {500, 1000, 2000, 5000};
    for (int s = 0; s < 4; s++) {
        int nn = sizes[s];
        double* Xb = (double*)malloc(nn * sizeof(double));
        double* Yb = (double*)malloc(nn * sizeof(double));
        generate_linear(Xb, Yb, nn, 42);
        
        /* Value only (no permutation) */
        start = clock();
        double mi = pcmci_mi_value(Xb, Yb, nn, k);
        elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("  n=%4d: MI=%.4f, time=%.1f ms (value only)\n", nn, mi, elapsed * 1000);
        
        free(Xb);
        free(Yb);
    }
    
    printf("\n==========================================\n");
    printf("CMI test complete!\n");
    
    free(X);
    free(Y);
    free(Z);
    
    return 0;
}
