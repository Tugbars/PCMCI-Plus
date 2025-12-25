/**
 * @file example_gpdc.c
 * @brief Test GPDC (Gaussian Process Distance Correlation) implementation
 *
 * Demonstrates distance correlation and GPDC for detecting nonlinear dependencies.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#define _USE_MATH_DEFINES

#include "pcmci.h"
#include "pcmci_gpdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*============================================================================
 * Test Data Generation
 *============================================================================*/

/* Linear relationship: Y = 0.8*X + noise */
static void generate_linear(double* X, double* Y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
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

/* Circle relationship: X^2 + Y^2 = 1 + noise (zero correlation but dependent) */
static void generate_circle(double* X, double* Y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        double theta = (double)rand() / RAND_MAX * 2.0 * M_PI;
        double r = 1.0 + 0.1 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
        X[i] = r * cos(theta);
        Y[i] = r * sin(theta);
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

/* Conditional dependency: Y = X + Z + noise */
static void generate_conditional(double* X, double* Y, double* Z, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Z[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Y[i] = 0.5 * X[i] + 0.5 * Z[i] + 0.2 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/* Conditional independence: X <- Z -> Y (confounded) */
static void generate_cond_independent(double* X, double* Y, double* Z, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        Z[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        X[i] = 0.8 * Z[i] + 0.3 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
        Y[i] = 0.8 * Z[i] + 0.3 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/* Nonlinear conditional: Y = sin(Z) + noise, X = cos(Z) + noise */
static void generate_nonlinear_confound(double* X, double* Y, double* Z, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        Z[i] = (double)rand() / RAND_MAX * 2.0 * M_PI - M_PI;
        X[i] = cos(Z[i]) + 0.2 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
        Y[i] = sin(Z[i]) + 0.2 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
}

/*============================================================================
 * Main Test
 *============================================================================*/

int main(void)
{
    printf("GPDC (Gaussian Process Distance Correlation) Test\n");
    printf("==================================================\n\n");
    
    int n = 500;  /* Smaller n for GPDC since GP is O(n³) */
    int n_perm = 100;
    
    double* X = (double*)malloc(n * sizeof(double));
    double* Y = (double*)malloc(n * sizeof(double));
    double* Z = (double*)malloc(n * sizeof(double));
    
    pcmci_gpdc_config_t config = pcmci_gpdc_default_config();
    config.n_perm = n_perm;
    config.seed = 12345;
    
    printf("Settings: n=%d, n_perm=%d\n\n", n, n_perm);
    
    /* ========== Part 1: Distance Correlation Tests ========== */
    printf("=== Part 1: Distance Correlation (dCor) ===\n\n");
    
    /* Test 1: Linear */
    printf("Test 1: Linear relationship (Y = 0.8*X + noise)\n");
    generate_linear(X, Y, n, 42);
    
    clock_t start = clock();
    pcmci_gpdc_result_t result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    pcmci_ci_result_t parcorr = pcmci_parcorr_test(X, Y, NULL, n, 0);
    printf("  Parcorr: r = %.4f, p = %.4f\n", parcorr.val, parcorr.pvalue);
    printf("  dCor:    r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  Both should detect linear dependency\n\n");
    
    /* Test 2: Quadratic */
    printf("Test 2: Quadratic relationship (Y = X^2 + noise)\n");
    generate_quadratic(X, Y, n, 42);
    
    parcorr = pcmci_parcorr_test(X, Y, NULL, n, 0);
    printf("  Parcorr: r = %.4f, p = %.4f\n", parcorr.val, parcorr.pvalue);
    
    start = clock();
    result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  dCor:    r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  Parcorr ~0, dCor detects nonlinear dependency!\n\n");
    
    /* Test 3: Sinusoidal */
    printf("Test 3: Sinusoidal relationship (Y = sin(2*pi*X) + noise)\n");
    generate_sinusoidal(X, Y, n, 42);
    
    parcorr = pcmci_parcorr_test(X, Y, NULL, n, 0);
    printf("  Parcorr: r = %.4f, p = %.4f\n", parcorr.val, parcorr.pvalue);
    
    start = clock();
    result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  dCor:    r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  dCor detects sinusoidal dependency\n\n");
    
    /* Test 4: Circle */
    printf("Test 4: Circle relationship (X^2 + Y^2 = 1)\n");
    generate_circle(X, Y, n, 42);
    
    parcorr = pcmci_parcorr_test(X, Y, NULL, n, 0);
    printf("  Parcorr: r = %.4f, p = %.4f\n", parcorr.val, parcorr.pvalue);
    
    start = clock();
    result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  dCor:    r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  Parcorr=0 (uncorrelated), but dCor detects circular dependency!\n\n");
    
    /* Test 5: Independent */
    printf("Test 5: Independent variables\n");
    generate_independent(X, Y, n, 42);
    
    start = clock();
    result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  dCor:    r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  Should have high p-value (no dependency)\n\n");
    
    /* ========== Part 2: GPDC Conditional Tests ========== */
    printf("=== Part 2: GPDC Conditional Independence ===\n\n");
    
    /* Test 6: Conditional dependency */
    printf("Test 6: Conditional dependency (Y = 0.5*X + 0.5*Z + noise)\n");
    generate_conditional(X, Y, Z, n, 42);
    
    /* Unconditional */
    result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    printf("  dCor(X,Y):     r = %.4f, p = %.4f\n", result.dcor, result.pvalue);
    
    /* Conditional */
    start = clock();
    result = pcmci_gpdc_test(X, Y, Z, n, 1, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  GPDC(X,Y|Z):   r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  Both significant (X affects Y even given Z)\n\n");
    
    /* Test 7: Conditional independence (linear confound) */
    printf("Test 7: Conditional independence (X <- Z -> Y, linear)\n");
    generate_cond_independent(X, Y, Z, n, 42);
    
    /* Unconditional - should be spuriously correlated */
    result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    printf("  dCor(X,Y):     r = %.4f, p = %.4f (spurious!)\n", result.dcor, result.pvalue);
    
    /* Conditional - should be independent */
    start = clock();
    result = pcmci_gpdc_test(X, Y, Z, n, 1, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  GPDC(X,Y|Z):   r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  GPDC should show independence (high p-value)\n\n");
    
    /* Test 8: Nonlinear confounding */
    printf("Test 8: Nonlinear confound (X=cos(Z), Y=sin(Z))\n");
    generate_nonlinear_confound(X, Y, Z, n, 42);
    
    /* Unconditional */
    result = pcmci_dcor_test(X, Y, n, n_perm, 12345);
    printf("  dCor(X,Y):     r = %.4f, p = %.4f (spurious!)\n", result.dcor, result.pvalue);
    
    /* Conditional with GPDC */
    start = clock();
    result = pcmci_gpdc_test(X, Y, Z, n, 1, &config);
    elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("  GPDC(X,Y|Z):   r = %.4f, p = %.4f (%.1f ms)\n", 
           result.dcor, result.pvalue, elapsed * 1000);
    printf("  GP should remove nonlinear confound -> independence\n\n");
    
    /* ========== Performance Benchmark ========== */
    printf("=== Performance Benchmark ===\n\n");
    
    int sizes[] = {100, 200, 500, 1000};
    for (int s = 0; s < 4; s++) {
        int nn = sizes[s];
        double* Xb = (double*)malloc(nn * sizeof(double));
        double* Yb = (double*)malloc(nn * sizeof(double));
        generate_linear(Xb, Yb, nn, 42);
        
        /* dCor value only (no permutation) */
        start = clock();
        double dcor = pcmci_dcor(Xb, Yb, nn);
        elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("  n=%4d: dCor=%.4f, time=%.1f ms (value only)\n", nn, dcor, elapsed * 1000);
        
        free(Xb);
        free(Yb);
    }
    
    printf("\n==================================================\n");
    printf("GPDC test complete!\n");
    
    free(X);
    free(Y);
    free(Z);
    
    return 0;
}
