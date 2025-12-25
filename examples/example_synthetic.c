/**
 * @file example_synthetic.c
 * @brief Synthetic market data example for cross-market causal discovery
 *
 * Simulates a scenario with:
 *   - Stock market index (SPY)
 *   - Crypto (BTC)
 *   - Bond yields (TLT)
 *   - Volatility (VIX)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcmci.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Box-Muller transform for Gaussian noise */
static double randn(void)
{
    static int have_spare = 0;
    static double spare;

    if (have_spare)
    {
        have_spare = 0;
        return spare;
    }

    double u, v, s;
    do
    {
        u = (double)rand() / RAND_MAX * 2.0 - 1.0;
        v = (double)rand() / RAND_MAX * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have_spare = 1;

    return u * s;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("PCMCI+ Cross-Market Causal Discovery Example\n");
    printf("=============================================\n\n");

    /*----------------------------------------------------------------------
     * Simulate market returns with known causal structure:
     *
     * True structure (lag 1):
     *   VIX(t-1)  --> SPY(t)     [negative: high vol predicts lower returns]
     *   SPY(t-1)  --> BTC(t)     [positive: stocks lead crypto]
     *   VIX(t-1)  --> TLT(t)     [positive: flight to safety]
     *   SPY(t-1)  --> VIX(t)     [negative: drops cause vol spikes]
     *
     * Plus autoregressive terms for persistence.
     * No contemporaneous links (for simplicity).
     *----------------------------------------------------------------------*/

    int32_t n_vars = 4;
    int32_t T = 1000; /* ~4 years of daily data */
    int32_t tau_max = 3;

    const char *names[] = {"SPY", "BTC", "TLT", "VIX"};

    /* Indices for clarity */
    enum
    {
        SPY = 0,
        BTC = 1,
        TLT = 2,
        VIX = 3
    };

    pcmci_dataframe_t *df = pcmci_dataframe_alloc(n_vars, T, tau_max);
    if (!df)
    {
        fprintf(stderr, "Failed to allocate dataframe\n");
        return 1;
    }
    pcmci_dataframe_set_names(df, names);

    double *spy = df->data + SPY * T;
    double *btc = df->data + BTC * T;
    double *tlt = df->data + TLT * T;
    double *vix = df->data + VIX * T;

    /* Coefficients */
    double ar_spy = 0.05; /* SPY mean-reversion */
    double ar_btc = 0.02; /* BTC slight persistence */
    double ar_tlt = 0.1;  /* TLT persistence */
    double ar_vix = 0.85; /* VIX strong persistence */

    double vix_to_spy = -0.15; /* High VIX -> lower SPY */
    double spy_to_btc = 0.25;  /* SPY leads BTC */
    double vix_to_tlt = 0.12;  /* Flight to safety */
    double spy_to_vix = -0.30; /* Drops spike VIX */

    /* Volatilities (return std devs) */
    double vol_spy = 0.01;  /* ~16% annualized */
    double vol_btc = 0.03;  /* ~48% annualized */
    double vol_tlt = 0.008; /* ~13% annualized */
    double vol_vix = 0.05;  /* VIX is mean-reverting around ~20 */

    /* Initialize */
    srand(12345);
    spy[0] = 0;
    btc[0] = 0;
    tlt[0] = 0;
    vix[0] = 0.2; /* VIX ~20 level */
    spy[1] = 0;
    btc[1] = 0;
    tlt[1] = 0;
    vix[1] = 0.2;
    spy[2] = 0;
    btc[2] = 0;
    tlt[2] = 0;
    vix[2] = 0.2;

    /* Generate time series */
    for (int32_t t = tau_max; t < T; t++)
    {
        double e_spy = vol_spy * randn();
        double e_btc = vol_btc * randn();
        double e_tlt = vol_tlt * randn();
        double e_vix = vol_vix * randn();

        /* SPY: AR + VIX effect */
        spy[t] = ar_spy * spy[t - 1] + vix_to_spy * (vix[t - 1] - 0.2) + e_spy;

        /* BTC: AR + SPY effect */
        btc[t] = ar_btc * btc[t - 1] + spy_to_btc * spy[t - 1] + e_btc;

        /* TLT: AR + VIX effect (flight to safety) */
        tlt[t] = ar_tlt * tlt[t - 1] + vix_to_tlt * (vix[t - 1] - 0.2) + e_tlt;

        /* VIX: AR (mean-reverting to 0.2) + SPY effect */
        vix[t] = 0.2 + ar_vix * (vix[t - 1] - 0.2) + spy_to_vix * spy[t - 1] + e_vix;
        vix[t] = fmax(0.05, vix[t]); /* VIX floor */
    }

    printf("Generated %d time points for %d markets\n", T, n_vars);
    printf("\nTrue causal structure (lag=1):\n");
    printf("  VIX(t-1) --> SPY(t) [%.2f]  (high vol -> lower returns)\n", vix_to_spy);
    printf("  SPY(t-1) --> BTC(t) [%.2f]  (stocks lead crypto)\n", spy_to_btc);
    printf("  VIX(t-1) --> TLT(t) [%.2f]  (flight to safety)\n", vix_to_tlt);
    printf("  SPY(t-1) --> VIX(t) [%.2f]  (drops spike vol)\n", spy_to_vix);
    printf("\n");

    /*----------------------------------------------------------------------
     * Run PCMCI+
     *----------------------------------------------------------------------*/

    pcmci_config_t config = pcmci_default_config();
    config.tau_max = tau_max;
    config.alpha_level = 0.01; /* More stringent for noisy data */
    config.fdr_method = PCMCI_FDR_BH;
    config.verbosity = 2;
    config.n_threads = 0; /* Auto-detect */

    printf("Running PCMCI+ (alpha=%.3f, tau_max=%d)...\n\n",
           config.alpha_level, config.tau_max);

    pcmci_result_t *result = pcmci_run(df, &config);

    if (!result)
    {
        fprintf(stderr, "PCMCI+ failed\n");
        pcmci_dataframe_free(df);
        return 1;
    }

    /*----------------------------------------------------------------------
     * Analyze results
     *----------------------------------------------------------------------*/

    printf("\n========================================\n");
    printf("Causal Discovery Results\n");
    printf("========================================\n\n");

    /* True links to find */
    typedef struct
    {
        int32_t i;
        int32_t j;
        int32_t tau;
        const char *desc;
    } true_link_t;
    true_link_t true_links[] = {
        {VIX, SPY, 1, "VIX->SPY"},
        {SPY, BTC, 1, "SPY->BTC"},
        {VIX, TLT, 1, "VIX->TLT"},
        {SPY, VIX, 1, "SPY->VIX"}};
    int32_t n_true = sizeof(true_links) / sizeof(true_links[0]);

    bool *found = (bool *)calloc(n_true, sizeof(bool));
    int32_t false_positives = 0;

    printf("Discovered links:\n");
    for (int32_t i = 0; i < result->n_links; i++)
    {
        pcmci_link_t *link = &result->links[i];

        /* Check if it's a true link */
        bool is_true = false;
        for (int32_t j = 0; j < n_true; j++)
        {
            if (link->i == true_links[j].i &&
                link->j == true_links[j].j &&
                link->tau == true_links[j].tau)
            {
                found[j] = true;
                is_true = true;
                break;
            }
        }

        /* Skip autoregressive terms from false positive count */
        bool is_ar = (link->i == link->j);

        printf("  %s(t-%d) --> %s(t): val=%+.4f, pval=%.2e %s\n",
               names[link->i], link->tau, names[link->j],
               link->val, link->pvalue,
               is_true ? "[TRUE]" : (is_ar ? "[AR]" : "[FP?]"));

        if (!is_true && !is_ar)
        {
            false_positives++;
        }
    }

    printf("\n");
    printf("True link recovery:\n");
    int32_t tp = 0;
    for (int32_t j = 0; j < n_true; j++)
    {
        printf("  %s: %s\n", true_links[j].desc, found[j] ? "FOUND" : "MISSED");
        if (found[j])
            tp++;
    }

    printf("\nSummary:\n");
    printf("  True Positives: %d / %d\n", tp, n_true);
    printf("  False Positives (non-AR): %d\n", false_positives);
    printf("  Runtime: %.3f sec\n", result->runtime_secs);

    if (tp > 0 && (tp + false_positives) > 0)
    {
        double precision = (double)tp / (tp + false_positives);
        double recall = (double)tp / n_true;
        printf("  Precision: %.1f%%\n", 100.0 * precision);
        printf("  Recall: %.1f%%\n", 100.0 * recall);
    }

    free(found);
    pcmci_result_free(result);
    pcmci_dataframe_free(df);

    return 0;
}