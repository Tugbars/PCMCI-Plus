# PCMCI+ in C

High-performance C implementation of PCMCI+ for time series causal discovery.

Based on: Runge, J. (2020). *Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets.* UAI 2020.

## Features

- **PC-stable skeleton discovery** with parallelized conditioning set enumeration
- **MCI (Momentary Conditional Independence)** for robust causal strength estimation
- **Partial correlation test** with MKL-accelerated linear algebra
- **Multiple testing correction**: Benjamini-Hochberg, Benjamini-Yekutieli, Bonferroni
- **OpenMP parallelization** for skeleton discovery and batch testing
- **Memory-efficient** SoA data layout with 64-byte alignment for AVX-512

## Requirements

- C11 compiler (GCC 9+ or Clang 10+)
- Intel MKL (oneAPI)
- OpenMP
- CMake 3.16+

## Building

```bash
# Set up MKL environment
source /opt/intel/oneapi/setvars.sh

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run examples
./pcmci_example
./pcmci_synthetic
```

## Quick Start

```c
#include "pcmci.h"

int main() {
    // Load your time series data (n_vars x T matrix, row-major)
    double* data = load_your_data();  
    int32_t n_vars = 4;
    int32_t T = 1000;
    int32_t tau_max = 3;
    
    // Create dataframe
    pcmci_dataframe_t* df = pcmci_dataframe_create(data, n_vars, T, tau_max);
    
    // Configure
    pcmci_config_t config = pcmci_default_config();
    config.tau_max = tau_max;
    config.alpha_level = 0.05;
    config.fdr_method = PCMCI_FDR_BH;
    
    // Run PCMCI+
    pcmci_result_t* result = pcmci_run(df, &config);
    
    // Process results
    for (int32_t i = 0; i < result->n_links; i++) {
        pcmci_link_t* link = &result->links[i];
        printf("X%d(t-%d) --> X%d(t): val=%.4f, pval=%.4e\n",
               link->i, link->tau, link->j, link->val, link->pvalue);
    }
    
    // Cleanup
    pcmci_result_free(result);
    pcmci_dataframe_free(df);
    
    return 0;
}
```

## API Overview

### Data Structures

```c
// Time series container
pcmci_dataframe_t* pcmci_dataframe_create(double* data, int32_t n_vars, 
                                           int32_t T, int32_t tau_max);

// Algorithm configuration
pcmci_config_t config = pcmci_default_config();
config.tau_max = 5;           // Maximum lag
config.alpha_level = 0.05;    // Significance threshold
config.fdr_method = PCMCI_FDR_BH;  // BH, BY, or Bonferroni
config.n_threads = 8;         // OpenMP threads (0 = auto)
config.verbosity = 1;         // 0=silent, 1=progress, 2=detailed
```

### Main Functions

```c
// Full PCMCI+ algorithm
pcmci_result_t* pcmci_run(const pcmci_dataframe_t* df, const pcmci_config_t* config);

// Just skeleton discovery
pcmci_graph_t* pcmci_skeleton(const pcmci_dataframe_t* df, const pcmci_config_t* config);

// Just MCI phase (on existing skeleton)
void pcmci_mci(const pcmci_dataframe_t* df, pcmci_graph_t* skeleton,
               const pcmci_config_t* config);

// Standalone partial correlation test
pcmci_ci_result_t pcmci_parcorr_test(const double* X, const double* Y,
                                      const double* Z, int32_t n, int32_t k);
```

### Results

```c
typedef struct {
    pcmci_graph_t* graph;       // Final causal graph
    pcmci_link_t* links;        // Array of significant links
    int32_t n_links;            // Number of significant links
    double runtime_secs;        // Total runtime
} pcmci_result_t;

typedef struct {
    int32_t i;      // Source variable
    int32_t tau;    // Time lag
    int32_t j;      // Target variable
    double val;     // Partial correlation
    double pvalue;  // FDR-adjusted p-value
} pcmci_link_t;
```

## Use Case: Cross-Market Analysis

The included `example_synthetic.c` demonstrates discovering causal links between:
- Stock indices (SPY)
- Cryptocurrency (BTC)
- Bonds (TLT)
- Volatility (VIX)

```bash
./pcmci_synthetic
```

For real data, export your returns as a CSV and load into the dataframe:

```c
// Example: Load from CSV (you'd implement csv_load)
double* returns = csv_load("market_returns.csv", &n_vars, &T);
pcmci_dataframe_t* df = pcmci_dataframe_create(returns, n_vars, T, tau_max);

const char* names[] = {"SPY", "QQQ", "BTC", "ETH", "TLT", "VIX"};
pcmci_dataframe_set_names(df, names);
```

## Performance Notes

- Skeleton discovery is O(n² × τ_max × C(k, cond_dim)) where k = neighbors
- MCI is O(n² × τ_max) after skeleton
- Partial correlation is O(n × k²) per test (dominated by QR decomposition)
- OpenMP parallelizes over target variables in skeleton discovery
- Batch testing shares residualized X across multiple Y candidates

For large problems (>20 variables, >5000 time points), consider:
- Limiting `max_cond_dim` (e.g., 3-5)
- Using stricter `alpha_level` in skeleton phase
- Running on many-core systems

## License

GPL-3.0-or-later

## References

1. Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. *Proceedings of Machine Learning Research, UAI 2020*.

2. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

3. Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5(11).
