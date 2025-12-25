# PCMCI+ : High-Performance Causal Discovery for Time Series

A fast C implementation of the PCMCI+ algorithm for causal discovery in multivariate time series data, with Python bindings.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

PCMCI+ (Runge, 2020) is a state-of-the-art algorithm for discovering causal relationships in time series data. It combines:

- **PC algorithm** for skeleton discovery (removing spurious correlations)
- **MCI (Momentary Conditional Independence)** for robust causal strength estimation

This implementation focuses on **performance** and **production use**, achieving sub-millisecond latency for small systems and handling 20+ variables interactively.

### Key Features

- **Intel MKL Acceleration**: BLAS/LAPACK operations optimized for Intel CPUs
- **OpenMP Parallelization**: Multi-threaded skeleton discovery and CI tests  
- **Spearman Rank Correlation**: Robust to outliers and non-Gaussian data
- **Cholesky Fast-Path**: 2-4x faster residualization than QR decomposition
- **Lock-Free Parallel Skeleton**: Eliminates thread contention
- **Lazy P-Value Computation**: Skips expensive calculations for weak correlations
- **FDR Correction**: Benjamini-Hochberg multiple testing correction

### Performance

| Variables | Samples | Time (8 P-cores) |
|-----------|---------|------------------|
| 5         | 500     | ~1 ms            |
| 10        | 1000    | ~180 ms          |
| 20        | 500     | ~110 ms          |
| 5         | 5000    | ~25 ms           |

*Benchmarked on Intel i9-14900KF with MKL 2025*

## Installation

### Prerequisites

- **CMake** 3.16+
- **Intel oneAPI** (MKL + Compiler) or OpenBLAS
- **C11 Compiler** (GCC, Clang, MSVC, or Intel ICX)
- **Python 3.8+** (for bindings)

### Windows (Intel oneAPI)

1. Install [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

2. Open **Intel oneAPI Command Prompt** and build:

```cmd
cd PCMCI-Plus
mkdir build && cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

3. The DLL is automatically copied to `python/` folder.

### Linux

```bash
# With Intel MKL
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# With OpenBLAS
sudo apt install libopenblas-dev
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_OPENBLAS=ON
make -j$(nproc)
```

### macOS

```bash
brew install openblas libomp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

## Python Usage

### Quick Start

```python
import numpy as np
from pcmci import PCMCI, run_pcmci
from visualize import plot_graph, print_parents

# Your data: shape (n_variables, n_timepoints)
data = np.random.randn(5, 1000)

# Run PCMCI+
result = run_pcmci(data, tau_max=3, alpha=0.05)

# View results
print(result)
print_parents(result)

# Visualize
plot_graph(result, save_path='causal_graph.png')
```

### Detailed Usage

```python
from pcmci import PCMCI

# Create PCMCI object
pcmci = PCMCI(
    data,                          # Shape: (n_vars, T)
    tau_max=5,                     # Maximum lag to test
    var_names=['X', 'Y', 'Z']      # Optional variable names
)

# Run with custom parameters
result = pcmci.run(
    alpha=0.05,           # Significance level
    max_cond_dim=-1,      # Max conditioning set size (-1 = auto)
    n_threads=8,          # Number of threads (0 = auto)
    use_spearman=True,    # Spearman (robust) vs Pearson
    winsorize_thresh=0.0, # Winsorization (0.01 = 1%/99%)
    fdr_method=1,         # 0=none, 1=Benjamini-Hochberg
    verbosity=0           # 0=silent, 1=progress, 2=debug
)

# Access results
print(f"Significant links: {result.n_significant}")
print(f"Runtime: {result.runtime:.3f}s")

# Iterate over discovered links
for link in result.significant_links:
    print(f"{link.source_var}(t-{link.tau}) -> {link.target_var}(t): "
          f"r={link.val:.3f}, p={link.pval:.2e}")

# Get parents of a specific variable
parents = result.get_parents(var=2)  # Returns [(source, tau, val, pval), ...]

# Raw matrices for further analysis
val_matrix = result.val_matrix      # Shape: (n_vars, tau_max+1, n_vars)
pval_matrix = result.pval_matrix
adj_matrix = result.adj_matrix      # Boolean adjacency
```

### Visualization

```python
from visualize import (
    plot_graph,           # Network graph of causal links
    plot_matrix,          # Heatmap of causal strengths
    plot_lag_functions,   # Correlation vs lag plots
    plot_time_series_graph,  # Temporal DAG
    print_parents,        # Text summary of parents
)

# Causal graph (requires networkx)
plot_graph(result, figsize=(12, 8), save_path='graph.png')

# Correlation matrix
plot_matrix(result, matrix_type='val')  # 'val', 'pval', or 'adj'

# Lag functions for each variable
plot_lag_functions(result, target_var=0)
```

### Loading Real Data

```python
import pandas as pd

# From CSV (columns = variables, rows = time points)
df = pd.read_csv('your_data.csv')
data = df.values.T  # Transpose to (n_vars, T)
var_names = list(df.columns)

result = run_pcmci(data, tau_max=5, alpha=0.05, var_names=var_names)
```

## C API

### Basic Example

```c
#include "pcmci.h"

int main() {
    // Create data: 5 variables, 1000 time points
    double* data = load_your_data();  // Row-major [n_vars x T]
    
    pcmci_dataframe_t* df = pcmci_dataframe_create(data, 5, 1000, 3);
    
    // Configure
    pcmci_config_t config = pcmci_default_config();
    config.tau_max = 3;
    config.alpha_level = 0.05;
    config.n_threads = 8;
    
    // Run
    pcmci_result_t* result = pcmci_run(df, &config);
    
    // Print results
    pcmci_graph_print(result->graph, NULL);
    
    // Cleanup
    pcmci_result_free(result);
    pcmci_dataframe_free(df);
    
    return 0;
}
```

### Key Functions

```c
// Configuration
pcmci_config_t pcmci_default_config(void);

// Data management
pcmci_dataframe_t* pcmci_dataframe_create(double* data, int32_t n_vars, 
                                           int32_t T, int32_t tau_max);
pcmci_dataframe_t* pcmci_dataframe_create_copy(const double* data, ...);
void pcmci_dataframe_free(pcmci_dataframe_t* df);

// Main algorithm
pcmci_result_t* pcmci_run(const pcmci_dataframe_t* df, const pcmci_config_t* config);
void pcmci_result_free(pcmci_result_t* result);

// Individual phases (for advanced use)
pcmci_graph_t* pcmci_skeleton(const pcmci_dataframe_t* df, const pcmci_config_t* config);
void pcmci_mci(const pcmci_dataframe_t* df, pcmci_graph_t* skeleton,
               const pcmci_config_t* config);

// Graph queries
bool pcmci_graph_has_link(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j);
double pcmci_graph_get_val(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j);
double pcmci_graph_get_pval(const pcmci_graph_t* g, int32_t i, int32_t tau, int32_t j);
```

## Algorithm Details

### PCMCI+ Overview

1. **Skeleton Discovery (PC-stable)**
   - Start with fully connected lagged graph
   - Iteratively test conditional independence: X ⊥ Y | Z
   - Remove links where p-value > α
   - Increase conditioning set size until no more removals

2. **MCI Phase**
   - For each remaining link (i, τ) → j
   - Condition on parents of both i and j
   - Compute partial correlation and p-value
   - Apply FDR correction

### Partial Correlation Test

For testing X ⊥ Y | Z:

1. Residualize X and Y on Z using least squares
2. Compute correlation of residuals
3. Convert to t-statistic: t = r × √(n-k-2) / √(1-r²)
4. Compute p-value from Student's t distribution

### Optimizations

- **Cholesky decomposition** for residualization (vs QR)
- **Lazy p-value**: Skip lgamma/betai when |t| < 1.5
- **Lock-free skeleton**: Thread-local removal buffers
- **Denormal flushing**: FTZ+DAZ for consistent performance
- **Aligned memory**: 64-byte alignment for SIMD

## Configuration

### Environment Variables (Windows)

```cmd
set MKL_NUM_THREADS=8
set OMP_NUM_THREADS=8
set KMP_AFFINITY=granularity=fine,compact,1,0
set KMP_BLOCKTIME=0
```

### Tuning Header

Include `pcmci_tuning.h` for automatic CPU/BLAS optimization:

```c
#include "pcmci_tuning.h"

int main() {
    pcmci_tuning_init(8, 1);  // 8 threads, verbose
    
    // ... your code ...
    
    pcmci_tuning_cleanup();
}
```

## Project Structure

```
PCMCI-Plus/
├── include/
│   ├── pcmci.h              # Main API header
│   ├── pcmci_types.h        # Type definitions
│   ├── pcmci_internal.h     # Internal functions
│   └── pcmci_tuning.h       # CPU/BLAS tuning
├── src/
│   ├── dataframe.c          # Data management
│   ├── parcorr.c            # Partial correlation tests
│   ├── skeleton.c           # PC-stable skeleton discovery
│   ├── mci.c                # MCI phase
│   ├── graph.c              # Graph operations
│   ├── fdr.c                # FDR correction
│   ├── robust.c             # Spearman, winsorization
│   └── pcmci.c              # Main algorithm
├── python/
│   ├── pcmci.py             # High-level Python API
│   ├── pcmci_bindings.py    # ctypes wrapper
│   ├── visualize.py         # Plotting utilities
│   ├── pcmci_demo.ipynb     # Jupyter notebook demo
│   └── test_pcmci.py        # Test script
├── examples/
│   ├── example_basic.c      # Basic C example
│   ├── example_synthetic.c  # Synthetic data example
│   └── benchmark_latency.c  # Performance benchmark
├── CMakeLists.txt
├── pcmci.def                # Windows DLL exports
└── README.md
```

## References

**PCMCI+ Algorithm:**
> Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. *Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 124:1388-1397.

**Original PCMCI:**
> Runge, J., et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5(11), eaau4996.

**Tigramite (Python reference implementation):**
> https://github.com/jakobrunge/tigramite

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Intel MKL for high-performance linear algebra
- Jakob Runge for the original PCMCI+ algorithm and Tigramite implementation
- OpenMP for parallelization support

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Areas for Future Work

- [ ] CMI (Conditional Mutual Information) test for nonlinear dependencies
- [ ] GPDC (Gaussian Process Distance Correlation) test
- [ ] Block bootstrap confidence intervals
- [ ] GPU acceleration (cuBLAS)
- [ ] Time-varying causal discovery