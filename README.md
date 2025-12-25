# PCMCI+ : High-Performance Causal Discovery for Time Series

A production-grade C implementation of the PCMCI+ algorithm with Python bindings. Sub-millisecond causal graph discovery with SIMD, OpenMP, and Intel MKL acceleration.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

<img width="1389" height="590" alt="Nonlinear Dependency Detection" src="https://github.com/user-attachments/assets/95001c36-9a34-4bc4-9726-7e87ad2f1cd1" />

<img width="2379" height="1075" alt="ex2_nonlinear" src="https://github.com/user-attachments/assets/4d74da39-5445-46af-a528-51e412fcedca" />


PCMCI+ (Runge, 2020) discovers causal relationships in multivariate time series by combining:

- **PC algorithm** for skeleton discovery (removes spurious correlations)
- **MCI (Momentary Conditional Independence)** for robust causal strength estimation

This implementation achieves **100x speedup** over the reference Python implementation (Tigramite) while providing multiple independence tests for both linear and nonlinear dependencies.

## Features

### Independence Tests

| Test | Function | Detects | Speed | Use Case |
|------|----------|---------|-------|----------|
| **Partial Correlation** | `parcorr_test()` | Linear | < 1ms | Default, fast screening |
| **CMI** | `cmi_test()` | Nonlinear | ~1ms | KSG estimator, any dependency |
| **Distance Correlation** | `dcor_test()` | Any | ~10ms | Zero iff independent |
| **GPDC** | `gpdc_test()` | Nonlinear conditional | ~10s | GP regression + dCor |

### Performance Optimizations

- **Intel MKL / OpenBLAS** acceleration
- **OpenMP** parallelization (skeleton discovery, CI tests)
- **Cholesky fast-path** for residualization (2-4x faster than QR)
- **Lock-free parallel skeleton** eliminates thread contention
- **Lazy p-value computation** skips expensive calculations for weak correlations
- **64-byte aligned memory** for SIMD operations

### Benchmarks

| Variables | Samples | Runtime (8 cores) |
|-----------|---------|-------------------|
| 5 | 500 | ~1 ms |
| 5 | 5000 | ~25 ms |
| 10 | 1000 | ~180 ms |
| 20 | 500 | ~110 ms |

*Intel i9-14900KF, MKL 2025, τ_max=3*

## Installation

### Prerequisites

- **CMake** 3.16+
- **C11 compiler** (GCC, Clang, MSVC, or Intel ICX)
- **Intel oneAPI** (MKL + Compiler) or **OpenBLAS**
- **Python 3.8+** with NumPy (for bindings)

### Windows (Intel oneAPI)

```cmd
# Open Intel oneAPI Command Prompt
cd PCMCI-Plus
mkdir build && cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build .
copy pcmci.dll ..\python\
```

### Linux

```bash
# With Intel MKL
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# With OpenBLAS
sudo apt install libopenblas-dev liblapacke-dev
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

## Usage

### Python

See `python/market_analysis.ipynb` for a complete walkthrough with real market data, or `python/visual_examples.py` for visual demonstrations.

```python
import numpy as np
import pcmci

# Causal discovery
data = np.random.randn(5, 1000)  # (n_vars, T)
result = pcmci.run_pcmci(data, tau_max=3, alpha=0.05)

for link in result.significant_links:
    print(f"X{link.source_var}(t-{link.tau}) → X{link.target_var}(t): r={link.val:.3f}")

# Independence tests
r, p = pcmci.parcorr_test(X, Y, Z)           # Linear, conditional
result = pcmci.cmi_test(X, Y, Z, n_perm=100) # Nonlinear, conditional
dc = pcmci.dcor(X, Y)                         # Distance correlation
```

### C

See `examples/` folder for complete examples:

| Example | Description |
|---------|-------------|
| `example_basic.c` | Minimal PCMCI+ usage |
| `example_synthetic.c` | Known ground truth validation |
| `example_cmi.c` | CMI nonlinear detection |
| `example_gpdc.c` | GPDC conditional independence |

```c
#include "pcmci.h"

pcmci_dataframe_t* df = pcmci_dataframe_create(data, n_vars, T, tau_max);
pcmci_config_t config = pcmci_default_config();
config.alpha_level = 0.05;

pcmci_result_t* result = pcmci_run(df, &config);
// ... use result->graph, result->n_links ...
pcmci_result_free(result);
pcmci_dataframe_free(df);
```

## Algorithm

### What PCMCI+ Does

PCMCI+ answers the question: **"Which variables causally influence which others, and at what time lag?"**

Given a multivariate time series (e.g., 5 sensors measured over 1000 time points), PCMCI+ outputs a **causal graph** showing directed edges like:

```
Temperature(t-2) → Pressure(t)     strength=0.45, p<0.001
Humidity(t-1) → Temperature(t)     strength=0.32, p<0.01
```

This is fundamentally different from correlation matrices, which only show undirected associations and cannot distinguish:
- **Causation** from correlation
- **Direct** effects from indirect effects  
- **True** relationships from confounded ones

### The Two-Phase Approach

**Phase 1: Skeleton Discovery (PC-stable algorithm)**

Start with a fully connected graph where every variable at every lag could potentially cause every other variable. Then systematically remove false links:

```
Initial: All possible links exist
         X(t-1)→Y, X(t-2)→Y, X(t-3)→Y, Z(t-1)→Y, Z(t-2)→Y, ...

Step 1:  Test each link unconditionally
         Remove if X(t-τ) ⊥ Y(t)  [p > α]

Step 2:  Test remaining links conditioning on 1 variable
         Remove if X(t-τ) ⊥ Y(t) | Z(t-1)  [p > α]

Step 3:  Condition on 2 variables, then 3, ...
         Continue until no more links removed
         
Result:  Sparse skeleton with only potentially causal links
```

The "PC-stable" variant ensures results don't depend on variable ordering.

**Phase 2: Momentary Conditional Independence (MCI)**

The skeleton may still contain some false positives. MCI refines each link by conditioning on the **parents of both endpoints**:

```
For link X(t-τ) → Y(t):
    
    Parents of X: {W(t-1), V(t-2)}     # Things that cause X
    Parents of Y: {X(t-τ), Z(t-1)}     # Things that cause Y (excluding X)
    
    Test: X(t-τ) ⊥ Y(t) | {W(t-1), V(t-2), Z(t-1)}
    
    This removes confounding from:
    - Common causes of X and Y
    - Indirect paths through other variables
```

### Why This Works

**Problem 1: Confounding**
```
True structure:    Z → X
                   Z → Y
                   (no X→Y link)

Correlation sees:  X ↔ Y (spurious!)
PCMCI+ sees:       X ⊥ Y | Z (correct: no direct link)
```

**Problem 2: Autocorrelation**
```
Time series are serially correlated: X(t) ≈ X(t-1) ≈ X(t-2)

Naive correlation: Everything correlates with everything
PCMCI+: Conditions on past values, tests only "new" information
```

**Problem 3: Indirect Effects**
```
True structure:    A(t-1) → B(t) → C(t+1)

Correlation sees:  A correlates with C
PCMCI+ sees:       A(t-1) → B(t) → C(t+1), no direct A→C
```

**Problem 4: Multiple Lags**
```
Question: Does X affect Y? At what delay?

PCMCI+ tests each lag separately:
    X(t-1) → Y(t): p=0.73  (not significant)
    X(t-2) → Y(t): p=0.002 (significant!) 
    X(t-3) → Y(t): p=0.45  (not significant)
    
Answer: X causes Y with a 2-step delay
```

### Independence Tests

The core operation is testing **conditional independence**: X ⊥ Y | Z

| Test | Method | Strengths | Limitations |
|------|--------|-----------|-------------|
| **Partial Correlation** | Residualize X,Y on Z, correlate residuals | Fast, well-understood | Linear only |
| **CMI** | k-NN entropy estimation (KSG) | Detects nonlinear, consistent | Slower, needs more data |
| **Distance Correlation** | Centered distance matrices | Detects any dependence | No conditional version |
| **GPDC** | GP regression + distance correlation | Nonlinear conditional | O(n³), slow |

**When to use which:**
- Start with partial correlation (default) — fast, works for most cases
- Use CMI if you suspect nonlinear relationships (Y = X², Y = sin(X))
- Use GPDC for nonlinear confounding (rare, expensive)

### Behavior and Parameters

**`tau_max` (maximum lag)**
- How far back in time to look for causes
- Higher = more thorough but slower (tests grow as τ²)
- Domain knowledge helps: daily data with weekly cycles → τ_max=7

**`alpha` (significance level)**
- Threshold for keeping/removing links
- Lower α = fewer false positives, may miss weak true links
- Higher α = more links detected, more false positives
- Default 0.05 is usually reasonable; use 0.01 for stricter control

**`max_cond_dim` (maximum conditioning set size)**
- Limits how many variables to condition on simultaneously
- -1 = automatic (up to number of parents)
- Lower values = faster but may miss some confounders
- Usually not needed unless you have many variables (>20)

**Output interpretation:**
```python
# A significant link means:
# "After controlling for all other potential causes,
#  X(t-τ) still predicts Y(t) beyond chance"

for link in result.significant_links:
    # link.val: Partial correlation [-1, 1]
    #   +0.5 = strong positive effect
    #   -0.3 = moderate negative effect
    #   ~0   = no direct effect
    
    # link.pval: Probability this is a false positive
    #   <0.001 = very confident
    #   <0.05  = significant
    #   >0.05  = not significant (link removed)
```

### Use Cases

**Finance & Economics**
- Volatility spillovers between markets (does US stress predict EU stress?)
- Lead-lag relationships for trading signals
- Macroeconomic causal chains (rates → inflation → employment)

**Neuroscience**
- Effective connectivity between brain regions
- Neural information flow from fMRI/EEG
- Stimulus → response causal pathways

**Climate Science**
- Teleconnections (El Niño → regional weather patterns)
- Attribution of extreme events
- Ocean-atmosphere interactions

**Engineering & IoT**
- Root cause analysis in sensor networks
- Fault propagation in industrial systems
- Predictive maintenance (which sensor predicts failures?)

**Biology**
- Gene regulatory network inference
- Metabolic pathway discovery
- Ecological food web dynamics

### Limitations

1. **Assumes causal sufficiency**: All common causes must be measured. Unmeasured confounders can create false links.

2. **Assumes stationarity**: The causal structure shouldn't change over time. For regime changes, use rolling windows.

3. **Assumes no instantaneous effects at sub-sampling resolution**: If X→Y happens faster than your sampling rate, it appears as lag-0 (contemporaneous).

4. **Faithfulness assumption**: Statistical independence implies causal independence. Pathological cancellations can violate this (rare in practice).

5. **Sample size requirements**: Need enough data for reliable conditional independence tests. Rule of thumb: T > 50 × (number of variables) × τ_max.

### Comparison with Other Methods

| Method | Handles Lags | Handles Confounding | Nonlinear | Speed |
|--------|--------------|---------------------|-----------|-------|
| Correlation | ❌ | ❌ | ❌ | ⚡ |
| Granger Causality | ✅ | ❌ | ❌ | ⚡ |
| VAR models | ✅ | ❌ | ❌ | ⚡ |
| Transfer Entropy | ✅ | Partial | ✅ | 🐢 |
| CCM (Convergent Cross Mapping) | ✅ | ✅ | ✅ | 🐢 |
| **PCMCI+** | ✅ | ✅ | Optional | ⚡ |

PCMCI+ combines the speed of linear methods with the robustness of constraint-based causal discovery.

### Why Not Just Correlation?

| Problem | Correlation | PCMCI+ |
|---------|-------------|--------|
| Confounding (X ← Z → Y) | Shows spurious X↔Y | Correctly finds X ⊥ Y \| Z |
| Autocorrelation | Inflated significance | Conditions on past values |
| Nonlinear dependencies | Misses Y = X² | CMI/dCor detect it |
| Multiple lags | Which lag matters? | Tests each τ separately |

## Project Structure

```
PCMCI-Plus/
├── include/
│   ├── pcmci.h              # Main API
│   ├── pcmci_cmi.h          # CMI (Conditional Mutual Information)
│   ├── pcmci_gpdc.h         # GPDC (GP Distance Correlation)
│   └── pcmci_internal.h     # Internal functions
├── src/
│   ├── pcmci.c              # Main algorithm
│   ├── parcorr.c            # Partial correlation
│   ├── skeleton.c           # PC-stable skeleton
│   ├── mci.c                # MCI phase
│   ├── cmi.c                # CMI with KD-tree
│   └── gpdc.c               # GPDC with GP regression
├── python/
│   ├── pcmci.py             # High-level API
│   ├── pcmci_bindings.py    # ctypes bindings
│   ├── visualize.py         # Plotting utilities
│   ├── market_analysis.ipynb    # Real data demo
│   ├── extended_analysis.ipynb  # Advanced analysis
│   └── visual_examples.py   # Showcase visualizations
├── examples/
│   ├── example_basic.c
│   ├── example_synthetic.c
│   ├── example_cmi.c
│   └── example_gpdc.c
├── CMakeLists.txt
└── pcmci.def                # Windows DLL exports
```

## Configuration

### Environment Variables

```bash
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0
```

### Runtime Options

```python
result = pcmci.run_pcmci(
    data,
    tau_max=5,            # Maximum lag
    alpha=0.05,           # Significance level
    var_names=['A','B'],  # Variable names
    use_spearman=True,    # Robust to outliers
    n_threads=0,          # 0 = auto-detect
    fdr_method=1,         # Benjamini-Hochberg
)
```

## References

**PCMCI+ Algorithm:**
> Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. *UAI 2020*, PMLR 124:1388-1397.

**CMI Estimator (KSG):**
> Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

**Distance Correlation:**
> Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and testing dependence by correlation of distances. *Annals of Statistics*, 35(6), 2769-2794.

**Tigramite (Reference Implementation):**
> https://github.com/jakobrunge/tigramite

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Acknowledgments

- Jakob Runge for the PCMCI+ algorithm and Tigramite
- Intel MKL for high-performance linear algebra
- OpenMP for parallelization
