#!/usr/bin/env python3
"""
Example: PCMCI+ Python bindings demo

This script demonstrates:
1. Creating synthetic data with known causal structure
2. Running PCMCI+ analysis
3. Visualizing results

Requirements:
    pip install numpy matplotlib networkx

Usage:
    cd python
    python example_demo.py
    
SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
import sys
import os

# Ensure we can find the modules when running directly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pcmci import PCMCI, run_pcmci, version
from visualize import plot_graph, plot_matrix, plot_lag_functions, print_parents

# =============================================================================
# Generate Synthetic Data
# =============================================================================

def generate_var_data(n_vars: int = 4, T: int = 1000, seed: int = 42):
    """
    Generate synthetic VAR data with known causal structure.
    
    True causal structure (lag 1):
        X0 -> X0 (autoregressive)
        X0 -> X1 (causal)
        X1 -> X1 (autoregressive)  
        X1 -> X2 (causal, lag 2)
        X2 -> X2 (autoregressive)
        X2 -> X3 (causal)
        X3 -> X3 (autoregressive)
    """
    np.random.seed(seed)
    
    data = np.zeros((n_vars, T))
    noise_std = 0.5
    
    # Coefficients
    auto_coef = 0.6  # Autoregressive
    cross_coef = 0.4  # Cross-variable
    
    for t in range(2, T):
        # X0: pure AR(1)
        data[0, t] = auto_coef * data[0, t-1] + noise_std * np.random.randn()
        
        # X1: AR(1) + X0(t-1)
        data[1, t] = auto_coef * data[1, t-1] + cross_coef * data[0, t-1] + noise_std * np.random.randn()
        
        # X2: AR(1) + X1(t-2)
        data[2, t] = auto_coef * data[2, t-1] + cross_coef * data[1, t-2] + noise_std * np.random.randn()
        
        # X3: AR(1) + X2(t-1)
        data[3, t] = auto_coef * data[3, t-1] + cross_coef * data[2, t-1] + noise_std * np.random.randn()
    
    return data


def generate_market_data(T: int = 1000, seed: int = 42):
    """
    Generate synthetic market data simulating:
    - SPY: Stock market index
    - BTC: Cryptocurrency
    - TLT: Bond ETF
    - VIX: Volatility index
    
    Causal structure:
        VIX(t-1) -> SPY(t)   [negative: high vol → lower returns]
        SPY(t-1) -> BTC(t)   [positive: stocks lead crypto]
        VIX(t-1) -> TLT(t)   [positive: flight to safety]
        SPY(t-1) -> VIX(t)   [negative: drops cause vol spikes]
    """
    np.random.seed(seed)
    
    n_vars = 4
    data = np.zeros((n_vars, T))
    var_names = ['SPY', 'BTC', 'TLT', 'VIX']
    
    for t in range(1, T):
        # SPY: autoregressive + inverse VIX effect
        data[0, t] = 0.5 * data[0, t-1] - 0.3 * data[3, t-1] + 0.5 * np.random.randn()
        
        # BTC: follows SPY with lag, higher noise
        data[1, t] = 0.4 * data[1, t-1] + 0.35 * data[0, t-1] + 0.8 * np.random.randn()
        
        # TLT: flight to safety (positive VIX correlation)
        data[2, t] = 0.6 * data[2, t-1] + 0.25 * data[3, t-1] + 0.4 * np.random.randn()
        
        # VIX: mean-reverting, spikes on SPY drops
        data[3, t] = 0.7 * data[3, t-1] - 0.4 * data[0, t-1] + 0.6 * np.random.randn()
    
    return data, var_names


# =============================================================================
# Main Demo
# =============================================================================

def main():
    print("=" * 60)
    print("PCMCI+ Python Bindings Demo")
    print("=" * 60)
    
    # Check library version
    print(f"\nLibrary version: {version()}")
    
    # ---------------------------------------------------------------------
    # Demo 1: Simple VAR data
    # ---------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Demo 1: Synthetic VAR Data (4 variables)")
    print("-" * 60)
    
    data = generate_var_data(n_vars=4, T=1000)
    var_names = ['X0', 'X1', 'X2', 'X3']
    
    print(f"Data shape: {data.shape} (n_vars, T)")
    print(f"True structure: X0→X1(lag 1), X1→X2(lag 2), X2→X3(lag 1), plus AR(1)")
    
    # Run PCMCI+
    pcmci = PCMCI(data, tau_max=3, var_names=var_names)
    result = pcmci.run(alpha=0.05, verbosity=0)
    
    print(f"\nResults:")
    print(result)
    
    # Print parents
    print("\n")
    print_parents(result)
    
    # ---------------------------------------------------------------------
    # Demo 2: Market data
    # ---------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Demo 2: Synthetic Market Data")
    print("-" * 60)
    
    data, var_names = generate_market_data(T=1000)
    
    print(f"Variables: {var_names}")
    print("True structure: VIX→SPY(-), SPY→BTC(+), VIX→TLT(+), SPY→VIX(-)")
    
    result = run_pcmci(data, tau_max=3, alpha=0.05, var_names=var_names)
    
    print(f"\nResults:")
    print(result)
    
    # ---------------------------------------------------------------------
    # Demo 3: Visualizations (if matplotlib available)
    # ---------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        print("\n" + "-" * 60)
        print("Demo 3: Generating Visualizations")
        print("-" * 60)
        
        # Causal graph
        fig1, ax1 = plot_graph(result, save_path='pcmci_graph.png')
        print("Saved: pcmci_graph.png")
        
        # Correlation matrix
        fig2, ax2 = plot_matrix(result, matrix_type='val', save_path='pcmci_matrix.png')
        print("Saved: pcmci_matrix.png")
        
        # Lag functions
        fig3, ax3 = plot_lag_functions(result, save_path='pcmci_lags.png')
        print("Saved: pcmci_lags.png")
        
        plt.close('all')
        
    except ImportError:
        print("\nSkipping visualizations (matplotlib not installed)")
        print("Install with: pip install matplotlib networkx")
    
    # ---------------------------------------------------------------------
    # Demo 4: Performance benchmark
    # ---------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Demo 4: Performance Benchmark")
    print("-" * 60)
    
    import time
    
    for n_vars in [5, 10, 20]:
        data = np.random.randn(n_vars, 500)
        
        start = time.perf_counter()
        result = run_pcmci(data, tau_max=3, alpha=0.05)
        elapsed = time.perf_counter() - start
        
        print(f"  {n_vars} vars, 500 samples: {elapsed*1000:.1f} ms ({result.n_significant} links)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()