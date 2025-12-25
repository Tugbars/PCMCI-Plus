#!/usr/bin/env python3
"""
Simple test script for PCMCI+ Python bindings.

Run from the python/ directory:
    python test_pcmci.py

Or from project root:
    python python/test_pcmci.py

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
import sys
import os

# Ensure we can find the modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import PCMCI modules
from pcmci import PCMCI, run_pcmci, version
from visualize import plot_graph, plot_matrix, print_parents

def generate_test_data(n_vars=4, T=1000, seed=42):
    """Generate simple VAR(1) data with known causal structure."""
    np.random.seed(seed)
    data = np.zeros((n_vars, T))
    
    for t in range(1, T):
        # X0 -> X0 (AR)
        data[0, t] = 0.6 * data[0, t-1] + 0.5 * np.random.randn()
        # X0 -> X1
        data[1, t] = 0.5 * data[1, t-1] + 0.4 * data[0, t-1] + 0.5 * np.random.randn()
        # X1 -> X2
        data[2, t] = 0.5 * data[2, t-1] + 0.4 * data[1, t-1] + 0.5 * np.random.randn()
        # X2 -> X3
        data[3, t] = 0.5 * data[3, t-1] + 0.4 * data[2, t-1] + 0.5 * np.random.randn()
    
    return data

def main():
    print("=" * 60)
    print("PCMCI+ Python Bindings Test")
    print("=" * 60)
    print(f"Library version: {version()}")
    print()
    
    # Generate test data
    print("Generating test data...")
    data = generate_test_data(n_vars=4, T=1000)
    var_names = ['X0', 'X1', 'X2', 'X3']
    print(f"  Shape: {data.shape}")
    print(f"  True structure: X0→X1→X2→X3 (chain, lag 1)")
    print()
    
    # Run PCMCI+
    print("Running PCMCI+...")
    pcmci = PCMCI(data, tau_max=3, var_names=var_names)
    result = pcmci.run(alpha=0.05, verbosity=0)
    
    print(f"\nResults:")
    print(f"  Significant links: {result.n_significant}")
    print(f"  Runtime: {result.runtime*1000:.1f} ms")
    print()
    
    # Show significant links
    print("Discovered causal links:")
    for link in result.significant_links:
        arrow = "→" if link.tau > 0 else "↔"
        print(f"  {var_names[link.source_var]}(t-{link.tau}) {arrow} {var_names[link.target_var]}(t): "
              f"r={link.val:.3f}, p={link.pval:.2e}")
    print()
    
    # Show parents
    print_parents(result)
    
    # Try to save visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        print("\nGenerating visualizations...")
        plot_graph(result, save_path='test_graph.png')
        print("  Saved: test_graph.png")
        
        plot_matrix(result, save_path='test_matrix.png')
        print("  Saved: test_matrix.png")
        
    except ImportError:
        print("\nSkipping visualizations (matplotlib not installed)")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())