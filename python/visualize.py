"""
Visualization utilities for PCMCI+ causal graphs.

Requires: matplotlib, networkx (optional for graph layouts)

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
from typing import Optional, List, Tuple, Dict

# Handle both package import and direct execution
try:
    from .pcmci import PCMCIResult, Link
except ImportError:
    from pcmci import PCMCIResult, Link

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_graph(
    result: PCMCIResult,
    alpha: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 8),
    node_size: int = 2000,
    font_size: int = 10,
    edge_width_scale: float = 3.0,
    show_values: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot causal graph using networkx.
    
    Parameters
    ----------
    result : PCMCIResult
        Result from PCMCI.run()
    alpha : float, optional
        Override significance threshold for display
    figsize : tuple
        Figure size
    node_size : int
        Size of nodes
    font_size : int
        Font size for labels
    edge_width_scale : float
        Scale factor for edge widths (based on |correlation|)
    show_values : bool
        Show correlation values on edges
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError("Please install matplotlib and networkx: pip install matplotlib networkx")
    
    alpha = alpha or result.alpha
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for each (var, lag) combination
    # Layout: columns are variables, rows are time lags
    pos = {}
    labels = {}
    
    var_names = result.var_names or [f"X{i}" for i in range(result.n_vars)]
    
    # Add nodes
    for j in range(result.n_vars):
        # Current time node
        node_id = (j, 0)
        G.add_node(node_id)
        pos[node_id] = (j, 0)
        labels[node_id] = f"{var_names[j]}(t)"
    
    # Add lagged nodes only if they have outgoing edges
    for link in result.significant_links:
        if link.tau > 0:
            node_id = (link.source_var, link.tau)
            if node_id not in G.nodes:
                G.add_node(node_id)
                pos[node_id] = (link.source_var, -link.tau)
                labels[node_id] = f"{var_names[link.source_var]}(t-{link.tau})"
    
    # Add edges
    edge_colors = []
    edge_widths = []
    edge_labels = {}
    
    for link in result.significant_links:
        src = (link.source_var, link.tau)
        tgt = (link.target_var, 0)
        
        G.add_edge(src, tgt)
        
        # Color based on sign
        edge_colors.append('blue' if link.val > 0 else 'red')
        edge_widths.append(abs(link.val) * edge_width_scale + 0.5)
        
        if show_values:
            edge_labels[(src, tgt)] = f"{link.val:.2f}"
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', 
                           edgecolors='black', linewidths=2, ax=ax)
    
    # Draw edges
    if edge_colors:
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                               arrows=True, arrowsize=20, 
                               connectionstyle="arc3,rad=0.1", ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)
    
    if show_values and edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=font_size-2, ax=ax)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='Positive'),
        Line2D([0], [0], color='red', linewidth=2, label='Negative'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(title or f"Causal Graph (α={alpha}, {result.n_significant} links)")
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_time_series_graph(
    result: PCMCIResult,
    figsize: Tuple[int, int] = (14, 6),
    n_time_steps: int = 4,
    node_size: int = 1500,
    font_size: int = 9,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot time series DAG showing temporal structure.
    
    Each variable is shown at multiple time points, with arrows showing
    causal relationships across time.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError("Please install matplotlib and networkx")
    
    G = nx.DiGraph()
    pos = {}
    labels = {}
    
    var_names = result.var_names or [f"X{i}" for i in range(result.n_vars)]
    n_vars = result.n_vars
    
    # Create nodes for each (var, time) pair
    for t in range(n_time_steps):
        for v in range(n_vars):
            node = (v, t)
            G.add_node(node)
            pos[node] = (t, n_vars - 1 - v)  # Time on x-axis, vars on y-axis
            labels[node] = f"{var_names[v]}\nt-{n_time_steps-1-t}" if t < n_time_steps-1 else f"{var_names[v]}\nt"
    
    # Add edges based on discovered links
    edge_colors = []
    edge_widths = []
    
    for link in result.significant_links:
        # Add edge for each applicable time step
        for t in range(link.tau, n_time_steps):
            src = (link.source_var, t - link.tau)
            tgt = (link.target_var, t)
            G.add_edge(src, tgt)
            edge_colors.append('blue' if link.val > 0 else 'red')
            edge_widths.append(abs(link.val) * 2 + 0.5)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightgreen',
                           edgecolors='black', linewidths=1.5, ax=ax)
    
    if edge_colors:
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                               arrows=True, arrowsize=15, ax=ax)
    
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)
    
    ax.set_title(title or f"Time Series DAG ({result.n_significant} causal links)")
    ax.axis('off')
    
    # Add time axis label
    ax.annotate('Time →', xy=(0.5, -0.05), xycoords='axes fraction', 
                fontsize=12, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_matrix(
    result: PCMCIResult,
    matrix_type: str = 'val',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    show_values: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot correlation or p-value matrix as heatmap.
    
    Parameters
    ----------
    result : PCMCIResult
    matrix_type : str
        'val' for correlation values, 'pval' for p-values, 'adj' for adjacency
    figsize : tuple
    cmap : str
        Colormap name
    show_values : bool
        Annotate cells with values
    title : str, optional
    save_path : str, optional
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install matplotlib")
    
    var_names = result.var_names or [f"X{i}" for i in range(result.n_vars)]
    
    # Select matrix and create combined view (sum over tau for visualization)
    if matrix_type == 'val':
        # Show strongest link for each (i,j) pair across all lags
        matrix = np.zeros((result.n_vars, result.n_vars))
        for i in range(result.n_vars):
            for j in range(result.n_vars):
                vals = result.val_matrix[i, :, j]
                idx = np.argmax(np.abs(vals))
                matrix[i, j] = vals[idx] if result.adj_matrix[i, idx, j] else 0
        vmin, vmax = -1, 1
        fmt = '.2f'
        title = title or "Causal Strength (strongest lag)"
    elif matrix_type == 'pval':
        # Show minimum p-value
        matrix = np.min(result.pval_matrix, axis=1)
        vmin, vmax = 0, 1
        fmt = '.3f'
        cmap = 'viridis_r'
        title = title or "P-values (minimum across lags)"
    else:  # adj
        matrix = np.any(result.adj_matrix, axis=1).astype(float)
        vmin, vmax = 0, 1
        fmt = '.0f'
        cmap = 'Greys'
        title = title or "Adjacency (any lag)"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    
    # Add labels
    ax.set_xticks(range(result.n_vars))
    ax.set_yticks(range(result.n_vars))
    ax.set_xticklabels([f"{v}(t)" for v in var_names])
    ax.set_yticklabels([f"{v}(t-τ)" for v in var_names])
    
    ax.set_xlabel("Target (effect)")
    ax.set_ylabel("Source (cause)")
    
    # Annotate
    if show_values:
        for i in range(result.n_vars):
            for j in range(result.n_vars):
                text_color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha='center', va='center',
                       color=text_color, fontsize=9)
    
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_lag_functions(
    result: PCMCIResult,
    target_var: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot correlation as function of lag for each variable pair.
    
    Parameters
    ----------
    result : PCMCIResult
    target_var : int, optional
        If specified, only show links to this target variable
    figsize : tuple
    title : str, optional
    save_path : str, optional
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install matplotlib")
    
    var_names = result.var_names or [f"X{i}" for i in range(result.n_vars)]
    
    targets = [target_var] if target_var is not None else range(result.n_vars)
    n_plots = len(list(targets))
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    lags = np.arange(result.tau_max + 1)
    
    for idx, j in enumerate(targets):
        ax = axes[idx]
        
        for i in range(result.n_vars):
            vals = result.val_matrix[i, :, j]
            sig = result.adj_matrix[i, :, j]
            
            # Plot line
            ax.plot(lags, vals, 'o-', label=var_names[i], alpha=0.7)
            
            # Mark significant points
            ax.scatter(lags[sig], vals[sig], s=100, marker='*', zorder=5)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag τ')
        ax.set_ylabel('Partial Correlation')
        ax.set_title(f"Links to {var_names[j]}(t)")
        ax.legend(loc='best', fontsize=8)
        ax.set_xticks(lags)
    
    plt.suptitle(title or "Lag Functions (★ = significant)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


# =============================================================================
# Summary Printing
# =============================================================================

def print_summary(result: PCMCIResult):
    """Print formatted summary of PCMCI+ results."""
    print(result.summary())


def print_parents(result: PCMCIResult):
    """Print parents for each variable."""
    var_names = result.var_names or [f"X{i}" for i in range(result.n_vars)]
    
    print("Parents of each variable:")
    print("=" * 50)
    
    for j in range(result.n_vars):
        parents = result.get_parents(j)
        print(f"\n{var_names[j]}(t):")
        if parents:
            for i, tau, val, pval in parents:
                print(f"  ← {var_names[i]}(t-{tau}): r={val:.3f}, p={pval:.2e}")
        else:
            print("  (no significant parents)")


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'plot_graph',
    'plot_time_series_graph', 
    'plot_matrix',
    'plot_lag_functions',
    'print_summary',
    'print_parents',
]