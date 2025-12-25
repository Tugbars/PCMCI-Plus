"""
PCMCI+ Python Bindings

Fast causal discovery for time series data.

Example:
    >>> import numpy as np
    >>> from pcmci_plus import PCMCI, plot_graph
    >>> 
    >>> data = np.random.randn(5, 1000)
    >>> pcmci = PCMCI(data, tau_max=3)
    >>> result = pcmci.run(alpha=0.05)
    >>> print(result)
    >>> plot_graph(result)

SPDX-License-Identifier: GPL-3.0-or-later
"""

# Handle both package import and direct execution
try:
    from .pcmci import (
        PCMCI,
        PCMCIResult,
        Link,
        run_pcmci,
        version,
    )

    from .visualize import (
        plot_graph,
        plot_time_series_graph,
        plot_matrix,
        plot_lag_functions,
        print_summary,
        print_parents,
    )
except ImportError:
    from pcmci import (
        PCMCI,
        PCMCIResult,
        Link,
        run_pcmci,
        version,
    )

    from visualize import (
        plot_graph,
        plot_time_series_graph,
        plot_matrix,
        plot_lag_functions,
        print_summary,
        print_parents,
    )

try:
    __version__ = version()
except:
    __version__ = "0.1.0"

__all__ = [
    # Core
    'PCMCI',
    'PCMCIResult',
    'Link',
    'run_pcmci',
    'version',
    # Visualization
    'plot_graph',
    'plot_time_series_graph',
    'plot_matrix', 
    'plot_lag_functions',
    'print_summary',
    'print_parents',
]