"""
High-level Pythonic API for PCMCI+ causal discovery.

Example usage:
    import numpy as np
    from pcmci import PCMCI
    
    # Generate or load data: shape (n_vars, T)
    data = np.random.randn(5, 1000)
    
    # Run PCMCI+
    pcmci = PCMCI(data, tau_max=3)
    result = pcmci.run(alpha=0.05)
    
    # Access results
    print(result.significant_links)
    result.plot()

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import ctypes
from ctypes import byref, POINTER, c_double, c_char_p

# Handle both package import and direct execution
try:
    from . import pcmci_bindings as _bind
except ImportError:
    import pcmci_bindings as _bind

# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class Link:
    """A causal link from (i, tau) -> j."""
    source_var: int
    tau: int
    target_var: int
    val: float
    pval: float
    
    def __repr__(self):
        return f"X{self.source_var}(t-{self.tau}) --> X{self.target_var}(t): val={self.val:.4f}, p={self.pval:.2e}"

@dataclass
class PCMCIResult:
    """Result of PCMCI+ analysis."""
    
    # Core results
    val_matrix: np.ndarray      # Shape: (n_vars, tau_max+1, n_vars)
    pval_matrix: np.ndarray     # Shape: (n_vars, tau_max+1, n_vars)
    adj_matrix: np.ndarray      # Shape: (n_vars, tau_max+1, n_vars), boolean
    
    # Metadata
    n_vars: int
    tau_max: int
    alpha: float
    n_significant: int
    runtime: float
    var_names: Optional[List[str]] = None
    
    @property
    def significant_links(self) -> List[Link]:
        """Get list of significant causal links."""
        links = []
        for i in range(self.n_vars):
            for tau in range(self.tau_max + 1):
                for j in range(self.n_vars):
                    if self.adj_matrix[i, tau, j]:
                        links.append(Link(
                            source_var=i,
                            tau=tau,
                            target_var=j,
                            val=self.val_matrix[i, tau, j],
                            pval=self.pval_matrix[i, tau, j]
                        ))
        return links
    
    def get_parents(self, var: int) -> List[Tuple[int, int, float, float]]:
        """Get parents of a variable: list of (source_var, tau, val, pval)."""
        parents = []
        for i in range(self.n_vars):
            for tau in range(self.tau_max + 1):
                if self.adj_matrix[i, tau, var]:
                    parents.append((i, tau, self.val_matrix[i, tau, var], self.pval_matrix[i, tau, var]))
        return parents
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"PCMCI+ Result",
            f"=============",
            f"Variables: {self.n_vars}",
            f"Max lag: {self.tau_max}",
            f"Alpha: {self.alpha}",
            f"Significant links: {self.n_significant}",
            f"Runtime: {self.runtime:.3f}s",
            f"",
            f"Significant Causal Links:",
        ]
        for link in self.significant_links:
            src = self.var_names[link.source_var] if self.var_names else f"X{link.source_var}"
            tgt = self.var_names[link.target_var] if self.var_names else f"X{link.target_var}"
            lines.append(f"  {src}(t-{link.tau}) --> {tgt}(t): val={link.val:.4f}, p={link.pval:.2e}")
        return "\n".join(lines)
    
    def __repr__(self):
        return self.summary()
    
    def to_tigramite_format(self) -> Dict[int, List[Tuple[Tuple[int, int], float, float]]]:
        """Convert to tigramite-compatible format for comparison."""
        result = {j: [] for j in range(self.n_vars)}
        for link in self.significant_links:
            result[link.target_var].append((
                (link.source_var, -link.tau),
                link.val,
                link.pval
            ))
        return result

# =============================================================================
# Main PCMCI Class
# =============================================================================

class PCMCI:
    """
    PCMCI+ causal discovery algorithm.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data of shape (n_vars, T) where n_vars is the number of
        variables and T is the number of time points.
    tau_max : int
        Maximum time lag to test.
    var_names : list of str, optional
        Names for each variable.
    
    Example
    -------
    >>> data = np.random.randn(5, 1000)
    >>> pcmci = PCMCI(data, tau_max=3)
    >>> result = pcmci.run(alpha=0.05)
    >>> print(result)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        tau_max: int = 1,
        var_names: Optional[List[str]] = None,
    ):
        # Validate and store data
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array (n_vars, T), got shape {data.shape}")
        
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.n_vars, self.T = self.data.shape
        self.tau_max = tau_max
        self.var_names = var_names or [f"X{i}" for i in range(self.n_vars)]
        
        if self.T <= tau_max:
            raise ValueError(f"T ({self.T}) must be greater than tau_max ({tau_max})")
        
        # Create C dataframe (copy data to ensure alignment)
        self._df = _bind._lib.pcmci_dataframe_create_copy(
            self.data.ctypes.data_as(POINTER(c_double)),
            self.n_vars,
            self.T,
            self.tau_max
        )
        
        if not self._df:
            raise MemoryError("Failed to create dataframe")
        
        # Set variable names
        if var_names:
            names_arr = (c_char_p * self.n_vars)()
            for i, name in enumerate(var_names):
                names_arr[i] = name.encode('utf-8')
            _bind._lib.pcmci_dataframe_set_names(self._df, names_arr)
    
    def __del__(self):
        """Clean up C resources."""
        if hasattr(self, '_df') and self._df:
            _bind._lib.pcmci_dataframe_free(self._df)
    
    def run(
        self,
        alpha: float = 0.05,
        max_cond_dim: int = -1,
        n_threads: int = 0,
        use_spearman: bool = True,
        winsorize_thresh: float = 0.0,
        verbosity: int = 0,
        fdr_method: int = 1,  # 0=none, 1=BH
    ) -> PCMCIResult:
        """
        Run PCMCI+ algorithm.
        
        Parameters
        ----------
        alpha : float
            Significance level for independence tests.
        max_cond_dim : int
            Maximum conditioning dimension (-1 for automatic).
        n_threads : int
            Number of threads (0 for automatic).
        use_spearman : bool
            Use Spearman correlation (True) or Pearson (False).
        winsorize_thresh : float
            Winsorization threshold (e.g., 0.01 for 1%/99%).
        verbosity : int
            Verbosity level (0=silent, 1=progress, 2=debug).
        fdr_method : int
            FDR correction method (0=none, 1=Benjamini-Hochberg).
        
        Returns
        -------
        PCMCIResult
            Result object containing adjacency matrix, values, and p-values.
        """
        # Create config using defaults then override
        config = _bind._lib.pcmci_default_config()
        config.tau_max = self.tau_max
        config.alpha_level = alpha
        config.alpha_mci = 0.0  # Same as alpha_level
        config.max_cond_dim = max_cond_dim
        config.n_threads = n_threads
        config.verbosity = verbosity
        config.fdr_method = fdr_method
        config.corr_method = 0 if use_spearman else 1  # SPEARMAN=0, PEARSON=1
        config.winsorize_thresh = winsorize_thresh
        
        # Run algorithm
        result_ptr = _bind._lib.pcmci_run(self._df, byref(config))
        
        if not result_ptr:
            raise RuntimeError("PCMCI+ failed")
        
        try:
            result = result_ptr.contents
            graph = result.graph.contents
            
            # Extract matrices
            n = self.n_vars
            tau = self.tau_max
            total = n * (tau + 1) * n
            
            # Copy data from C arrays to numpy
            adj_flat = np.ctypeslib.as_array(graph.adj, shape=(total,)).copy()
            val_flat = np.ctypeslib.as_array(graph.val_matrix, shape=(total,)).copy()
            pval_flat = np.ctypeslib.as_array(graph.pval_matrix, shape=(total,)).copy()
            
            # Reshape to (n_vars, tau_max+1, n_vars)
            adj_matrix = adj_flat.reshape((n, tau + 1, n)).astype(bool)
            val_matrix = val_flat.reshape((n, tau + 1, n))
            pval_matrix = pval_flat.reshape((n, tau + 1, n))
            
            return PCMCIResult(
                val_matrix=val_matrix,
                pval_matrix=pval_matrix,
                adj_matrix=adj_matrix,
                n_vars=self.n_vars,
                tau_max=self.tau_max,
                alpha=alpha,
                n_significant=result.n_links,
                runtime=result.runtime_secs,
                var_names=self.var_names,
            )
        finally:
            _bind._lib.pcmci_result_free(result_ptr)
    
    def run_skeleton(
        self,
        alpha: float = 0.05,
        max_cond_dim: int = -1,
        n_threads: int = 0,
        verbosity: int = 0,
    ) -> np.ndarray:
        """
        Run only skeleton discovery phase.
        
        Returns boolean adjacency matrix of shape (n_vars, tau_max+1, n_vars).
        """
        config = _bind._lib.pcmci_default_config()
        config.tau_max = self.tau_max
        config.alpha_level = alpha
        config.max_cond_dim = max_cond_dim
        config.n_threads = n_threads
        config.verbosity = verbosity
        
        graph_ptr = _bind._lib.pcmci_skeleton(self._df, byref(config))
        
        if not graph_ptr:
            raise RuntimeError("Skeleton discovery failed")
        
        try:
            graph = graph_ptr.contents
            n = self.n_vars
            tau = self.tau_max
            total = n * (tau + 1) * n
            
            adj_flat = np.ctypeslib.as_array(graph.adj, shape=(total,)).copy()
            return adj_flat.reshape((n, tau + 1, n)).astype(bool)
        finally:
            _bind._lib.pcmci_graph_free(graph_ptr)

# =============================================================================
# Convenience Functions
# =============================================================================

def run_pcmci(
    data: np.ndarray,
    tau_max: int = 1,
    alpha: float = 0.05,
    var_names: Optional[List[str]] = None,
    **kwargs
) -> PCMCIResult:
    """
    Convenience function to run PCMCI+ in one call.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data of shape (n_vars, T).
    tau_max : int
        Maximum time lag.
    alpha : float
        Significance level.
    var_names : list of str, optional
        Variable names.
    **kwargs
        Additional arguments passed to PCMCI.run()
    
    Returns
    -------
    PCMCIResult
    """
    pcmci = PCMCI(data, tau_max=tau_max, var_names=var_names)
    return pcmci.run(alpha=alpha, **kwargs)

def version() -> str:
    """Get PCMCI+ library version."""
    return _bind.version()

# =============================================================================
# Export
# =============================================================================

__all__ = [
    'PCMCI',
    'PCMCIResult', 
    'Link',
    'run_pcmci',
    'version',
]