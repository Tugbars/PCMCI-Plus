"""
Low-level ctypes bindings for PCMCI+ C library.

This module provides direct access to the C API. For a more Pythonic interface,
use the pcmci module instead.

SPDX-License-Identifier: GPL-3.0-or-later
"""

import ctypes
import os
import sys
import platform
from ctypes import (
    c_int32, c_int64, c_double, c_bool, c_char_p, c_void_p,
    POINTER, Structure, byref
)

# =============================================================================
# Library Loading
# =============================================================================

def _add_dll_directories():
    """Add MKL and compiler DLL directories to search path (Windows only)."""
    if platform.system() != 'Windows':
        return
    
    # Common Intel oneAPI installation paths
    oneapi_paths = [
        os.environ.get('MKLROOT', ''),
        os.environ.get('ONEAPI_ROOT', ''),
        r'C:\Program Files (x86)\Intel\oneAPI',
        r'C:\Program Files\Intel\oneAPI',
    ]
    
    # MKL and compiler runtime directories to add
    subdirs = [
        r'mkl\latest\bin',
        r'mkl\latest\redist\intel64',
        r'compiler\latest\bin',
        r'compiler\latest\windows\redist\intel64_win\compiler',
        # For older installations
        r'mkl\latest\redist\intel64',
        r'redist\intel64\mkl',
        r'redist\intel64\compiler',
    ]
    
    added_paths = []
    
    for base in oneapi_paths:
        if not base or not os.path.exists(base):
            continue
        
        for subdir in subdirs:
            dll_path = os.path.join(base, subdir)
            if os.path.exists(dll_path):
                # Add to DLL search path (Python 3.8+)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(dll_path)
                        added_paths.append(dll_path)
                    except OSError:
                        pass
                # Also add to PATH for older Python / subprocess
                if dll_path not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = dll_path + os.pathsep + os.environ.get('PATH', '')
    
    return added_paths

def _find_library():
    """Find the PCMCI shared library."""
    # Determine library name based on platform
    if platform.system() == 'Windows':
        lib_names = ['pcmci.dll', 'libpcmci.dll']
    elif platform.system() == 'Darwin':
        lib_names = ['libpcmci.dylib']
    else:
        lib_names = ['libpcmci.so']
    
    # Search paths
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),  # Same directory as this file
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'Release'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'Debug'),
        '/usr/local/lib',
        '/usr/lib',
    ]
    
    # Also check LD_LIBRARY_PATH / PATH
    if platform.system() == 'Windows':
        env_paths = os.environ.get('PATH', '').split(';')
    else:
        env_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    search_paths.extend(env_paths)
    
    for path in search_paths:
        for lib_name in lib_names:
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                return os.path.abspath(lib_path)
    
    raise OSError(
        f"Could not find PCMCI library. Searched for {lib_names} in:\n" +
        "\n".join(f"  - {p}" for p in search_paths[:6])
    )

# Add MKL directories before loading the library
_add_dll_directories()

# Load the library
_lib_path = _find_library()
try:
    _lib = ctypes.CDLL(_lib_path)
except OSError as e:
    # Provide helpful error message
    error_msg = str(e)
    if platform.system() == 'Windows':
        error_msg += "\n\nMKL DLLs not found. Try one of:\n"
        error_msg += "1. Run from Intel oneAPI command prompt\n"
        error_msg += "2. Set MKLROOT environment variable:\n"
        error_msg += '   set MKLROOT=C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\n'
        error_msg += "3. Add MKL bin to PATH:\n"
        error_msg += '   set PATH=%MKLROOT%\\bin;%PATH%\n'
        error_msg += "4. Copy MKL DLLs to the same folder as pcmci.dll:\n"
        error_msg += "   - mkl_core.2.dll\n"
        error_msg += "   - mkl_intel_thread.2.dll\n"
        error_msg += "   - mkl_def.2.dll\n"
        error_msg += "   - libiomp5md.dll\n"
    raise OSError(error_msg) from e

# =============================================================================
# Structure Definitions
# =============================================================================

class VarLag(Structure):
    """Variable-lag pair (i, tau)."""
    _fields_ = [
        ("var", c_int32),
        ("tau", c_int32),
    ]

class CIResult(Structure):
    """Conditional independence test result."""
    _fields_ = [
        ("val", c_double),      # Test statistic value (correlation)
        ("pvalue", c_double),   # P-value
        ("stat", c_double),     # Raw statistic (t-value)
        ("df", c_int32),        # Degrees of freedom
    ]

class Link(Structure):
    """A causal link."""
    _fields_ = [
        ("i", c_int32),
        ("tau", c_int32),
        ("j", c_int32),
        ("val", c_double),
        ("pvalue", c_double),
    ]

class Sepset(Structure):
    """Separation set for a removed link."""
    _fields_ = [
        ("vars", POINTER(c_int32)),
        ("taus", POINTER(c_int32)),
        ("size", c_int32),
    ]

class Graph(Structure):
    """Causal graph structure."""
    _fields_ = [
        ("adj", POINTER(c_bool)),
        ("link_types", POINTER(c_int32)),  # pcmci_link_type_t
        ("val_matrix", POINTER(c_double)),
        ("pval_matrix", POINTER(c_double)),
        ("sepsets", POINTER(Sepset)),
        ("n_vars", c_int32),
        ("tau_max", c_int32),
    ]

class DataFrame(Structure):
    """Time series dataframe."""
    _fields_ = [
        ("data", POINTER(c_double)),
        ("var_names", POINTER(c_char_p)),
        ("n_vars", c_int32),
        ("T", c_int32),
        ("tau_max", c_int32),
        ("owns_data", c_bool),
    ]

class Config(Structure):
    """PCMCI+ configuration - must match C struct exactly."""
    _fields_ = [
        # Test parameters
        ("test_type", c_int32),         # pcmci_test_type_t
        ("corr_method", c_int32),       # pcmci_corr_method_t
        ("alpha_level", c_double),
        ("alpha_mci", c_double),
        # Robustness
        ("winsorize_thresh", c_double),
        # Algorithm parameters
        ("tau_min", c_int32),
        ("tau_max", c_int32),
        ("max_cond_dim", c_int32),
        ("pc_stable", c_bool),
        # Multiple testing
        ("fdr_method", c_int32),        # pcmci_fdr_method_t
        # Parallelization
        ("n_threads", c_int32),
        # Verbosity
        ("verbosity", c_int32),
    ]

class Result(Structure):
    """PCMCI+ result - must match C struct exactly."""
    _fields_ = [
        ("graph", POINTER(Graph)),
        ("links", POINTER(Link)),
        ("n_links", c_int32),
        # Statistics
        ("n_tests", c_int64),
        ("runtime_secs", c_double),
        # Intermediate
        ("skeleton", POINTER(Graph)),
    ]

# =============================================================================
# Function Signatures
# =============================================================================

# Version
_lib.pcmci_version.argtypes = []
_lib.pcmci_version.restype = c_char_p

# DataFrame functions
_lib.pcmci_dataframe_create.argtypes = [POINTER(c_double), c_int32, c_int32, c_int32]
_lib.pcmci_dataframe_create.restype = POINTER(DataFrame)

_lib.pcmci_dataframe_create_copy.argtypes = [POINTER(c_double), c_int32, c_int32, c_int32]
_lib.pcmci_dataframe_create_copy.restype = POINTER(DataFrame)

_lib.pcmci_dataframe_alloc.argtypes = [c_int32, c_int32, c_int32]
_lib.pcmci_dataframe_alloc.restype = POINTER(DataFrame)

_lib.pcmci_dataframe_set_names.argtypes = [POINTER(DataFrame), POINTER(c_char_p)]
_lib.pcmci_dataframe_set_names.restype = None

_lib.pcmci_dataframe_free.argtypes = [POINTER(DataFrame)]
_lib.pcmci_dataframe_free.restype = None

# Config functions
_lib.pcmci_default_config.argtypes = []
_lib.pcmci_default_config.restype = Config

# Graph functions
_lib.pcmci_graph_alloc.argtypes = [c_int32, c_int32]
_lib.pcmci_graph_alloc.restype = POINTER(Graph)

_lib.pcmci_graph_free.argtypes = [POINTER(Graph)]
_lib.pcmci_graph_free.restype = None

_lib.pcmci_graph_has_link.argtypes = [POINTER(Graph), c_int32, c_int32, c_int32]
_lib.pcmci_graph_has_link.restype = c_bool

_lib.pcmci_graph_get_val.argtypes = [POINTER(Graph), c_int32, c_int32, c_int32]
_lib.pcmci_graph_get_val.restype = c_double

_lib.pcmci_graph_get_pval.argtypes = [POINTER(Graph), c_int32, c_int32, c_int32]
_lib.pcmci_graph_get_pval.restype = c_double

# Main algorithm
_lib.pcmci_run.argtypes = [POINTER(DataFrame), POINTER(Config)]
_lib.pcmci_run.restype = POINTER(Result)

_lib.pcmci_result_free.argtypes = [POINTER(Result)]
_lib.pcmci_result_free.restype = None

# Skeleton
_lib.pcmci_skeleton.argtypes = [POINTER(DataFrame), POINTER(Config)]
_lib.pcmci_skeleton.restype = POINTER(Graph)

# MCI
_lib.pcmci_mci.argtypes = [POINTER(DataFrame), POINTER(Graph), POINTER(Config)]
_lib.pcmci_mci.restype = POINTER(Graph)

# Partial correlation test
_lib.pcmci_parcorr_test.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
    c_int32, c_int32
]
_lib.pcmci_parcorr_test.restype = CIResult

# =============================================================================
# Helper Functions
# =============================================================================

def graph_idx(n_vars: int, tau_max: int, i: int, tau: int, j: int) -> int:
    """Calculate flat index into graph arrays."""
    return i * (tau_max + 1) * n_vars + tau * n_vars + j

def version() -> str:
    """Get library version string."""
    return _lib.pcmci_version().decode('utf-8')

# =============================================================================
# Export
# =============================================================================

__all__ = [
    '_lib',
    'VarLag', 'CIResult', 'Sepset', 'Graph', 'DataFrame', 'Config', 'Result',
    'graph_idx', 'version',
]