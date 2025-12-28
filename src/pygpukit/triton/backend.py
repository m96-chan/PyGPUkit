"""
Triton backend detection and configuration.
"""

import os
from functools import lru_cache

_triton = None
_triton_available = None


@lru_cache(maxsize=1)
def triton_available() -> bool:
    """Check if Triton is available."""
    global _triton, _triton_available

    if _triton_available is not None:
        return _triton_available

    try:
        import triton

        _triton = triton
        _triton_available = True
        return True
    except ImportError:
        _triton_available = False
        return False


def get_triton():
    """Get the triton module."""
    if not triton_available():
        raise RuntimeError("Triton is not available. Install with: pip install triton")
    return _triton


def get_triton_device() -> str:
    """Get the Triton device string."""
    return "cuda"


def use_triton_backend() -> bool:
    """Check if Triton backend should be used."""
    if not triton_available():
        return False

    # Check environment variable
    env_val = os.environ.get("PYGPUKIT_USE_TRITON", "1").lower()
    return env_val in ("1", "true", "yes")


def triton_version() -> str:
    """Get Triton version string."""
    if not triton_available() or _triton is None:
        return "not installed"
    return str(_triton.__version__)
