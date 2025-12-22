"""Native module loader with automatic CUDA version selection.

This module detects the CUDA driver version and loads the appropriate
native module (_pygpukit_native_cu129 or _pygpukit_native_cu131).
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# Cache for loaded module
_native_module: ModuleType | None = None
_cuda_version: tuple[int, int] | None = None


def get_driver_cuda_version() -> tuple[int, int] | None:
    """Get CUDA version supported by the installed driver.

    Returns:
        Tuple of (major, minor) version, e.g., (12, 9) for CUDA 12.9.
        Returns None if detection fails.
    """
    global _cuda_version
    if _cuda_version is not None:
        return _cuda_version

    # Method 1: Try nvidia-smi (most reliable)
    version = _get_version_from_nvidia_smi()
    if version:
        _cuda_version = version
        return version

    # Method 2: Try CUDA Driver API directly
    version = _get_version_from_driver_api()
    if version:
        _cuda_version = version
        return version

    return None


def _get_version_from_nvidia_smi() -> tuple[int, int] | None:
    """Get CUDA version from nvidia-smi output."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        # Parse nvidia-smi output for CUDA version
        # nvidia-smi shows "CUDA Version: X.Y" in its output
        result2 = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result2.returncode != 0:
            return None

        for line in result2.stdout.split("\n"):
            if "CUDA Version:" in line:
                # Extract version like "12.9" from "CUDA Version: 12.9"
                parts = line.split("CUDA Version:")
                if len(parts) >= 2:
                    version_str = parts[1].strip().split()[0]
                    major, minor = version_str.split(".")[:2]
                    return (int(major), int(minor))
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return None


def _get_version_from_driver_api() -> tuple[int, int] | None:
    """Get CUDA version from CUDA Driver API."""
    try:
        if sys.platform == "win32":
            cuda = ctypes.WinDLL("nvcuda.dll")
        else:
            cuda = ctypes.CDLL("libcuda.so.1")

        # cuDriverGetVersion returns the CUDA version
        version = ctypes.c_int()
        result = cuda.cuDriverGetVersion(ctypes.byref(version))
        if result == 0:  # CUDA_SUCCESS
            # Version is encoded as 1000*major + 10*minor
            v = version.value
            major = v // 1000
            minor = (v % 1000) // 10
            return (major, minor)
    except (OSError, AttributeError):
        pass
    return None


def get_native_module() -> ModuleType:
    """Load and return the appropriate native module.

    Automatically selects between cu129 and cu131 based on driver version.
    Falls back to the available module if only one is present.

    Returns:
        The loaded native module.

    Raises:
        ImportError: If no compatible native module is found.
    """
    global _native_module
    if _native_module is not None:
        return _native_module

    cuda_version = get_driver_cuda_version()

    # Determine which module to load
    # CUDA 13.1+ drivers can use cu131, older drivers use cu129
    prefer_cu131 = cuda_version is not None and cuda_version >= (13, 1)

    # Try to import the preferred module first
    if prefer_cu131:
        try:
            from pygpukit import _pygpukit_native_cu131 as native

            _native_module = native
            return native
        except ImportError:
            pass

    # Try cu129 (works with CUDA 12.8+ drivers)
    try:
        from pygpukit import _pygpukit_native_cu129 as native

        _native_module = native
        return native
    except ImportError:
        pass

    # Try cu131 as fallback
    try:
        from pygpukit import _pygpukit_native_cu131 as native

        _native_module = native
        return native
    except ImportError:
        pass

    # Try the legacy single module name (for backwards compatibility)
    try:
        from pygpukit import _pygpukit_native as native

        _native_module = native
        return native
    except ImportError:
        pass

    raise ImportError(
        "No compatible PyGPUkit native module found. "
        f"Driver CUDA version: {cuda_version}. "
        "Please ensure you have a compatible NVIDIA driver installed."
    )


def get_loaded_cuda_version() -> str:
    """Get the CUDA version of the loaded native module.

    Returns:
        String like "cu129" or "cu131", or "unknown" if not determinable.
    """
    module = get_native_module()
    module_name = module.__name__

    if module_name.endswith("_cu129"):
        return "cu129"
    elif module_name.endswith("_cu131"):
        return "cu131"
    else:
        return "unknown"


# Convenience: expose the module directly
def __getattr__(name: str):
    """Allow attribute access to native module members."""
    module = get_native_module()
    return getattr(module, name)
