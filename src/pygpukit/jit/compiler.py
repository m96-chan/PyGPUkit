"""JIT compiler for CUDA kernels using NVRTC.

NVRTC (NVIDIA Runtime Compilation) is used to compile CUDA kernels at runtime.
NVRTC is optional - use `is_nvrtc_available()` to check availability.

If NVRTC is not available:
- JIT compilation will raise RuntimeError
- Pre-compiled kernels (matmul, add, etc.) will still work via the native backend
- CPU simulation mode will continue to work
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def is_nvrtc_available() -> bool:
    """Check if NVRTC JIT compiler is available.

    NVRTC enables runtime compilation of custom CUDA kernels.
    It is optional - pre-compiled GPU operations work without NVRTC.

    Returns:
        True if NVRTC is available and functional, False otherwise.

    Example:
        >>> import pygpukit as gp
        >>> if gp.is_nvrtc_available():
        ...     kernel = gp.jit(source, func="my_kernel")
        ... else:
        ...     print("JIT not available, using pre-compiled kernels")
    """
    try:
        from pygpukit.core.backend import get_native_module, has_native_module

        if not has_native_module():
            return False

        native = get_native_module()
        return native.is_nvrtc_available()
    except Exception:
        return False


def get_nvrtc_path() -> str | None:
    """Get the path to the discovered NVRTC library.

    Returns:
        Path to NVRTC DLL/SO if found, None otherwise.

    Example:
        >>> import pygpukit as gp
        >>> path = gp.get_nvrtc_path()
        >>> if path:
        ...     print(f"NVRTC found at: {path}")
    """
    try:
        from pygpukit.core.backend import get_native_module, has_native_module

        # Prefer native module's path (what's actually loaded at runtime)
        if has_native_module():
            native = get_native_module()
            path = native.get_nvrtc_library_path()
            if path:
                return path

        # Fall back to Python-side discovery
        from pygpukit.core.backend import _find_nvrtc_dll

        return _find_nvrtc_dll()
    except Exception:
        return None


def get_nvrtc_version() -> tuple[int, int] | None:
    """Get NVRTC version if available.

    Returns:
        Tuple of (major, minor) version numbers, or None if NVRTC unavailable.

    Example:
        >>> import pygpukit as gp
        >>> version = gp.get_nvrtc_version()
        >>> if version:
        ...     print(f"NVRTC {version[0]}.{version[1]}")
    """
    try:
        from pygpukit.core.backend import get_native_module, has_native_module

        if not has_native_module():
            return None

        native = get_native_module()
        if not native.is_nvrtc_available():
            return None

        return native.get_nvrtc_version()
    except Exception:
        return None


class JITKernel:
    """A JIT-compiled CUDA kernel.

    This class wraps a CUDA kernel that has been compiled at runtime
    using NVRTC (NVIDIA Runtime Compilation).
    """

    def __init__(
        self,
        source: str,
        func_name: str,
        options: list[str] | None = None,
        block_size: int = 256,
    ) -> None:
        """Initialize a JITKernel.

        Args:
            source: CUDA source code.
            func_name: Name of the kernel function.
            options: Compilation options (e.g., ["-O3"]).
            block_size: Default block size for kernel launches.

        Raises:
            ValueError: If the function name is not found in source.
        """
        self._source = source
        self._name = func_name
        self._options = options or []
        self._block_size = block_size
        self._ptx: str | None = None
        self._module: Any = None
        self._kernel: Any = None
        self._is_compiled = False

        # Validate function name exists in source
        if not self._find_kernel_in_source(source, func_name):
            raise ValueError(f"Function '{func_name}' not found in source code")

        # Compile the kernel
        self._compile()

    def _find_kernel_in_source(self, source: str, func_name: str) -> bool:
        """Check if the kernel function exists in source."""
        # Look for __global__ void func_name patterns
        pattern = rf"__global__\s+\w+\s+{re.escape(func_name)}\s*\("
        return bool(re.search(pattern, source))

    def _compile(self) -> None:
        """Compile the CUDA source code.

        For CPU simulation backend, we just mark as compiled.
        For native backend, we use C++ NVRTC via pybind11.
        """
        from pygpukit.core.backend import NativeBackend, get_backend

        backend = get_backend()

        if isinstance(backend, NativeBackend) and backend.is_available():
            self._compile_native()
        else:
            # CPU simulation - just mark as compiled
            self._is_compiled = True
            self._ptx = f"// Simulated PTX for {self._name}"

    def _compile_native(self) -> None:
        """Compile using native C++ module (NVRTC).

        Raises:
            RuntimeError: If NVRTC is not available with helpful installation instructions.
        """
        from pygpukit.core.backend import _find_nvrtc_dll, get_native_module

        native = get_native_module()

        # Check NVRTC availability first
        if not native.is_nvrtc_available():
            nvrtc_path = _find_nvrtc_dll()
            if nvrtc_path:
                # NVRTC DLL found but not working
                msg = (
                    f"NVRTC library found at {nvrtc_path} but failed to initialize.\n"
                    "This may indicate a version mismatch or corrupted installation.\n"
                    "Try updating your NVIDIA GPU driver:\n"
                    "  https://www.nvidia.com/Download/index.aspx"
                )
            else:
                # NVRTC DLL not found
                msg = (
                    "NVRTC (NVIDIA Runtime Compiler) is not available.\n"
                    "JIT compilation of custom kernels requires NVRTC.\n\n"
                    "Pre-compiled GPU operations (matmul, add, mul) work without NVRTC.\n"
                    "To use custom JIT kernels, NVRTC can be obtained from:\n"
                    "  https://developer.nvidia.com/cuda-downloads\n\n"
                    "Check availability: pygpukit.is_nvrtc_available()"
                )
            raise RuntimeError(msg)

        # Use native JITKernel which handles NVRTC compilation
        self._kernel = native.JITKernel(self._source, self._name, self._options)
        self._ptx = self._kernel.ptx
        self._is_compiled = self._kernel.is_compiled

    @property
    def source(self) -> str:
        """Return the source code."""
        return self._source

    @property
    def name(self) -> str:
        """Return the kernel function name."""
        return self._name

    @property
    def options(self) -> list[str]:
        """Return the compilation options."""
        return self._options

    @property
    def block_size(self) -> int:
        """Return the default block size."""
        return self._block_size

    @property
    def is_compiled(self) -> bool:
        """Return whether the kernel is compiled."""
        return self._is_compiled

    @property
    def ptx(self) -> str | None:
        """Return the compiled PTX code."""
        return self._ptx

    def _compute_cache_key(self) -> str:
        """Compute a cache key for this kernel."""
        content = self._source + str(self._options)
        return hashlib.sha256(content.encode()).hexdigest()

    def __call__(
        self, *args: Any, grid_size: int | tuple[int, int] | None = None, **kwargs: Any
    ) -> None:
        """Launch the kernel.

        Args:
            *args: Kernel arguments.
            grid_size: Number of blocks. If None, computed from first array argument.
            **kwargs: Additional kernel arguments.
        """
        from pygpukit.core.backend import NativeBackend, get_backend

        backend = get_backend()

        if not isinstance(backend, NativeBackend) or not backend.is_available():
            # CPU simulation - do nothing (operations are simulated elsewhere)
            return

        if not self._is_compiled or self._kernel is None:
            raise RuntimeError("Kernel not compiled")

        # Native kernel handles launching via its own interface
        # The native JITKernel.launch() method is used for kernel execution
        # For now, operations are handled directly via native ops module
        pass

    def __repr__(self) -> str:
        status = "compiled" if self._is_compiled else "not compiled"
        return f"JITKernel(name={self._name}, {status})"


def jit(
    source: str,
    func: str,
    options: list[str] | None = None,
    block_size: int = 256,
) -> JITKernel:
    """JIT compile a CUDA kernel.

    Args:
        source: CUDA source code containing the kernel.
        func: Name of the kernel function to compile.
        options: Compilation options (e.g., ["-O3", "-arch=sm_80"]).
        block_size: Default block size for kernel launches.

    Returns:
        A JITKernel instance.

    Example:
        >>> src = '''
        ... extern "C" __global__
        ... void scale(float* x, float factor, int n) {
        ...     int idx = blockIdx.x * blockDim.x + threadIdx.x;
        ...     if (idx < n) x[idx] *= factor;
        ... }
        ... '''
        >>> kernel = jit(src, func="scale")
        >>> kernel(x, 0.5, n)
    """
    return JITKernel(source, func, options, block_size)
