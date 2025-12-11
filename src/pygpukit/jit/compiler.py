"""JIT compiler for CUDA kernels using NVRTC."""

from __future__ import annotations

import hashlib
import re
from typing import Any


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
        pattern = rf'__global__\s+\w+\s+{re.escape(func_name)}\s*\('
        return bool(re.search(pattern, source))

    def _compile(self) -> None:
        """Compile the CUDA source code.

        For CPU simulation backend, we just mark as compiled.
        For real CUDA, we would use NVRTC.
        """
        from pygpukit.core.backend import CUDABackend, get_backend

        backend = get_backend()

        if isinstance(backend, CUDABackend) and backend.is_available():
            self._compile_nvrtc()
        else:
            # CPU simulation - just mark as compiled
            self._is_compiled = True
            self._ptx = f"// Simulated PTX for {self._name}"

    def _compile_nvrtc(self) -> None:
        """Compile using NVRTC."""
        try:
            from cuda import cuda, nvrtc

            # Create program
            err, prog = nvrtc.nvrtcCreateProgram(
                self._source.encode(), b"kernel.cu", 0, [], []
            )
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"Failed to create program: {err}")

            # Compile with options
            opts = [o.encode() for o in self._options]
            err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                # Get compilation log
                log_size = nvrtc.nvrtcGetProgramLogSize(prog)[1]
                log = b" " * log_size
                nvrtc.nvrtcGetProgramLog(prog, log)
                nvrtc.nvrtcDestroyProgram(prog)
                raise RuntimeError(f"Compilation failed: {log.decode()}")

            # Get PTX
            ptx_size = nvrtc.nvrtcGetPTXSize(prog)[1]
            ptx = b" " * ptx_size
            nvrtc.nvrtcGetPTX(prog, ptx)
            nvrtc.nvrtcDestroyProgram(prog)

            self._ptx = ptx.decode()

            # Load module and get function
            err, self._module = cuda.cuModuleLoadData(ptx)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to load module: {err}")

            err, self._kernel = cuda.cuModuleGetFunction(
                self._module, self._name.encode()
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to get function: {err}")

            self._is_compiled = True

        except ImportError:
            # Fall back to simulation
            self._is_compiled = True
            self._ptx = f"// Simulated PTX for {self._name}"

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
        from pygpukit.core.array import GPUArray
        from pygpukit.core.backend import CUDABackend, get_backend

        backend = get_backend()

        if not isinstance(backend, CUDABackend) or not backend.is_available():
            # CPU simulation - do nothing (operations are simulated elsewhere)
            return

        if not self._is_compiled or self._kernel is None:
            raise RuntimeError("Kernel not compiled")

        # Compute grid size if not provided
        if grid_size is None:
            for arg in args:
                if isinstance(arg, GPUArray):
                    grid_size = (arg.size + self._block_size - 1) // self._block_size
                    break
            if grid_size is None:
                grid_size = 1

        # Prepare arguments
        # This is a simplified version - real implementation needs proper arg handling
        from cuda import cuda

        # Launch kernel
        cuda.cuLaunchKernel(
            self._kernel,
            grid_size,
            1,
            1,  # grid dim
            self._block_size,
            1,
            1,  # block dim
            0,
            None,  # shared mem, stream
            args,
            0,
        )

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
