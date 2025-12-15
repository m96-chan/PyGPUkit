"""JIT compiler for CUDA kernels using NVRTC.

NVRTC (NVIDIA Runtime Compilation) is used to compile CUDA kernels at runtime.
NVRTC is optional - use `is_nvrtc_available()` to check availability.

If NVRTC is not available:
- JIT compilation will raise NvrtcError
- Pre-compiled kernels (matmul, add, etc.) will still work via the native backend
- CPU simulation mode will continue to work
"""

from __future__ import annotations

import hashlib
import re
from enum import IntEnum
from typing import Any


class NvrtcErrorCode(IntEnum):
    """NVRTC error codes for structured error handling.

    These codes map directly to NVRTC's nvrtcResult enum plus custom codes.
    """

    Success = 0
    OutOfMemory = 1
    ProgramCreationFailure = 2
    InvalidInput = 3
    InvalidProgram = 4
    InvalidOption = 5
    Compilation = 6
    BuiltinOperationFailure = 7
    NoNameExpressionsAfterCompilation = 8
    NoLoweredNamesBeforeCompilation = 9
    NameExpressionNotValid = 10
    InternalError = 11
    # Custom error codes (1000+)
    NotLoaded = 1000  # NVRTC DLL not loaded
    PtxLoadFailed = 1001  # cuModuleLoadData failed
    FunctionNotFound = 1002  # cuModuleGetFunction failed
    LaunchFailed = 1003  # cuLaunchKernel failed


class NvrtcError(RuntimeError):
    """NVRTC JIT compilation error with structured information.

    Attributes:
        code: Structured error code (NvrtcErrorCode)
        compilation_log: NVRTC compiler output (if available)

    Example:
        >>> try:
        ...     kernel = pygpukit.jit(bad_source, "my_kernel")
        ... except pygpukit.NvrtcError as e:
        ...     print(f"Error code: {e.code.name}")
        ...     if e.compilation_log:
        ...         print(f"Compiler log: {e.compilation_log}")
    """

    def __init__(
        self,
        message: str,
        code: NvrtcErrorCode | int = NvrtcErrorCode.InternalError,
        compilation_log: str = "",
    ) -> None:
        super().__init__(message)
        self._code = NvrtcErrorCode(code) if isinstance(code, int) else code
        self._compilation_log = compilation_log

    @property
    def code(self) -> NvrtcErrorCode:
        """Return the structured error code."""
        return self._code

    @property
    def compilation_log(self) -> str:
        """Return the NVRTC compiler output log."""
        return self._compilation_log

    def __str__(self) -> str:
        base = super().__str__()
        return f"[{self._code.name}] {base}"


def _wrap_native_nvrtc_error(exc: Exception) -> NvrtcError:
    """Convert native NvrtcError to Python NvrtcError."""
    code = getattr(exc, "code", NvrtcErrorCode.InternalError)
    log = getattr(exc, "compilation_log", "")

    # Convert native enum to Python enum if needed
    if hasattr(code, "value"):
        code = code.value

    return NvrtcError(str(exc), code, log)


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


# ============================================================================
# Driver Version Requirements
# ============================================================================

# Minimum supported CUDA driver version (CUDA 11.0)
# Version format: MAJOR*1000 + MINOR*10 (e.g., 11.0 = 11000)
MIN_DRIVER_VERSION = 11000
MIN_DRIVER_VERSION_STR = "11.0"

# Minimum required GPU architecture (Ampere)
MIN_SM_VERSION = 80
MIN_SM_VERSION_STR = "SM 8.0 (Ampere)"


def get_driver_requirements() -> dict[str, str]:
    """Get driver and hardware requirements for PyGPUkit.

    Returns:
        Dictionary with minimum requirements and recommendations.

    Example:
        >>> import pygpukit as gp
        >>> reqs = gp.get_driver_requirements()
        >>> print(reqs['min_driver_version'])
        '11.0'
    """
    return {
        "min_driver_version": MIN_DRIVER_VERSION_STR,
        "min_gpu_architecture": MIN_SM_VERSION_STR,
        "recommended_driver_version": "12.0+",
        "recommended_gpu": "RTX 30xx/40xx series or newer",
        "supported_architectures": "Ampere (SM 80-86), Ada (SM 89), Hopper (SM 90)",
        "notes": (
            "PyGPUkit requires Ampere or newer GPUs. "
            "Older architectures (Pascal, Turing) are not supported. "
            "For best performance, use the latest NVIDIA driver."
        ),
    }


def check_driver_compatibility() -> tuple[bool, str]:
    """Check if current driver meets minimum requirements.

    Returns:
        Tuple of (is_compatible, message) where is_compatible is True if the
        driver meets requirements, and message contains details.

    Example:
        >>> import pygpukit as gp
        >>> ok, msg = gp.check_driver_compatibility()
        >>> if not ok:
        ...     print(f"Warning: {msg}")
    """
    try:
        from pygpukit.core.backend import get_native_module, has_native_module

        if not has_native_module():
            return False, "Native module not available (CPU simulation mode)"

        native = get_native_module()

        if not native.is_cuda_available():
            return False, "CUDA is not available"

        driver_version = native.get_driver_version()
        if driver_version < MIN_DRIVER_VERSION:
            driver_str = f"{driver_version // 1000}.{(driver_version % 1000) // 10}"
            return False, (
                f"Driver version {driver_str} is below minimum required "
                f"{MIN_DRIVER_VERSION_STR}. Please update your NVIDIA driver."
            )

        sm_version = native.get_sm_version()
        if sm_version < MIN_SM_VERSION:
            return False, (
                f"GPU SM {sm_version // 10}.{sm_version % 10} is below minimum "
                f"required {MIN_SM_VERSION_STR}. PyGPUkit requires Ampere or newer."
            )

        # All checks passed
        driver_str = f"{driver_version // 1000}.{(driver_version % 1000) // 10}"
        return True, f"Driver {driver_str}, SM {sm_version // 10}.{sm_version % 10}"

    except Exception as e:
        return False, f"Error checking compatibility: {e}"


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

    # Retry configuration for transient errors
    _MAX_RETRIES = 3
    _RETRY_DELAY_MS = 100  # Base delay in milliseconds
    _RETRYABLE_ERRORS = {
        NvrtcErrorCode.OutOfMemory,
        NvrtcErrorCode.InternalError,
        NvrtcErrorCode.BuiltinOperationFailure,
    }

    def _compile_native(self) -> None:
        """Compile using native C++ module (NVRTC).

        Automatically selects appropriate -arch option based on GPU and driver.
        Falls back to lower architectures if PTX loading fails.
        Retries on transient errors (out of memory, internal errors).

        Raises:
            NvrtcError: If NVRTC is not available or compilation fails.
        """
        import time
        import warnings

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
            raise NvrtcError(msg, NvrtcErrorCode.NotLoaded)

        # Prepare options with auto arch selection
        options = self._prepare_compile_options(native)

        # Try compilation with fallback on PTX load failure
        fallback_archs = self._get_fallback_archs(native)
        last_error: Exception | None = None

        for arch_attempt, arch in enumerate(fallback_archs):
            current_options = self._replace_arch_option(options, arch)

            # Retry loop for transient errors
            for retry in range(self._MAX_RETRIES):
                try:
                    self._kernel = native.JITKernel(self._source, self._name, current_options)
                    self._ptx = self._kernel.ptx
                    self._is_compiled = self._kernel.is_compiled

                    # Warn if fallback was used
                    if arch_attempt > 0:
                        warnings.warn(
                            f"JIT compilation succeeded using fallback architecture "
                            f"'{arch}'. Original architecture failed. Consider updating "
                            f"your NVIDIA driver for better compatibility.",
                            UserWarning,
                            stacklevel=4,
                        )
                    return  # Success
                except Exception as e:
                    last_error = e
                    err_code = self._get_error_code(e)
                    err_msg = str(e)

                    # Check if this is a retryable transient error
                    if err_code in self._RETRYABLE_ERRORS and retry < self._MAX_RETRIES - 1:
                        # Exponential backoff
                        delay = self._RETRY_DELAY_MS * (2**retry) / 1000.0
                        time.sleep(delay)
                        continue

                    # Compilation errors should not be retried
                    if "Compilation failed" in err_msg:
                        break

                    # PTX load failure - try next fallback arch
                    is_ptx_load_error = (
                        "PTX" in err_msg
                        or "module" in err_msg.lower()
                        or "CUDA_ERROR" in err_msg
                        or err_code == NvrtcErrorCode.PtxLoadFailed
                    )
                    if is_ptx_load_error and arch_attempt < len(fallback_archs) - 1:
                        break  # Try next arch

                    # Other error - stop retrying this arch
                    break

            # If we reach here without returning, try next arch
            # (unless it was a compilation error)
            if last_error and "Compilation failed" in str(last_error):
                break  # Don't try other archs for syntax errors

        # All attempts failed
        if last_error is not None:
            if hasattr(last_error, "code") and hasattr(last_error, "compilation_log"):
                raise _wrap_native_nvrtc_error(last_error) from None
            msg = str(last_error)
            if "Compilation failed" in msg:
                raise NvrtcError(msg, NvrtcErrorCode.Compilation) from None
            elif "not found in module" in msg or "Function" in msg:
                raise NvrtcError(msg, NvrtcErrorCode.FunctionNotFound) from None
            elif "PTX" in msg or "module" in msg.lower():
                raise NvrtcError(msg, NvrtcErrorCode.PtxLoadFailed) from None
            else:
                raise NvrtcError(msg, NvrtcErrorCode.InternalError) from None

    def _get_error_code(self, exc: Exception) -> NvrtcErrorCode:
        """Extract error code from exception."""
        if hasattr(exc, "code"):
            code = exc.code
            if hasattr(code, "value"):
                return NvrtcErrorCode(code.value)
            elif isinstance(code, int):
                try:
                    return NvrtcErrorCode(code)
                except ValueError:
                    return NvrtcErrorCode.InternalError
        return NvrtcErrorCode.InternalError

    def _prepare_compile_options(self, native: Any) -> list[str]:
        """Prepare compilation options with auto arch selection."""
        options = list(self._options)

        # Check if user already specified -arch
        has_arch = any(
            opt.startswith("-arch=") or opt.startswith("--gpu-architecture=") for opt in options
        )

        if not has_arch:
            # Auto-select arch based on current GPU
            try:
                recommended_arch = native.get_recommended_arch()
                options.append(f"-arch={recommended_arch}")
            except Exception:
                # Fallback to sm_80 (minimum supported)
                options.append("-arch=sm_80")

        return options

    def _get_fallback_archs(self, native: Any) -> list[str]:
        """Get list of architectures to try (primary + fallbacks)."""
        # Check if user specified arch
        user_arch = None
        for opt in self._options:
            if opt.startswith("-arch="):
                user_arch = opt.split("=", 1)[1]
                break
            elif opt.startswith("--gpu-architecture="):
                user_arch = opt.split("=", 1)[1]
                break

        if user_arch:
            # User specified arch - use it as primary, add fallbacks
            archs = [user_arch]
            try:
                fallbacks = native.get_fallback_archs()
                for fb in fallbacks:
                    if fb not in archs:
                        archs.append(fb)
            except Exception:
                archs.extend(["sm_86", "sm_80", "compute_80"])
            return archs
        else:
            # Auto-select - use recommended arch as primary
            try:
                return native.get_fallback_archs()
            except Exception:
                return ["sm_86", "sm_80", "compute_80"]

    def _replace_arch_option(self, options: list[str], new_arch: str) -> list[str]:
        """Replace -arch option with new architecture."""
        result = []
        arch_found = False
        for opt in options:
            if opt.startswith("-arch=") or opt.startswith("--gpu-architecture="):
                result.append(f"-arch={new_arch}")
                arch_found = True
            else:
                result.append(opt)
        if not arch_found:
            result.append(f"-arch={new_arch}")
        return result

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


# ============================================================================
# JIT Warmup System
# ============================================================================

import threading
from typing import Callable

# Global warmup state
_warmup_lock = threading.Lock()
_warmup_done = False
_warmup_thread: threading.Thread | None = None
_warmup_error: Exception | None = None

# Warmup test kernel
_WARMUP_KERNEL_SOURCE = """
extern "C" __global__ void _pygpukit_warmup_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = x[idx];
}
"""


def warmup(
    background: bool = False,
    callback: Callable[[], None] | None = None,
) -> bool:
    """Warm up the JIT compiler.

    This function pre-initializes the NVRTC JIT compiler by compiling a simple
    test kernel. This ensures that subsequent JIT compilations are faster as
    the compiler is already loaded and initialized.

    Args:
        background: If True, run warmup in a background thread.
        callback: Optional callback to invoke when warmup completes (only used
            when background=True).

    Returns:
        True if warmup succeeded (or was already done), False if NVRTC is
        not available. When background=True, returns True immediately and
        warmup continues in background.

    Example:
        >>> import pygpukit as gp
        >>> # Synchronous warmup
        >>> gp.warmup()
        True
        >>> # Background warmup
        >>> gp.warmup(background=True, callback=lambda: print("Ready!"))
        True
    """
    global _warmup_done, _warmup_thread, _warmup_error

    with _warmup_lock:
        if _warmup_done:
            return _warmup_error is None

        if _warmup_thread is not None and _warmup_thread.is_alive():
            # Warmup already in progress
            if not background:
                _warmup_thread.join()
            return True

    if background:
        _warmup_thread = threading.Thread(
            target=_do_warmup,
            args=(callback,),
            daemon=True,
        )
        _warmup_thread.start()
        return True
    else:
        return _do_warmup(callback)


def _do_warmup(callback: Callable[[], None] | None = None) -> bool:
    """Perform the actual warmup."""
    global _warmup_done, _warmup_error

    try:
        # Check if NVRTC is available
        if not is_nvrtc_available():
            _warmup_error = NvrtcError("NVRTC not available", NvrtcErrorCode.NotLoaded)
            _warmup_done = True
            return False

        # Compile warmup kernel
        try:
            _ = JITKernel(
                _WARMUP_KERNEL_SOURCE,
                "_pygpukit_warmup_kernel",
                options=[],  # Use default arch
            )
        except NvrtcError as e:
            _warmup_error = e
            _warmup_done = True
            return False

        _warmup_done = True
        _warmup_error = None

        if callback is not None:
            try:
                callback()
            except Exception:
                pass  # Ignore callback errors

        return True
    except Exception as e:
        _warmup_error = e
        _warmup_done = True
        return False


def is_warmup_done() -> bool:
    """Check if JIT warmup has completed.

    Returns:
        True if warmup has completed (successfully or with error), False if
        warmup is in progress or has not started.

    Example:
        >>> import pygpukit as gp
        >>> gp.warmup(background=True)
        True
        >>> # ... do other initialization ...
        >>> while not gp.is_warmup_done():
        ...     time.sleep(0.01)
        >>> print("JIT compiler ready!")
    """
    return _warmup_done


def get_warmup_error() -> Exception | None:
    """Get the warmup error if warmup failed.

    Returns:
        The exception that caused warmup to fail, or None if warmup succeeded
        or has not completed.
    """
    return _warmup_error
