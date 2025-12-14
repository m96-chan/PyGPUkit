#!/usr/bin/env python3
"""Demo: PyGPUkit Runtime Modes

This demo shows the three runtime modes of PyGPUkit:
1. Full JIT Mode - NVRTC found, custom kernels available
2. GPU Fallback Mode - NVRTC not found, pre-compiled kernels only
3. CPU Simulation Mode - No GPU, NumPy-based simulation

Run this script to see which mode your system supports.
"""

import sys


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_status(label: str, value: str, ok: bool = True) -> None:
    """Print a status line with checkmark or X."""
    mark = "[OK]" if ok else "[--]"
    print(f"  {mark} {label}: {value}")


def demo_full_jit_mode() -> bool:
    """Demo: Full JIT Mode with NVRTC available."""
    print_header("Mode 1: Full JIT (NVRTC Available)")

    import pygpukit as gp

    if not gp.is_cuda_available():
        print("  [SKIP] CUDA not available")
        return False

    if not gp.is_nvrtc_available():
        print("  [SKIP] NVRTC not available")
        return False

    # Show NVRTC info
    nvrtc_path = gp.get_nvrtc_path()
    nvrtc_version = gp.get_nvrtc_version()

    print_status("CUDA", "Available", True)
    print_status("NVRTC", f"v{nvrtc_version[0]}.{nvrtc_version[1]}", True)
    print_status("NVRTC Path", nvrtc_path or "System", True)

    # Demo: Custom JIT kernel
    print("\n  [Demo] Custom JIT Kernel:")

    kernel_source = '''
    extern "C" __global__
    void scale_array(float* data, float factor, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] *= factor;
        }
    }
    '''

    try:
        kernel = gp.jit(kernel_source, func="scale_array")
        print(f"    - Kernel compiled: {kernel.name}")
        print(f"    - PTX generated: {len(kernel.ptx)} bytes")
        print("    - Custom kernels: AVAILABLE")
    except Exception as e:
        print(f"    - JIT failed: {e}")
        return False

    # Demo: Pre-compiled operations
    print("\n  [Demo] Pre-compiled Operations:")
    import numpy as np

    A = gp.from_numpy(np.random.randn(256, 256).astype(np.float32))
    B = gp.from_numpy(np.random.randn(256, 256).astype(np.float32))

    C = gp.matmul(A, B)
    print(f"    - matmul(256x256): OK, result shape {C.shape}")

    D = gp.add(A, B)
    print(f"    - add(256x256): OK, result shape {D.shape}")

    print("\n  [Result] Full JIT Mode: ALL FEATURES AVAILABLE")
    return True


def demo_gpu_fallback_mode() -> bool:
    """Demo: GPU Fallback Mode without NVRTC."""
    print_header("Mode 2: GPU Fallback (No NVRTC)")

    import pygpukit as gp

    if not gp.is_cuda_available():
        print("  [SKIP] CUDA not available")
        return False

    # This mode is when CUDA works but NVRTC doesn't
    # We simulate by showing what would happen

    print_status("CUDA", "Available", True)
    print_status("NVRTC", "Not Available", False)

    print("\n  [Info] In this mode:")
    print("    - Pre-compiled GPU operations work (matmul, add, mul)")
    print("    - Custom JIT kernels are NOT available")
    print("    - GPU memory and scheduling work normally")

    # Demo: Pre-compiled operations still work
    print("\n  [Demo] Pre-compiled Operations (Still Work):")
    import numpy as np

    A = gp.from_numpy(np.random.randn(256, 256).astype(np.float32))
    B = gp.from_numpy(np.random.randn(256, 256).astype(np.float32))

    C = gp.matmul(A, B)
    print(f"    - matmul(256x256): OK, result shape {C.shape}")

    D = gp.add(A, B)
    print(f"    - add(256x256): OK, result shape {D.shape}")

    E = gp.mul(A, B)
    print(f"    - mul(256x256): OK, result shape {E.shape}")

    # Show what happens when JIT is attempted without NVRTC
    print("\n  [Demo] JIT Kernel Attempt (Would Fail):")
    print("    - Calling gp.jit() without NVRTC raises RuntimeError")
    print("    - Error message includes installation instructions")
    print("    - Pre-compiled ops remain functional")

    print("\n  [Result] GPU Fallback Mode: PRE-COMPILED OPS ONLY")
    return True


def demo_cpu_simulation_mode() -> bool:
    """Demo: CPU Simulation Mode without GPU."""
    print_header("Mode 3: CPU Simulation (No GPU)")

    # Force CPU backend for demo
    from pygpukit.core.backend import CPUSimulationBackend, set_backend

    original_backend = None
    try:
        from pygpukit.core.backend import _backend

        original_backend = _backend
    except Exception:
        pass

    # Set CPU backend
    set_backend(CPUSimulationBackend())

    import pygpukit as gp

    print_status("CUDA", "Not Available (Simulated)", False)
    print_status("NVRTC", "Not Available", False)
    print_status("Backend", "CPU Simulation", True)

    print("\n  [Info] In this mode:")
    print("    - All operations run on CPU using NumPy")
    print("    - API is identical - code works without changes")
    print("    - Useful for testing/development without GPU")

    # Demo: Operations work via NumPy
    print("\n  [Demo] CPU-Simulated Operations:")
    import numpy as np

    # Create arrays (backed by NumPy in simulation mode)
    A = gp.zeros((128, 128), dtype="float32")
    B = gp.ones((128, 128), dtype="float32")

    print(f"    - zeros(128x128): OK, dtype {A.dtype}")
    print(f"    - ones(128x128): OK, dtype {B.dtype}")

    # Operations work but run on CPU
    C = gp.add(A, B)
    print(f"    - add(128x128): OK (CPU), result shape {C.shape}")

    # JIT also works in simulation (just marks as compiled)
    kernel_source = '''
    extern "C" __global__
    void dummy(float* x) {}
    '''
    kernel = gp.jit(kernel_source, func="dummy")
    print(f"    - jit kernel: OK (simulated), compiled={kernel.is_compiled}")

    # Restore original backend
    if original_backend is not None:
        set_backend(original_backend)
    else:
        from pygpukit.core.backend import reset_backend

        reset_backend()

    print("\n  [Result] CPU Simulation Mode: FULL API, CPU EXECUTION")
    return True


def main() -> None:
    """Main demo entry point."""
    print("=" * 60)
    print(" PyGPUkit Runtime Modes Demo")
    print(" Version: ", end="")

    try:
        import pygpukit as gp

        print(gp.__version__)
    except Exception:
        print("(import failed)")

    print("=" * 60)

    # Check current system status
    print_header("System Status")

    try:
        import pygpukit as gp

        cuda_available = gp.is_cuda_available()
        nvrtc_available = gp.is_nvrtc_available()
        nvrtc_path = gp.get_nvrtc_path()
        nvrtc_version = gp.get_nvrtc_version()

        print_status("CUDA Available", str(cuda_available), cuda_available)
        print_status("NVRTC Available", str(nvrtc_available), nvrtc_available)

        if nvrtc_path:
            print_status("NVRTC Path", nvrtc_path, True)
        if nvrtc_version:
            print_status("NVRTC Version", f"{nvrtc_version[0]}.{nvrtc_version[1]}", True)

        # Determine current mode
        if cuda_available and nvrtc_available:
            current_mode = "Full JIT Mode"
        elif cuda_available:
            current_mode = "GPU Fallback Mode"
        else:
            current_mode = "CPU Simulation Mode"

        print(f"\n  Current Mode: {current_mode}")

    except Exception as e:
        print(f"  Error checking status: {e}")

    # Run demos
    demo_full_jit_mode()
    demo_gpu_fallback_mode()
    demo_cpu_simulation_mode()

    # Summary
    print_header("Summary")
    print("""
  PyGPUkit supports three runtime modes:

  1. FULL JIT MODE (CUDA + NVRTC)
     - All features available
     - Custom JIT kernels work
     - Pre-compiled ops work
     - Best performance

  2. GPU FALLBACK MODE (Driver only)
     - Pre-compiled ops work (matmul, add, mul)
     - Custom JIT kernels NOT available
     - GPU memory/scheduling work
     - NVRTC optional for JIT

  3. CPU SIMULATION MODE (No GPU)
     - Full API compatibility
     - Runs on CPU via NumPy
     - For testing/development
     - No GPU required

  Check your mode with:
    import pygpukit as gp
    print(f"CUDA: {gp.is_cuda_available()}")
    print(f"NVRTC: {gp.is_nvrtc_available()}")
    print(f"NVRTC Path: {gp.get_nvrtc_path()}")
""")


if __name__ == "__main__":
    main()
