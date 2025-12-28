"""Test Triton with raw pointers from PyGPUkit."""

import numpy as np
import pytest
import triton
import triton.language as tl
import pygpukit._pygpukit_native as native
from pygpukit.triton import from_gpuarray

pytestmark = pytest.mark.gpu  # Requires GPU backend, not CPU simulation


@triton.jit
def add_kernel(
    X,  # pointer
    Y,  # pointer
    Z,  # pointer
    N: tl.constexpr,
):
    """Simple add kernel."""
    pid = tl.program_id(0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)


def test_raw_pointer():
    """Test if Triton can use raw pointers."""
    N = 1024

    # Create PyGPUkit arrays
    x_np = np.arange(N, dtype=np.float32)
    y_np = np.arange(N, dtype=np.float32) * 2

    x = native.from_numpy(x_np)
    y = native.from_numpy(y_np)
    z = native.empty([N], native.Float32)

    print(f"x ptr: {hex(x.data_ptr())}")
    print(f"y ptr: {hex(y.data_ptr())}")
    print(f"z ptr: {hex(z.data_ptr())}")

    # Wrap for Triton
    tx = from_gpuarray(x)
    ty = from_gpuarray(y)
    tz = from_gpuarray(z)

    print(f"\nTritonArray tx: {tx}")
    print(f"tx.dtype: {tx.dtype}")
    print(f"tx.data_ptr(): {hex(tx.data_ptr())}")

    # Try launching with TritonArray wrappers
    grid = ((N + 127) // 128,)
    try:
        add_kernel[grid](tx, ty, tz, N)
        print("\nKernel launched with TritonArray wrappers!")
        native.device_synchronize()

        # Check result
        z_np = z.to_numpy()
        expected = x_np + y_np
        if np.allclose(z_np, expected):
            print("Result CORRECT!")
        else:
            print(f"Result WRONG: {z_np[:10]} vs {expected[:10]}")
    except Exception as e:
        print(f"\nTritonArray wrapper failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_raw_pointer()
