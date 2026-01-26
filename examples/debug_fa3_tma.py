"""Debug FA3 TMA kernel launch."""
import os
# Force TMA path
os.environ["PYGPUKIT_FA3_TMA"] = "1"
os.environ["PYGPUKIT_FA3"] = "0"
os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"

import numpy as np
import pygpukit as gpk
from pygpukit.core.backend import get_native_module
from pygpukit.core.dtypes import DataType

native = get_native_module()

# Check device info
print(f"Device capabilities available in native module")
# Check what's available
if hasattr(native, 'get_device_properties'):
    props = native.get_device_properties()
    print(f"Device props: {props}")
elif hasattr(native, 'get_sm_version'):
    print(f"SM version: {native.get_sm_version()}")

# Test: 512 blocks (16 heads, 32 Q tiles = seq_len 1024)
num_heads, seq_len, head_dim = 16, 1024, 128

# Create inputs
np.random.seed(42)
Q_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
K_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
V_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)

bf16 = DataType.from_string("bfloat16")
Q = gpk.from_numpy(Q_np).astype(bf16)
K = gpk.from_numpy(K_np).astype(bf16)
V = gpk.from_numpy(V_np).astype(bf16)

print(f"\nInput shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
print(f"Dtype: {Q.dtype}")

# Try to run SDPA
print("\nRunning SDPA...")
try:
    from pygpukit.ops.nn import sdpa_causal
    out = sdpa_causal(Q, K, V)
    native.device_synchronize()
    print(f"Output shape: {out.shape}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
