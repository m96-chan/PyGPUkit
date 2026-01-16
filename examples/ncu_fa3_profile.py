"""Simple FA3 TMA profiling script for Nsight Compute."""
import os
# Force TMA path
os.environ["PYGPUKIT_FA3_TMA"] = "1"
os.environ["PYGPUKIT_FA3"] = "0"
os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"

import numpy as np
import pygpukit as gpk
from pygpukit.ops.nn import sdpa_causal
from pygpukit.core.backend import get_native_module
from pygpukit.core.dtypes import DataType

native = get_native_module()

# Single config for profiling [heads, seq_len, head_dim]
num_heads, seq_len, head_dim = 32, 1024, 128

# Create inputs
np.random.seed(42)
Q_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
K_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
V_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)

bf16 = DataType.from_string("bfloat16")
Q = gpk.from_numpy(Q_np).astype(bf16)
K = gpk.from_numpy(K_np).astype(bf16)
V = gpk.from_numpy(V_np).astype(bf16)

# Warmup
for _ in range(3):
    out = sdpa_causal(Q, K, V)
native.device_synchronize()

# Profile target (single run)
out = sdpa_causal(Q, K, V)
native.device_synchronize()

print("Profile complete")
