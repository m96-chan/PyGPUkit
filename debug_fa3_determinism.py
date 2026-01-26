"""
Debug FA3 TMA Non-Determinism Bug

Runs the kernel multiple times and compares results to identify
which specific elements are non-deterministic.
"""
import os
import sys

# Force TMA path
os.environ["PYGPUKIT_FA3_TMA"] = "1"
os.environ["PYGPUKIT_FA3"] = "0"
os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"

import numpy as np
import pygpukit as gpk
from pygpukit.core.backend import get_native_module
from pygpukit.core.dtypes import DataType

native = get_native_module()


def run_kernel(Q, K, V, bf16, num_heads, seq_len, head_dim):
    """Run FA3 TMA kernel and return FP32 numpy result."""
    out = gpk.zeros((num_heads, seq_len, head_dim), dtype=bf16)
    native.sdpa_causal_timed(Q._native, K._native, V._native, out._native, 0.0)
    native.device_synchronize()
    return out.astype(DataType.from_string("float32")).to_numpy()


def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    num_heads = 32
    head_dim = 128

    print("=" * 60)
    print("FA3 TMA Determinism Debug")
    print("=" * 60)
    print(f"  seq_len   = {seq_len}")
    print(f"  num_heads = {num_heads}")
    print(f"  head_dim  = {head_dim}")
    print(f"  num_runs  = {num_runs}")
    print()

    # Create fixed inputs
    np.random.seed(42)
    Q_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    K_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    V_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)

    bf16 = DataType.from_string("bfloat16")
    Q = gpk.from_numpy(Q_np).astype(bf16)
    K = gpk.from_numpy(K_np).astype(bf16)
    V = gpk.from_numpy(V_np).astype(bf16)

    # Clear cache for fresh start
    native.clear_tma_cache()

    # Run kernel multiple times
    results = []
    for i in range(num_runs):
        result = run_kernel(Q, K, V, bf16, num_heads, seq_len, head_dim)
        results.append(result)
        print(f"Run {i+1}/{num_runs} done")

    print()

    # Compare all runs against the first
    reference = results[0]

    for run_idx in range(1, num_runs):
        diff = np.abs(results[run_idx] - reference)
        max_diff = np.max(diff)

        if max_diff > 1e-6:
            # Find locations with differences
            diff_mask = diff > 1e-6
            diff_locations = np.argwhere(diff_mask)

            print(f"Run {run_idx+1} vs Run 1:")
            print(f"  Max diff: {max_diff:.6e}")
            print(f"  Num diffs: {len(diff_locations)}")

            # Analyze which heads/positions have diffs
            diff_heads = np.unique(diff_locations[:, 0])
            print(f"  Affected heads: {diff_heads}")

            for head_idx in diff_heads:
                head_mask = diff_locations[:, 0] == head_idx
                head_locs = diff_locations[head_mask]
                q_positions = np.unique(head_locs[:, 1])
                print(f"    Head {head_idx}: Q positions {q_positions[:10]}{'...' if len(q_positions) > 10 else ''}")
                print(f"      Num elements: {len(head_locs)}")

                # Show first few differences
                for loc_idx in range(min(3, len(head_locs))):
                    h, q, d = head_locs[loc_idx]
                    ref_val = reference[h, q, d]
                    run_val = results[run_idx][h, q, d]
                    print(f"      [{h},{q},{d}]: ref={ref_val:.6f}, run={run_val:.6f}, diff={abs(run_val-ref_val):.6e}")
        else:
            print(f"Run {run_idx+1} vs Run 1: IDENTICAL (max_diff={max_diff:.6e})")

    print()

    # Check if all runs are identical
    all_identical = all(np.allclose(r, reference, atol=1e-6) for r in results[1:])
    if all_identical:
        print("RESULT: All runs produced identical output - DETERMINISTIC")
    else:
        print("RESULT: Runs produced different output - NON-DETERMINISTIC")

        # Detailed analysis of the non-deterministic pattern
        print()
        print("Detailed Analysis:")

        # Check if the same elements are always non-deterministic
        non_det_masks = []
        for run_idx in range(1, num_runs):
            diff = np.abs(results[run_idx] - reference)
            non_det_masks.append(diff > 1e-6)

        # Find consistently non-deterministic elements
        if len(non_det_masks) > 1:
            consistent_mask = non_det_masks[0]
            for mask in non_det_masks[1:]:
                consistent_mask = consistent_mask & mask

            consistent_locs = np.argwhere(consistent_mask)
            if len(consistent_locs) > 0:
                print(f"  Consistently non-deterministic elements: {len(consistent_locs)}")
                print(f"  Heads: {np.unique(consistent_locs[:, 0])}")
                print(f"  Q positions: {np.unique(consistent_locs[:, 1])}")


if __name__ == "__main__":
    main()
