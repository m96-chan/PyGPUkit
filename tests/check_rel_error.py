#!/usr/bin/env python3
"""Check GEMV kernel relative error vs FP32.

Formula: Rel.Err = ||C_test - C_fp32|| / ||C_fp32||
"""

import argparse

import numpy as np

from pygpukit.core import from_numpy, zeros
from pygpukit.core.backend import get_native_module


def rel_error(C_test: np.ndarray, C_fp32: np.ndarray) -> float:
    """Rel.Err = ||C_test - C_fp32|| / ||C_fp32||"""
    return np.linalg.norm(C_test - C_fp32) / np.linalg.norm(C_fp32)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(val: np.ndarray) -> np.ndarray:
    bits = val.astype(np.float32).view(np.uint32)
    return ((bits >> 16) & 0xFFFF).astype(np.uint16)


def float_to_fp8(val: np.ndarray) -> np.ndarray:
    val = np.clip(val.astype(np.float32), -448.0, 448.0)
    result = np.zeros(val.shape, dtype=np.uint8)
    sign_mask = (val < 0).astype(np.uint8) * 0x80
    abs_val = np.abs(val)
    zero_mask = abs_val == 0
    f32_bits = abs_val.view(np.uint32)
    exp_f32 = (f32_bits >> 23) & 0xFF
    mant_f32 = f32_bits & 0x7FFFFF
    e_fp8 = np.clip(exp_f32.astype(np.int32) - 120, 0, 15)
    m_fp8 = (mant_f32 >> 20).astype(np.uint8)
    result = sign_mask | (e_fp8.astype(np.uint8) << 3) | m_fp8
    result[zero_mask] = sign_mask[zero_mask]
    return result


def quantize_blockwise(data: np.ndarray, block_size: int):
    if data.ndim == 1:
        K = data.shape[0]
        n_blocks = (K + block_size - 1) // block_size
        fp8_data = np.zeros(K, dtype=np.uint8)
        scales = np.zeros(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            start, end = i * block_size, min((i + 1) * block_size, K)
            block = data[start:end]
            max_val = np.max(np.abs(block))
            scale = max_val / 448.0 if max_val > 0 else 1.0
            scales[i] = scale
            if max_val > 0:
                fp8_data[start:end] = float_to_fp8(block / scale)
        return fp8_data, scales
    else:
        N, K = data.shape
        n_blocks_n = (N + block_size - 1) // block_size
        n_blocks_k = (K + block_size - 1) // block_size
        fp8_data = np.zeros((N, K), dtype=np.uint8)
        scales = np.zeros((n_blocks_n, n_blocks_k), dtype=np.float32)
        for ni in range(n_blocks_n):
            for ki in range(n_blocks_k):
                n_start, n_end = ni * block_size, min((ni + 1) * block_size, N)
                k_start, k_end = ki * block_size, min((ki + 1) * block_size, K)
                block = data[n_start:n_end, k_start:k_end]
                max_val = np.max(np.abs(block))
                scale = max_val / 448.0 if max_val > 0 else 1.0
                scales[ni, ki] = scale
                if max_val > 0:
                    fp8_data[n_start:n_end, k_start:k_end] = float_to_fp8(block / scale)
        return fp8_data, scales


def check_bf16(native, A_f32, B_f32, C_fp32):
    _, N = len(A_f32), len(C_fp32)
    A_bf16 = f32_to_bf16(A_f32)
    B_bf16 = f32_to_bf16(B_f32.T.copy())  # gemv_bf16 uses B[K,N]
    A_gpu = from_numpy(A_bf16)
    B_gpu = from_numpy(B_bf16)
    C_gpu = zeros((N,), dtype="bfloat16")
    native.gemv_bf16(A_gpu._get_native(), B_gpu._get_native(), C_gpu._get_native())
    native.device_synchronize()
    return rel_error(bf16_to_f32(C_gpu.to_numpy()), C_fp32)


def check_w8a8(native, A_f32, B_f32, C_fp32):
    N = len(C_fp32)
    block = 128
    A_fp8, sA = quantize_blockwise(A_f32, block)
    B_fp8, sB = quantize_blockwise(B_f32, block)
    A_gpu = from_numpy(A_fp8)
    B_gpu = from_numpy(B_fp8)
    sA_gpu = from_numpy(sA)
    sB_gpu = from_numpy(sB.flatten())
    C_gpu = zeros((N,), dtype="bfloat16")
    native.gemv_fp8_fp8_bf16_sm120(
        A_gpu._get_native(),
        B_gpu._get_native(),
        sA_gpu._get_native(),
        sB_gpu._get_native(),
        C_gpu._get_native(),
    )
    native.device_synchronize()
    return rel_error(bf16_to_f32(C_gpu.to_numpy()), C_fp32)


def check_w8a16(native, A_f32, B_f32, C_fp32):
    N, K = B_f32.shape
    block = 128  # Kernel expects [N/128, K/128] scales
    A_bf16 = f32_to_bf16(A_f32)

    # Blockwise quantization for B: [N/128, K/128] scales
    n_blocks_n = (N + block - 1) // block
    n_blocks_k = (K + block - 1) // block
    B_fp8 = np.zeros((N, K), dtype=np.uint8)
    sB_f32 = np.zeros((n_blocks_n, n_blocks_k), dtype=np.float32)

    for ni in range(n_blocks_n):
        for ki in range(n_blocks_k):
            n_start, n_end = ni * block, min((ni + 1) * block, N)
            k_start, k_end = ki * block, min((ki + 1) * block, K)
            blk = B_f32[n_start:n_end, k_start:k_end]
            max_val = np.max(np.abs(blk))
            scale = max_val / 448.0 if max_val > 0 else 1.0
            sB_f32[ni, ki] = scale
            if max_val > 0:
                B_fp8[n_start:n_end, k_start:k_end] = float_to_fp8(blk / scale)

    sB_bf16 = f32_to_bf16(sB_f32.flatten())
    A_gpu = from_numpy(A_bf16)
    B_gpu = from_numpy(B_fp8)
    sB_gpu = from_numpy(sB_bf16)
    C_gpu = zeros((N,), dtype="bfloat16")
    native.gemv_fp8_bf16_opt(
        A_gpu._get_native(), B_gpu._get_native(), sB_gpu._get_native(), C_gpu._get_native()
    )
    native.device_synchronize()
    return rel_error(bf16_to_f32(C_gpu.to_numpy()), C_fp32)


def main():
    parser = argparse.ArgumentParser(description="Check GEMV relative error vs FP32")
    parser.add_argument("--kernel", default="all", help="Kernel: bf16, w8a16, w8a8, all")
    parser.add_argument("--sizes", default="4096,4096", help="K,N sizes")
    args = parser.parse_args()

    K, N = map(int, args.sizes.split(","))
    np.random.seed(42)
    A_f32 = np.random.randn(K).astype(np.float32) * 0.5
    B_f32 = np.random.randn(N, K).astype(np.float32) * 0.5
    C_fp32 = B_f32 @ A_f32

    native = get_native_module()

    print("=" * 60)
    print(f"GEMV Relative Error Check (K={K}, N={N})")
    print("Formula: Rel.Err = ||C_test - C_fp32|| / ||C_fp32||")
    print("=" * 60)

    kernels = {
        "bf16": ("BF16", check_bf16),
        "w8a16": ("W8A16", check_w8a16),
        "w8a8": ("W8A8", check_w8a8),
    }

    if args.kernel == "all":
        to_check = kernels.keys()
    else:
        to_check = [args.kernel]

    print(f"\n{'Kernel':<10} {'Rel.Err (vs FP32)':<20}")
    print("-" * 30)

    for k in to_check:
        if k not in kernels:
            print(f"{k:<10} NOT FOUND")
            continue
        name, fn = kernels[k]
        try:
            err = fn(native, A_f32, B_f32, C_fp32)
            print(f"{name:<10} {err * 100:.2f}%")
        except Exception as e:
            print(f"{name:<10} ERROR: {e}")

    print("-" * 30)


if __name__ == "__main__":
    main()
