"""GPU sampling operations for GPUArrays.

Corresponds to native/ops/sampling/.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray


def sample_token_gpu(
    logits: GPUArray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits on GPU.

    Performs sampling entirely on GPU, avoiding D2H transfer of full logits.
    Only returns the single sampled token ID.

    Sampling method selection:
    - temperature=0: greedy (argmax)
    - top_k > 0: top-k sampling
    - top_p < 1: top-p (nucleus) sampling
    - otherwise: multinomial with temperature

    Args:
        logits: Logits tensor [vocab_size] or [1, vocab_size].
        temperature: Sampling temperature (>0, lower = more deterministic).
        top_k: If >0, only sample from top-k tokens.
        top_p: If <1, sample from smallest set with cumulative prob >= top_p.

    Returns:
        Sampled token ID (int).
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    logits_native = logits._get_native()
    return native.sample_token_gpu(logits_native, temperature, top_k, top_p)


def sample_topk_to_buf_ptr(
    logits: GPUArray,
    result_buf: GPUArray,
    random_val_buf: GPUArray,
    top_k: int,
    temperature: float,
) -> None:
    """Top-K sampling with pointer (CUDA Graph replay compatible).

    Reads random_val from GPU buffer, allowing update before Graph replay.
    Result is written to pre-allocated buffer (no D2H copy).

    Args:
        logits: Logits tensor [vocab_size] or [1, vocab_size] (float16 only).
        result_buf: Pre-allocated int32 buffer [1] for sampled token ID.
        random_val_buf: Pre-allocated float32 buffer [1] for random value.
        top_k: Number of top tokens to consider.
        temperature: Sampling temperature (>0).
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    native.sample_topk_to_buf_ptr(
        logits._get_native(),
        result_buf._get_native(),
        random_val_buf._get_native(),
        top_k,
        temperature,
    )


def sample_greedy(logits: GPUArray) -> int:
    """Greedy sampling (argmax) from logits on GPU.

    Args:
        logits: Logits tensor [vocab_size] or [1, vocab_size].

    Returns:
        Token ID with highest logit value.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    logits_native = logits._get_native()
    return native.sample_greedy(logits_native)


def sample_multinomial(logits: GPUArray, temperature: float) -> int:
    """Multinomial sampling with temperature on GPU.

    Args:
        logits: Logits tensor [vocab_size] or [1, vocab_size].
        temperature: Sampling temperature (>0).

    Returns:
        Sampled token ID.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    logits_native = logits._get_native()
    return native.sample_multinomial(logits_native, temperature)


def sample_topk(logits: GPUArray, top_k: int, temperature: float) -> int:
    """Top-K sampling on GPU.

    Args:
        logits: Logits tensor [vocab_size] or [1, vocab_size].
        top_k: Number of top tokens to consider.
        temperature: Sampling temperature (>0).

    Returns:
        Sampled token ID from top-k.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    logits_native = logits._get_native()
    return native.sample_topk(logits_native, top_k, temperature)


def sample_topp(logits: GPUArray, top_p: float, temperature: float) -> int:
    """Top-P (nucleus) sampling on GPU.

    Args:
        logits: Logits tensor [vocab_size] or [1, vocab_size].
        top_p: Cumulative probability threshold (0 < p <= 1).
        temperature: Sampling temperature (>0).

    Returns:
        Sampled token ID from nucleus.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    logits_native = logits._get_native()
    return native.sample_topp(logits_native, top_p, temperature)


def set_sampling_seed(seed: int) -> None:
    """Set random seed for GPU sampling.

    Args:
        seed: Random seed for reproducibility.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    native.set_sampling_seed(seed)
