"""Sampling utilities for LLM inference.

Provides token sampling with temperature, top-k, and top-p.
"""

from __future__ import annotations

import numpy as np


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p.

    Args:
        logits: Logits array [vocab_size]
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative prob <= top_p (1.0 = disabled)

    Returns:
        Sampled token ID
    """
    # Apply temperature
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # Convert to probabilities
    logits_max = logits.max()
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum()

    # Top-k filtering
    if top_k > 0 and top_k < len(probs):
        top_k_indices = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_k_indices] = True
        probs = np.where(mask, probs, 0.0)
        probs_sum = probs.sum()
        probs = probs / probs_sum

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        cutoff_idx = min(cutoff_idx, len(sorted_probs))
        mask = np.zeros_like(probs, dtype=bool)
        mask[sorted_indices[:cutoff_idx]] = True
        probs = np.where(mask, probs, 0.0)
        probs_sum = probs.sum()
        probs = probs / probs_sum

    # Sample
    if temperature == 0:
        return int(np.argmax(probs))
    else:
        return int(np.random.choice(len(probs), p=probs))
