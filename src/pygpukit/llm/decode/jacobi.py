"""Jacobi decode strategy.

This module provides the DecodeJacobi strategy for parallel iterative
decoding without a draft model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pygpukit.llm.decode.base import DecodeStrategy

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.buffers import DecodeBuffers


class DecodeJacobi(DecodeStrategy):
    """Jacobi decode strategy for parallel iterative decoding.

    Jacobi decoding generates multiple tokens in parallel by iterating:
    1. Initialize N future positions with a guess
    2. Batch forward pass on all N positions
    3. Update each position with argmax(logits)
    4. Repeat until convergence or max_iter
    5. Accept converged tokens

    Unlike speculative decoding, Jacobi doesn't use a separate draft model.
    Instead, it relies on the iterative refinement of guesses to converge.
    """

    def __init__(
        self,
        n_tokens: int = 8,
        max_iter: int = 3,
        init_strategy: Literal["repeat", "ngram", "greedy"] = "repeat",
    ) -> None:
        """Initialize DecodeJacobi strategy.

        Args:
            n_tokens: Number of tokens to decode in parallel.
            max_iter: Maximum iterations for convergence.
            init_strategy: How to initialize guess tokens.
                - "repeat": Repeat last token (fast, simple).
                - "ngram": Use n-gram cache if available.
                - "greedy": Run greedy decode first (slow but accurate).
        """
        super().__init__()
        self._n_tokens = n_tokens
        self._max_iter = max_iter
        self._init_strategy = init_strategy

    def step(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Execute Jacobi decode step.

        Note: This returns hidden states for the last token only.
        Use step_jacobi() to get all accepted tokens.

        Args:
            token_id: Current token ID to process.
            position: Position in the sequence.
            context_len: Total context length.
            buffers: Pre-allocated decode buffers (unused for jacobi).

        Returns:
            Hidden states [1, hidden_size] for last accepted token.
        """
        # For the base step() interface, just do simple decode
        model = self.model
        return model._decode_step_fixed_cache(token_id, position, context_len)

    def step_jacobi(
        self,
        token_id: int,
        position: int,
        context_len: int,
    ) -> tuple[list[int], int, dict]:
        """Execute Jacobi decode step.

        Args:
            token_id: Current token ID (the last accepted token).
            position: Position in sequence.
            context_len: Total context length.

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs.
            - new_position: Updated position after accepting tokens.
            - stats: Dict with 'iterations', 'converged', 'accepted_count'.
        """
        model = self.model
        n_tokens = self._n_tokens
        max_iter = self._max_iter
        init_strategy = self._init_strategy

        # Snapshot KV cache
        kv_snapshot = model.snapshot_kv_cache()

        # Initialize guess
        guess = model._init_jacobi_guess(token_id, position, context_len, n_tokens, init_strategy)

        iterations_used = 0
        converged = False
        prev_guess = None

        for iteration in range(max_iter):
            iterations_used = iteration + 1

            # Restore KV to clean state
            model.restore_kv_cache(kv_snapshot)

            # Batch forward
            input_tokens = [token_id] + guess[:-1]
            verify_ctx = position + len(input_tokens)

            hidden = model._decode_step_fixed_cache_batch(input_tokens, position, verify_ctx)
            logits = model.get_logits(hidden)
            logits_np = logits.to_numpy()

            # Update guess with argmax
            new_guess = [int(np.argmax(logits_np[i])) for i in range(n_tokens)]

            # Check convergence
            if new_guess == guess:
                converged = True
                break

            prev_guess = guess
            guess = new_guess

        # Find longest converged prefix
        if converged:
            accepted_tokens = guess
        else:
            accepted_tokens = []
            if prev_guess is not None:
                for i in range(n_tokens):
                    if guess[i] == prev_guess[i]:
                        accepted_tokens.append(guess[i])
                    else:
                        break
            if len(accepted_tokens) == 0:
                accepted_tokens = [guess[0]]

        # Restore KV and re-run to update cache
        model.restore_kv_cache(kv_snapshot)

        new_pos = position
        new_ctx = context_len
        prev_token = token_id

        for acc_token in accepted_tokens:
            model._decode_step_fixed_cache(prev_token, new_pos, new_ctx)
            prev_token = acc_token
            new_pos += 1
            new_ctx += 1

        # Update n-gram cache
        if not hasattr(model, "_ngram_cache"):
            model._ngram_cache: dict[int, list[int]] = {}
        model._ngram_cache[token_id] = accepted_tokens.copy()

        stats = {
            "iterations": iterations_used,
            "converged": converged,
            "accepted_count": len(accepted_tokens),
        }

        return accepted_tokens, new_pos, stats

    def step_lookahead(
        self,
        token_id: int,
    ) -> tuple[list[int], dict]:
        """Jacobi decode with GPU-side lookahead KV (no CPU copies).

        Uses GPU-side KV snapshot for faster iteration.
        Uses the model's internal position tracking.

        Args:
            token_id: Current token ID.

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs.
            - stats: Dict with decode statistics.
        """
        # Delegate to model's lookahead method
        return self.model.decode_step_jacobi_lookahead(
            token_id,
            n_tokens=self._n_tokens,
            max_iter=self._max_iter,
            init_strategy=self._init_strategy,
        )

    @property
    def n_tokens(self) -> int:
        """Get number of parallel tokens."""
        return self._n_tokens

    @property
    def max_iter(self) -> int:
        """Get maximum iterations."""
        return self._max_iter

    @property
    def init_strategy(self) -> str:
        """Get initialization strategy."""
        return self._init_strategy
