"""Self-speculative decode strategy.

This module provides the DecodeSpeculative strategy for self-speculative
decoding using early layers as a draft model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.llm.decode.base import DecodeStrategy

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.buffers import DecodeBuffers


class DecodeSpeculative(DecodeStrategy):
    """Self-speculative decode strategy.

    Uses early transformer layers as a draft model to generate speculative
    tokens, then verifies them with the full model in a single batch pass.

    This can significantly speed up inference when the draft model has
    high acceptance rate.
    """

    def __init__(
        self,
        max_draft_tokens: int = 4,
        draft_layers: int = 8,
    ) -> None:
        """Initialize DecodeSpeculative strategy.

        Args:
            max_draft_tokens: Maximum number of draft tokens to generate.
            draft_layers: Number of early layers to use as draft model.
        """
        super().__init__()
        self._max_draft_tokens = max_draft_tokens
        self._draft_layers = draft_layers

    def step(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Execute speculative decode step.

        Note: This returns hidden states for the last token only.
        Use step_speculative() to get all accepted tokens.

        Args:
            token_id: Current token ID to process.
            position: Position in the sequence.
            context_len: Total context length.
            buffers: Pre-allocated decode buffers (unused for speculative).

        Returns:
            Hidden states [1, hidden_size] for last accepted token.
        """
        # For the base step() interface, just do simple decode
        model = self.model
        return model._decode_step_fixed_cache(token_id, position, context_len)

    def step_speculative(
        self,
        token_id: int,
        position: int,
        context_len: int,
    ) -> tuple[list[int], int, dict]:
        """Execute self-speculative decode step.

        Algorithm:
        1. Snapshot KV cache state
        2. Generate draft tokens using early layers
        3. Verify all draft tokens in one batch forward pass (full model)
        4. Accept tokens until first disagreement
        5. Restore KV cache and re-run for accepted tokens

        Args:
            token_id: Current token ID (the last accepted token).
            position: Position in sequence.
            context_len: Total context length.

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs.
            - new_position: Updated position after accepting tokens.
            - stats: Dict with 'draft_count', 'accepted_count'.
        """
        model = self.model
        max_draft_tokens = self._max_draft_tokens
        draft_layers = self._draft_layers

        # Snapshot KV cache
        kv_snapshot = model.snapshot_kv_cache()

        # Step 1: Generate draft tokens using early layers
        draft_tokens = []
        draft_pos = position
        draft_ctx = context_len
        current_token = token_id

        for _ in range(max_draft_tokens):
            hidden = model._draft_forward_early_layers(
                current_token, draft_pos, draft_ctx, draft_layers
            )
            logits = model._draft_get_logits(hidden)
            logits_np = logits.to_numpy()[-1]
            next_token = int(np.argmax(logits_np))

            draft_tokens.append(next_token)
            current_token = next_token
            draft_pos += 1
            draft_ctx += 1

        # Step 2: Restore KV cache for verification
        model.restore_kv_cache(kv_snapshot)

        # Step 3: Verify with full model in batch
        verify_input = [token_id] + draft_tokens[:-1]
        verify_ctx = position + len(verify_input)

        hidden_batch = model._decode_step_fixed_cache_batch(verify_input, position, verify_ctx)
        verify_logits = model.get_logits(hidden_batch)
        verify_logits_np = verify_logits.to_numpy()

        # Step 4: Accept/Reject tokens
        accepted_tokens = []
        for i, draft_token in enumerate(draft_tokens):
            target_token = int(np.argmax(verify_logits_np[i]))
            if target_token == draft_token:
                accepted_tokens.append(draft_token)
            else:
                accepted_tokens.append(target_token)
                break

        # Step 5: Restore KV and re-run accepted tokens
        model.restore_kv_cache(kv_snapshot)

        new_pos = position
        new_ctx = context_len
        prev_token = token_id

        for acc_token in accepted_tokens:
            model._decode_step_fixed_cache(prev_token, new_pos, new_ctx)
            prev_token = acc_token
            new_pos += 1
            new_ctx += 1

        stats = {
            "draft_count": len(draft_tokens),
            "accepted_count": len(
                [
                    t
                    for i, t in enumerate(accepted_tokens)
                    if i < len(draft_tokens) and t == draft_tokens[i]
                ]
            ),
        }

        return accepted_tokens, new_pos, stats

    def step_lookahead(
        self,
        token_id: int,
    ) -> tuple[list[int], dict]:
        """Self-speculative decode with GPU-side lookahead KV.

        Uses GPU-side KV snapshot (no CPU copies) for faster speculation.
        Uses the model's internal position tracking.

        Args:
            token_id: Current token ID.

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs.
            - stats: Dict with decode statistics.
        """
        # Delegate to model's lookahead method
        return self.model.decode_step_self_speculative_lookahead(
            token_id,
            max_draft_tokens=self._max_draft_tokens,
            draft_layers=self._draft_layers,
        )

    @property
    def max_draft_tokens(self) -> int:
        """Get maximum draft tokens."""
        return self._max_draft_tokens

    @property
    def draft_layers(self) -> int:
        """Get number of draft layers."""
        return self._draft_layers
