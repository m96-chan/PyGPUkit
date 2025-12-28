"""
Triton kernel implementations.

These are not optimized for maximum performance.
Focus: rapid prototyping and iteration for kernel development PoC.

All kernels use TritonArray wrapper for PyTorch-free operation.
All kernels use in-place output (pre-allocated `out` parameter).
"""

from .rmsnorm import rmsnorm
from .layernorm import layernorm
from .softmax import softmax
from .rotary import rotary

__all__ = [
    "rmsnorm",
    "layernorm",
    "softmax",
    "rotary",
]
