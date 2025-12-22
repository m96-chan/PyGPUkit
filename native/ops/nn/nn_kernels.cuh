/**
 * Neural Network operation kernels
 *
 * This file includes all NN kernel headers for convenience.
 * Individual kernel files can also be included directly.
 *
 * Refactored for better modularity - each kernel category now lives
 * in its own header file.
 */
#pragma once

// Activation functions (GELU, SiLU)
#include "activation_kernels.cuh"

// Normalization layers (LayerNorm, RMSNorm)
#include "norm_kernels.cuh"

// Softmax
#include "softmax_kernels.cuh"

// Attention (SDPA causal)
#include "attention_kernels.cuh"

// Memory operations (transpose, copy, concat)
#include "memory_kernels.cuh"

// KV cache operations
#include "kv_cache_kernels.cuh"

// Embedding lookup
#include "embedding_kernels.cuh"

// Elementwise operations (bias, RoPE, inplace ops)
#include "elementwise_kernels.cuh"
