/**
 * Neural Network operations dispatch
 *
 * Issue #133: This file aggregates all modular dispatch files into a single
 * translation unit to avoid duplicate kernel symbol errors.
 *
 * Modular source files are organized in subdirectories for maintainability
 * but are compiled together here.
 */

// Define this macro to include kernel definitions from kernel headers
#define PYGPUKIT_IMPLEMENT_NN_KERNELS

#include "nn_kernels.cuh"
#include "flash_attention.cuh"
#include "flash_decoding.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include <algorithm>
#include <cstdlib>

// Include all modular dispatch implementations
// These are organized in subdirectories but compiled as one translation unit

#include "activation/gelu.inl"
#include "activation/silu.inl"
#include "activation/sigmoid.inl"
#include "activation/tanh.inl"
#include "activation/relu2.inl"
#include "norm/layernorm.inl"
#include "norm/rmsnorm.inl"
#include "rope/rope_inplace.inl"
#include "rope/rope_ext.inl"
#include "pope/pope.inl"
#include "alibi/alibi.inl"
#include "linear/linear_bias.inl"
#include "attention/sdpa_causal.inl"
#include "tensor/tensor.inl"
#include "embedding/embedding.inl"
#include "elementwise/inplace.inl"
#include "cast/cast.inl"
#include "recurrent/lstm.inl"
#include "diffusion/diffusion.inl"
#include "llama4/llama4.inl"
#include "fused/fused.inl"
