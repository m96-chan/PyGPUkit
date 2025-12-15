/**
 * PyGPUkit Operations - Public API
 *
 * This header provides access to all GPU array operations:
 * - Elementwise: add, mul, sub, div
 * - Unary: exp, log, relu
 * - Reduction: sum, mean, max
 * - Matmul: matrix multiplication with TensorCore support
 */
#pragma once

#include "../core/memory.hpp"

namespace pygpukit {
namespace ops {

// ============================================================================
// Elementwise Operations
// ============================================================================

// Add: c = a + b
void add(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray add(const GPUArray& a, const GPUArray& b);

// Mul: c = a * b
void mul(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray mul(const GPUArray& a, const GPUArray& b);

// Sub: c = a - b
void sub(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray sub(const GPUArray& a, const GPUArray& b);

// Div: c = a / b
void div(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray div(const GPUArray& a, const GPUArray& b);

// ============================================================================
// Unary Operations
// ============================================================================

// Exp: c = exp(a)
void exp(const GPUArray& a, GPUArray& c);
GPUArray exp(const GPUArray& a);

// Log: c = log(a)
void log(const GPUArray& a, GPUArray& c);
GPUArray log(const GPUArray& a);

// ReLU: c = max(0, a)
void relu(const GPUArray& a, GPUArray& c);
GPUArray relu(const GPUArray& a);

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum: scalar sum of all elements
GPUArray sum(const GPUArray& a);

// Mean: scalar mean of all elements
GPUArray mean(const GPUArray& a);

// Max: scalar max of all elements
GPUArray max(const GPUArray& a);

// ============================================================================
// Matrix Multiplication
// ============================================================================

// Matmul: c = a @ b
// Automatically selects optimal kernel based on dtype and size:
// - FP32: L2-optimized, tiled, or Ampere-optimized kernel
// - FP32 + PYGPUKIT_ALLOW_TF32=1: TF32 TensorCore kernel
// - FP16/BF16: Simple or TensorCore kernel (PYGPUKIT_ALLOW_FP16_TC=1)
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray matmul(const GPUArray& a, const GPUArray& b);

// Matmul with explicit TF32 control
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32);
GPUArray matmul(const GPUArray& a, const GPUArray& b, bool use_tf32);

// ============================================================================
// Neural Network Operations
// ============================================================================

// Transpose: c = a.T
// input: [rows, cols], output: [cols, rows]
GPUArray transpose(const GPUArray& input);

// GELU: Gaussian Error Linear Unit activation
// y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
GPUArray gelu(const GPUArray& input);

// Bias Add: output[batch, features] += bias[features] (in-place)
void bias_add_inplace(GPUArray& output, const GPUArray& bias);

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// input: [batch, features], gamma/beta: [features]
GPUArray layernorm(const GPUArray& input, const GPUArray& gamma, const GPUArray& beta, float eps = 1e-5f);

} // namespace ops
} // namespace pygpukit
