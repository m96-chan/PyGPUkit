#pragma once

#include "../core/types.hpp"
#include "../core/memory.hpp"

namespace pygpukit {
namespace ops {

// ============================================================================
// Binary Element-wise Operations
// ============================================================================

// Element-wise addition: c = a + b
void add(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray add(const GPUArray& a, const GPUArray& b);

// Element-wise subtraction: c = a - b
void sub(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray sub(const GPUArray& a, const GPUArray& b);

// Element-wise multiplication: c = a * b
void mul(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray mul(const GPUArray& a, const GPUArray& b);

// Element-wise division: c = a / b
void div(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray div(const GPUArray& a, const GPUArray& b);

// ============================================================================
// Unary Element-wise Operations (float32/float64 only)
// ============================================================================

// Element-wise exponential: c = exp(a)
void exp(const GPUArray& a, GPUArray& c);
GPUArray exp(const GPUArray& a);

// Element-wise natural logarithm: c = log(a)
void log(const GPUArray& a, GPUArray& c);
GPUArray log(const GPUArray& a);

// Element-wise ReLU: c = max(0, a)
void relu(const GPUArray& a, GPUArray& c);
GPUArray relu(const GPUArray& a);

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix multiplication: c = a @ b
// a: (M, K), b: (K, N), c: (M, N)
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray matmul(const GPUArray& a, const GPUArray& b);

// Matrix multiplication with explicit TF32 control
// use_tf32: force TF32 TensorCore path (requires SM >= 80 and float32)
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32);
GPUArray matmul(const GPUArray& a, const GPUArray& b, bool use_tf32);

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum of all elements: returns a scalar GPUArray with shape {1}
GPUArray sum(const GPUArray& a);

// Mean of all elements: returns a scalar GPUArray with shape {1}
GPUArray mean(const GPUArray& a);

// Max of all elements: returns a scalar GPUArray with shape {1}
GPUArray max(const GPUArray& a);

} // namespace ops
} // namespace pygpukit
