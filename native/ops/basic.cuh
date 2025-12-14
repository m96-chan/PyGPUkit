#pragma once

#include "../core/types.hpp"
#include "../core/memory.hpp"

namespace pygpukit {
namespace ops {

// Element-wise addition: c = a + b
void add(const GPUArray& a, const GPUArray& b, GPUArray& c);

// Element-wise multiplication: c = a * b
void mul(const GPUArray& a, const GPUArray& b, GPUArray& c);

// Matrix multiplication: c = a @ b
// a: (M, K), b: (K, N), c: (M, N)
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c);

// Matrix multiplication with explicit TF32 control
// use_tf32: force TF32 TensorCore path (requires SM >= 80 and float32)
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32);

// Convenience functions that return new arrays
GPUArray add(const GPUArray& a, const GPUArray& b);
GPUArray mul(const GPUArray& a, const GPUArray& b);
GPUArray matmul(const GPUArray& a, const GPUArray& b);

// Matmul with explicit TF32 control
GPUArray matmul(const GPUArray& a, const GPUArray& b, bool use_tf32);

} // namespace ops
} // namespace pygpukit
