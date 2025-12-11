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

// Convenience functions that return new arrays
GPUArray add(const GPUArray& a, const GPUArray& b);
GPUArray mul(const GPUArray& a, const GPUArray& b);
GPUArray matmul(const GPUArray& a, const GPUArray& b);

} // namespace ops
} // namespace pygpukit
