/**
 * Common type definitions and conversion helpers
 */
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {

// BF16 conversion helpers (avoid constexpr __host__ issues)
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    unsigned short raw;
    memcpy(&raw, &val, sizeof(raw));
    unsigned int bits = ((unsigned int)raw) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    unsigned int bits;
    memcpy(&bits, &val, sizeof(bits));
    bits += 0x7FFF + ((bits >> 16) & 1);  // Round to nearest even
    unsigned short raw = (unsigned short)(bits >> 16);
    __nv_bfloat16 result;
    memcpy(&result, &raw, sizeof(result));
    return result;
}

} // namespace ops
} // namespace pygpukit
