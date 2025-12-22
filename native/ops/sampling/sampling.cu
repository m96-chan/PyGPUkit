/**
 * GPU Sampling Operations Dispatch
 */
#include "sampling_kernels.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include <random>
#include <stdexcept>

namespace pygpukit {
namespace ops {

using namespace sampling;

// Thread-local random generator for GPU sampling
static thread_local std::mt19937 rng(std::random_device{}());
static thread_local std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

// ============================================================================
// Greedy Sampling (Argmax)
// ============================================================================

int sample_greedy(const GPUArray& logits) {
    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::runtime_error("sample_greedy: expected 1D or 2D logits");
    }

    int vocab_size = (logits.ndim() == 1) ? logits.shape()[0] : logits.shape()[1];

    // Allocate result on GPU
    GPUArray result_gpu({1}, DataType::Int32);

    const int block_size = 256;

    cudaStream_t stream = internal::get_capture_stream();

    switch (logits.dtype()) {
        case DataType::Float32:
            sample_argmax_f32_kernel<<<1, block_size, 0, stream>>>(
                static_cast<const float*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size);
            break;
        case DataType::Float16:
            sample_argmax_f16_kernel<<<1, block_size, 0, stream>>>(
                static_cast<const __half*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size);
            break;
        case DataType::BFloat16:
            sample_argmax_bf16_kernel<<<1, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size);
            break;
        default:
            throw std::runtime_error("sample_greedy: unsupported dtype");
    }

    sync_and_check("sample_greedy kernel failed");

    // Copy result to host
    int result;
    memcpy_device_to_host(&result, result_gpu.data(), sizeof(int));
    return result;
}

// ============================================================================
// Multinomial Sampling (Temperature only)
// ============================================================================

int sample_multinomial(const GPUArray& logits, float temperature) {
    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::runtime_error("sample_multinomial: expected 1D or 2D logits");
    }
    if (temperature <= 0.0f) {
        throw std::runtime_error("sample_multinomial: temperature must be > 0");
    }

    int vocab_size = (logits.ndim() == 1) ? logits.shape()[0] : logits.shape()[1];

    // Allocate result on GPU
    GPUArray result_gpu({1}, DataType::Int32);

    // Generate random value on CPU (simple and deterministic)
    float random_val = uniform_dist(rng);

    const int block_size = 256;

    cudaStream_t stream = internal::get_capture_stream();

    switch (logits.dtype()) {
        case DataType::Float32:
            sample_multinomial_f32_kernel<<<1, block_size, 0, stream>>>(
                static_cast<const float*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, temperature, random_val);
            break;
        case DataType::Float16:
            sample_multinomial_f16_kernel<<<1, block_size, 0, stream>>>(
                static_cast<const __half*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, temperature, random_val);
            break;
        case DataType::BFloat16:
            sample_multinomial_bf16_kernel<<<1, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, temperature, random_val);
            break;
        default:
            throw std::runtime_error("sample_multinomial: unsupported dtype");
    }

    sync_and_check("sample_multinomial kernel failed");

    // Copy result to host
    int result;
    memcpy_device_to_host(&result, result_gpu.data(), sizeof(int));
    return result;
}

// ============================================================================
// Top-K Sampling
// ============================================================================

int sample_topk(const GPUArray& logits, int top_k, float temperature) {
    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::runtime_error("sample_topk: expected 1D or 2D logits");
    }
    if (temperature <= 0.0f) {
        throw std::runtime_error("sample_topk: temperature must be > 0");
    }
    if (top_k <= 0) {
        throw std::runtime_error("sample_topk: top_k must be > 0");
    }

    int vocab_size = (logits.ndim() == 1) ? logits.shape()[0] : logits.shape()[1];
    top_k = std::min(top_k, vocab_size);

    // Allocate result on GPU
    GPUArray result_gpu({1}, DataType::Int32);

    // Generate random value on CPU
    float random_val = uniform_dist(rng);

    const int block_size = 256;
    // Shared memory: top_k floats + top_k ints
    size_t shared_mem = top_k * (sizeof(float) + sizeof(int));

    cudaStream_t stream = internal::get_capture_stream();

    switch (logits.dtype()) {
        case DataType::Float32:
            sample_topk_f32_kernel<<<1, block_size, shared_mem, stream>>>(
                static_cast<const float*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, top_k, temperature, random_val);
            break;
        case DataType::Float16:
            sample_topk_f16_kernel<<<1, block_size, shared_mem, stream>>>(
                static_cast<const __half*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, top_k, temperature, random_val);
            break;
        case DataType::BFloat16:
            sample_topk_bf16_kernel<<<1, block_size, shared_mem, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, top_k, temperature, random_val);
            break;
        default:
            throw std::runtime_error("sample_topk: unsupported dtype");
    }

    sync_and_check("sample_topk kernel failed");

    // Copy result to host
    int result;
    memcpy_device_to_host(&result, result_gpu.data(), sizeof(int));
    return result;
}

// ============================================================================
// Top-K Sampling (CUDA Graph compatible)
// ============================================================================

void sample_topk_to_buf(
    const GPUArray& logits,
    GPUArray& result_buf,
    int top_k,
    float temperature,
    float random_val
) {
    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::runtime_error("sample_topk_to_buf: expected 1D or 2D logits");
    }
    if (result_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("sample_topk_to_buf: result_buf must be int32");
    }

    int vocab_size = (logits.ndim() == 1) ? logits.shape()[0] : logits.shape()[1];
    top_k = std::min(top_k, vocab_size);

    const int block_size = 256;
    size_t shared_mem = top_k * (sizeof(float) + sizeof(int));

    cudaStream_t stream = internal::get_capture_stream();

    switch (logits.dtype()) {
        case DataType::Float32:
            sample_topk_f32_kernel<<<1, block_size, shared_mem, stream>>>(
                static_cast<const float*>(logits.data()),
                static_cast<int*>(result_buf.data()),
                vocab_size, top_k, temperature, random_val);
            break;
        case DataType::Float16:
            sample_topk_f16_kernel<<<1, block_size, shared_mem, stream>>>(
                static_cast<const __half*>(logits.data()),
                static_cast<int*>(result_buf.data()),
                vocab_size, top_k, temperature, random_val);
            break;
        case DataType::BFloat16:
            sample_topk_bf16_kernel<<<1, block_size, shared_mem, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data()),
                static_cast<int*>(result_buf.data()),
                vocab_size, top_k, temperature, random_val);
            break;
        default:
            throw std::runtime_error("sample_topk_to_buf: unsupported dtype");
    }
    // No sync - caller is responsible (for CUDA Graph compatibility)
}

// ============================================================================
// Top-K Sampling with Pointer (CUDA Graph replay compatible)
// ============================================================================

void sample_topk_to_buf_ptr(
    const GPUArray& logits,
    GPUArray& result_buf,
    const GPUArray& random_val_buf,
    int top_k,
    float temperature
) {
    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::runtime_error("sample_topk_to_buf_ptr: expected 1D or 2D logits");
    }
    if (result_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("sample_topk_to_buf_ptr: result_buf must be int32");
    }
    if (random_val_buf.dtype() != DataType::Float32) {
        throw std::runtime_error("sample_topk_to_buf_ptr: random_val_buf must be float32");
    }
    if (logits.dtype() != DataType::Float16) {
        throw std::runtime_error("sample_topk_to_buf_ptr: only float16 logits supported (for now)");
    }

    int vocab_size = (logits.ndim() == 1) ? logits.shape()[0] : logits.shape()[1];
    top_k = std::min(top_k, vocab_size);

    const int block_size = 256;
    size_t shared_mem = top_k * (sizeof(float) + sizeof(int));

    cudaStream_t stream = internal::get_capture_stream();

    sample_topk_f16_ptr_kernel<<<1, block_size, shared_mem, stream>>>(
        static_cast<const __half*>(logits.data()),
        static_cast<int*>(result_buf.data()),
        static_cast<const float*>(random_val_buf.data()),
        vocab_size, top_k, temperature);
    // No sync - caller is responsible (for CUDA Graph compatibility)
}

// ============================================================================
// Top-P (Nucleus) Sampling
// ============================================================================

int sample_topp(const GPUArray& logits, float top_p, float temperature) {
    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::runtime_error("sample_topp: expected 1D or 2D logits");
    }
    if (temperature <= 0.0f) {
        throw std::runtime_error("sample_topp: temperature must be > 0");
    }
    if (top_p <= 0.0f || top_p > 1.0f) {
        throw std::runtime_error("sample_topp: top_p must be in (0, 1]");
    }

    int vocab_size = (logits.ndim() == 1) ? logits.shape()[0] : logits.shape()[1];

    // Allocate result on GPU
    GPUArray result_gpu({1}, DataType::Int32);

    // Generate random value on CPU
    float random_val = uniform_dist(rng);

    cudaStream_t stream = internal::get_capture_stream();

    switch (logits.dtype()) {
        case DataType::Float32:
            sample_topp_f32_kernel<<<1, 1, 0, stream>>>(
                static_cast<const float*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, top_p, temperature, random_val);
            break;
        case DataType::Float16:
            sample_topp_f16_kernel<<<1, 1, 0, stream>>>(
                static_cast<const __half*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, top_p, temperature, random_val);
            break;
        case DataType::BFloat16:
            sample_topp_bf16_kernel<<<1, 1, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data()),
                static_cast<int*>(result_gpu.data()),
                vocab_size, top_p, temperature, random_val);
            break;
        default:
            throw std::runtime_error("sample_topp: unsupported dtype");
    }

    sync_and_check("sample_topp kernel failed");

    // Copy result to host
    int result;
    memcpy_device_to_host(&result, result_gpu.data(), sizeof(int));
    return result;
}

// ============================================================================
// Unified Sampling API
// ============================================================================

int sample_token_gpu(
    const GPUArray& logits,
    float temperature,
    int top_k,
    float top_p
) {
    // Greedy sampling
    if (temperature == 0.0f || temperature < 1e-6f) {
        return sample_greedy(logits);
    }

    // Top-k sampling (if k > 0 and k < vocab_size)
    int vocab_size = (logits.ndim() == 1) ? logits.shape()[0] : logits.shape()[1];
    if (top_k > 0 && top_k < vocab_size) {
        return sample_topk(logits, top_k, temperature);
    }

    // Top-p sampling (if p < 1.0)
    if (top_p < 1.0f && top_p > 0.0f) {
        return sample_topp(logits, top_p, temperature);
    }

    // Pure multinomial sampling with temperature
    return sample_multinomial(logits, temperature);
}

// Set random seed for reproducibility
void set_sampling_seed(unsigned int seed) {
    rng.seed(seed);
}

} // namespace ops
} // namespace pygpukit
