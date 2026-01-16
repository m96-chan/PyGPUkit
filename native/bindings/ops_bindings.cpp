/**
 * PyGPUkit Operations Bindings - Main Entry Point
 *
 * This file calls all init functions from the modular binding files.
 * Each category is in its own subdirectory for better organization.
 */
#include "bindings_common.hpp"

void init_ops_bindings(py::module_& m) {
    // Elementwise operations
    init_elementwise_binary(m);
    init_elementwise_inplace(m);
    init_elementwise_compare(m);

    // Unary operations
    init_unary_math(m);
    init_unary_trig(m);

    // Reduction operations
    init_reduction_basic(m);
    init_reduction_argmax(m);
    init_reduction_softmax(m);

    // Tensor operations
    init_tensor_cast(m);
    init_tensor_transpose(m);
    init_tensor_reshape(m);
    init_tensor_repeat(m);

    // Neural network operations
    init_nn_activation(m);
    init_nn_norm(m);
    init_nn_attention(m);
    init_nn_rope(m);
    init_nn_recurrent(m);
    init_nn_diffusion(m);
    init_nn_conv(m);
    init_nn_llama4(m);

    // Embedding operations
    init_embedding_lookup(m);
    init_embedding_kv_cache(m);

    // GEMM operations (by dtype combination)
    init_gemm_generic(m);
    init_gemm_fp8xfp8_bf16(m);
    init_gemm_fp8xfp8_fp8(m);
    init_gemm_fp8xbf16_bf16(m);
    init_gemm_nvf4xbf16_bf16(m);
    init_gemm_grouped(m);
    init_gemm_int(m);

    // GEMV operations
    init_gemv_generic(m);
    init_gemv_fp8xfp8_bf16(m);
    init_gemv_nvf4xbf16_bf16(m);

    // Sampling operations
    init_sampling_basic(m);
    init_sampling_topk(m);
    init_sampling_seed(m);

    // Quantization operations
    init_quantize(m);

    // Attention operations
    init_paged_attention(m);

    // Continuous batching operations
    init_continuous_batching(m);

    // Audio processing operations
    init_audio(m);

    // cuBLASLt utility functions
    init_cublaslt(m);

    // cuBLAS utility functions
    init_cublas(m);

    // MoE (Mixture of Experts) operations
    init_moe(m);
}
