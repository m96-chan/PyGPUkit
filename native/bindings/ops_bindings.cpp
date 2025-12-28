#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../ops/ops.cuh"
#include "../ops/audio/audio.hpp"
#include "../jit/cublaslt_loader.hpp"

namespace py = pybind11;
using namespace pygpukit;

// Extern declarations for FP8 functions (must be at global scope)
extern "C" {
    // SM90 (Hopper) - FP8 with per-tensor scaling
    cudaError_t pygpukit_gemm_fp8_sm90(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_sm90_available();

    // SM100 (Blackwell datacenter) - FP8 with blockwise scaling
    cudaError_t pygpukit_gemm_fp8_sm100(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_sm100_available();

    // SM120 (Blackwell GeForce) - FP8 with blockwise scaling (disabled due to CUTLASS bug #2902)
    cudaError_t pygpukit_gemm_fp8_sm120(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_sm120_available();

    // SM120 (Blackwell GeForce) - Pure FP8 I/O GEMM
    cudaError_t pygpukit_gemm_fp8_fp8_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_fp8_sm120_available();

    // SM120 (Blackwell GeForce) - Pure FP8 I/O GEMM with blockwise scaling
    cudaError_t pygpukit_gemm_fp8_fp8_blockwise_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        const float* scale_A, const float* scale_B,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    void pygpukit_fp8_fp8_get_scale_sizes(
        int M, int N, int K,
        size_t* sfa_size, size_t* sfb_size
    );

    // SM120 FP8 GEMM tile variants (V2-V4)
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v2(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v3(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v4(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);

    // SM120 (Blackwell GeForce) - NVF4 (4-bit) with BF16 I/O
    cudaError_t pygpukit_gemm_nvf4_bf16_sm120(
        const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_nvf4_bf16_sm120_available();

    // SM120 (Blackwell GeForce) - Pure NVF4 GEMM (for benchmarking)
    cudaError_t pygpukit_benchmark_gemm_nvf4_sm120(
        __nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_nvf4_nvf4_sm120_available();

    // NVF4 GEMV for SM120
    bool pygpukit_gemv_nvf4_available();
    cudaError_t pygpukit_quantize_bf16_to_nvf4(
        const void* input, void* out_data, void* out_scale,
        int K, int N, cudaStream_t stream
    );
    // Row-major version for pure NVF4/NVF4 GEMV (coalesced memory access)
    cudaError_t pygpukit_quantize_bf16_to_nvf4_rowmajor(
        const void* input, void* out_data, void* out_scale,
        int K, int N, cudaStream_t stream
    );
    cudaError_t pygpukit_gemv_nvf4_bf16(
        const void* A, const void* B_data, const void* B_scale, void* C,
        int K, int N, float alpha, cudaStream_t stream
    );
    cudaError_t pygpukit_gemv_bf16(
        const void* A, const void* B, void* C,
        int K, int N, float alpha, float beta, cudaStream_t stream
    );
    void pygpukit_nvf4_get_sizes(int K, int N, size_t* data_size, size_t* scale_size);

    // W8A16 GEMM: FP8 weight x BF16 activation -> BF16 output
    cudaError_t pygpukit_w8a16_gemm_sm120(
        const void* A, const void* B_fp8, const void* B_scale, void* C,
        int M, int N, int K, int scale_stride_n, cudaStream_t stream
    );
    // W8A16 GEMM using CUTLASS: BF16 activation -> quantize to FP8 -> FP8xFP8 GEMM -> BF16 output
    cudaError_t pygpukit_w8a16_cutlass_sm120(
        const void* A, const void* B, void* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    // W8A16 GEMM using blockwise scaling (same compilation unit as working fp8_blockwise)
    cudaError_t pygpukit_w8a16_blockwise_sm120(
        const void* A, const void* B, void* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    // Optimized W8A16 GEMM: BF16 activations x FP8 weights -> BF16 output (uses fast FP8xFP8 internally)
    cudaError_t pygpukit_gemm_w8a16_optimized_sm120(
        const void* A_bf16, const uint8_t* B_fp8, void* D_bf16,
        const float* scale_A, const float* scale_B,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    // Grouped GEMM for MoE: FP8 weights x BF16 activations -> BF16 output
    cudaError_t pygpukit_grouped_gemm_init_lut();
    cudaError_t pygpukit_grouped_gemm_fp8_bf16(
        const void* A, const void* B_stacked, const void* B_scale,
        void* C, const int* row_expert_ids,
        int M, int N, int K, cudaStream_t stream
    );

    // Native Int8 GEMM using dp4a CUDA cores (exact, no FP8 approximation)
    cudaError_t pygpukit_gemm_int8_native_sm120(
        const int8_t* A, const int8_t* B, int32_t* D,
        int M, int N, int K,
        cudaStream_t stream
    );
    bool pygpukit_int8_native_gemm_available();

    // Int4 GEMM via Int8/FP8 approximation (SM120 has no native Int4 TensorCore)
    cudaError_t pygpukit_gemm_int4_int4_int32_sm120(
        const uint8_t* A_packed, const uint8_t* B_packed, int32_t* D,
        int M, int N, int K,
        float scale_A, float scale_B, float descale_D,
        cudaStream_t stream
    );
    cudaError_t pygpukit_gemm_int4_int4_int8_sm120(
        const uint8_t* A_packed, const uint8_t* B_packed, int8_t* D,
        int M, int N, int K,
        float scale_A, float scale_B, float descale_D,
        cudaStream_t stream
    );
    bool pygpukit_int4_gemm_sm120_available();

    // Int4 GEMV for M=1 decode (SM120)
    cudaError_t pygpukit_gemv_int4_int4_int32_sm120(
        const uint8_t* A, const uint8_t* B_nk, int32_t* C,
        int K, int N,
        float scale_A, float scale_B,
        cudaStream_t stream
    );
    bool pygpukit_int4_gemv_sm120_available();

    // Pure FP8/FP8/FP8 GEMV (SM120)
    cudaError_t pygpukit_gemv_fp8_fp8_bf16_sm120(
        const uint8_t* A, const uint8_t* B_nk,
        const float* scale_A, const float* scale_B,
        __nv_bfloat16* C,
        int K, int N, cudaStream_t stream
    );
    cudaError_t pygpukit_gemv_fp8_fp8_fp8_sm120(
        const uint8_t* A, const uint8_t* B_nk,
        const float* scale_A, const float* scale_B,
        uint8_t* C, float scale_C,
        int K, int N, cudaStream_t stream
    );
    bool pygpukit_gemv_fp8_fp8_sm120_available();

    // Pure NVF4/NVF4/NVF4 GEMV (SM120)
    cudaError_t pygpukit_gemv_nvf4_nvf4_bf16_sm120(
        const uint8_t* A_data, const uint8_t* A_scale,
        const uint8_t* B_data, const uint8_t* B_scale,
        __nv_bfloat16* C,
        int K, int N, cudaStream_t stream
    );
    bool pygpukit_gemv_nvf4_nvf4_sm120_available();
}

// Optimized FP8 GEMV (warp-level reduction, smem, vectorized)
namespace pygpukit {
namespace ops {
namespace gemv {
    cudaError_t launch_gemv_fp8_opt(
        const __nv_bfloat16* A, const uint8_t* B_nk, const __nv_bfloat16* B_scale,
        __nv_bfloat16* C, int K, int N, cudaStream_t stream
    );
    cudaError_t launch_gemv_fp8_opt_batched(
        const __nv_bfloat16* A, const uint8_t* B_nk, const __nv_bfloat16* B_scale,
        __nv_bfloat16* C, int K, int N, int batch_count, cudaStream_t stream
    );
}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit

// MoE (Mixture of Experts) functions - defined in ops/moe/moe.cu
namespace pygpukit {
namespace moe {
    void topk_with_indices_f32(
        const float* logits, float* values, int32_t* indices,
        int num_tokens, int num_experts, int k, cudaStream_t stream);
    void topk_with_indices_bf16(
        const __nv_bfloat16* logits, __nv_bfloat16* values, int32_t* indices,
        int num_tokens, int num_experts, int k, cudaStream_t stream);
    void softmax_topk_f32(float* values, int num_tokens, int k, cudaStream_t stream);
    void softmax_topk_bf16(__nv_bfloat16* values, int num_tokens, int k, cudaStream_t stream);
    void moe_compute_permutation(
        const int32_t* expert_indices, int32_t* expert_counts, int32_t* expert_offsets,
        int32_t* permute_indices, int32_t* reverse_perm,
        int num_tokens, int num_experts, int k, cudaStream_t stream);
    void moe_gather_f32(
        const float* hidden, const int32_t* permute_indices, float* gathered,
        int num_tokens, int hidden_size, int k, cudaStream_t stream);
    void moe_gather_bf16(
        const __nv_bfloat16* hidden, const int32_t* permute_indices, __nv_bfloat16* gathered,
        int num_tokens, int hidden_size, int k, cudaStream_t stream);
    void moe_scatter_f32(
        const float* expert_outputs, const float* router_weights, const int32_t* reverse_perm,
        float* output, int num_tokens, int hidden_size, int k, cudaStream_t stream);
    void moe_scatter_bf16(
        const __nv_bfloat16* expert_outputs, const __nv_bfloat16* router_weights,
        const int32_t* reverse_perm, __nv_bfloat16* output,
        int num_tokens, int hidden_size, int k, cudaStream_t stream);
    void expand_expert_offsets(
        const int32_t* expert_offsets, int32_t* row_expert_ids,
        int num_experts, int M_total, cudaStream_t stream);
}
}

void init_ops_bindings(py::module_& m) {
    // ========================================================================
    // Binary Element-wise operations
    // ========================================================================

    // Add
    m.def("add", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::add),
          py::arg("a"), py::arg("b"),
          "Element-wise addition of two GPUArrays");

    m.def("add_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::add),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise addition with output array");

    // Sub
    m.def("sub", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::sub),
          py::arg("a"), py::arg("b"),
          "Element-wise subtraction of two GPUArrays");

    m.def("sub_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::sub),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise subtraction with output array");

    // Mul
    m.def("mul", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::mul),
          py::arg("a"), py::arg("b"),
          "Element-wise multiplication of two GPUArrays");

    m.def("mul_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::mul),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise multiplication with output array");

    // Div
    m.def("div", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::div),
          py::arg("a"), py::arg("b"),
          "Element-wise division of two GPUArrays");

    m.def("div_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::div),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise division with output array");

    // ========================================================================
    // Unary Element-wise operations (float only)
    // ========================================================================

    // Exp
    m.def("exp", py::overload_cast<const GPUArray&>(&ops::exp),
          py::arg("a"),
          "Element-wise exponential (float32/float64 only)");

    m.def("exp_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::exp),
          py::arg("a"), py::arg("out"),
          "Element-wise exponential with output array");

    // Log
    m.def("log", py::overload_cast<const GPUArray&>(&ops::log),
          py::arg("a"),
          "Element-wise natural logarithm (float32/float64 only)");

    m.def("log_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::log),
          py::arg("a"), py::arg("out"),
          "Element-wise natural logarithm with output array");

    // ReLU
    m.def("relu", py::overload_cast<const GPUArray&>(&ops::relu),
          py::arg("a"),
          "Element-wise ReLU: max(0, x) (float32/float64 only)");

    m.def("relu_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::relu),
          py::arg("a"), py::arg("out"),
          "Element-wise ReLU with output array");

    // Sin
    m.def("sin", py::overload_cast<const GPUArray&>(&ops::sin),
          py::arg("a"),
          "Element-wise sine");

    m.def("sin_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::sin),
          py::arg("a"), py::arg("out"),
          "Element-wise sine with output array");

    // Cos
    m.def("cos", py::overload_cast<const GPUArray&>(&ops::cos),
          py::arg("a"),
          "Element-wise cosine");

    m.def("cos_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::cos),
          py::arg("a"), py::arg("out"),
          "Element-wise cosine with output array");

    // Sqrt
    m.def("sqrt", py::overload_cast<const GPUArray&>(&ops::sqrt),
          py::arg("a"),
          "Element-wise square root");

    m.def("sqrt_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::sqrt),
          py::arg("a"), py::arg("out"),
          "Element-wise square root with output array");

    // Rsqrt
    m.def("rsqrt", py::overload_cast<const GPUArray&>(&ops::rsqrt),
          py::arg("a"),
          "Element-wise reciprocal square root: 1/sqrt(x)");

    m.def("rsqrt_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::rsqrt),
          py::arg("a"), py::arg("out"),
          "Element-wise reciprocal square root with output array");

    // Abs
    m.def("abs", py::overload_cast<const GPUArray&>(&ops::abs),
          py::arg("a"),
          "Element-wise absolute value");

    m.def("abs_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::abs),
          py::arg("a"), py::arg("out"),
          "Element-wise absolute value with output array");

    // Neg
    m.def("neg", py::overload_cast<const GPUArray&>(&ops::neg),
          py::arg("a"),
          "Element-wise negation: -x");

    m.def("neg_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::neg),
          py::arg("a"), py::arg("out"),
          "Element-wise negation with output array");

    // Clamp
    m.def("clamp", py::overload_cast<const GPUArray&, float, float>(&ops::clamp),
          py::arg("a"), py::arg("min_val"), py::arg("max_val"),
          "Element-wise clamp: clamp(x, min, max)");

    m.def("clamp_", py::overload_cast<const GPUArray&, GPUArray&, float, float>(&ops::clamp),
          py::arg("a"), py::arg("out"), py::arg("min_val"), py::arg("max_val"),
          "Element-wise clamp with output array");

    // Where (conditional select)
    m.def("where", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&>(&ops::where),
          py::arg("cond"), py::arg("a"), py::arg("b"),
          "Conditional select: where(cond, a, b) = cond ? a : b");

    m.def("where_", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, GPUArray&>(&ops::where),
          py::arg("cond"), py::arg("a"), py::arg("b"), py::arg("out"),
          "Conditional select with output array");

    // ========================================================================
    // Matrix operations
    // ========================================================================

    m.def("matmul", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"),
          "Matrix multiplication of two GPUArrays");

    m.def("matmul_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Matrix multiplication with output array");

    // TF32 variants
    m.def("matmul_tf32", py::overload_cast<const GPUArray&, const GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("use_tf32"),
          "Matrix multiplication with explicit TF32 control");

    m.def("matmul_tf32_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"), py::arg("use_tf32"),
          "Matrix multiplication with explicit TF32 control and output array");

    // ========================================================================
    // Reduction operations
    // ========================================================================

    m.def("sum", &ops::sum,
          py::arg("a"),
          "Sum of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("mean", &ops::mean,
          py::arg("a"),
          "Mean of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("max", &ops::max,
          py::arg("a"),
          "Max of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("min", &ops::min,
          py::arg("a"),
          "Min of all elements, returns scalar GPUArray");

    m.def("argmax", &ops::argmax,
          py::arg("a"),
          "Index of maximum element, returns int64 GPUArray");

    m.def("sum_axis", &ops::sum_axis,
          py::arg("a"), py::arg("axis"),
          "Sum along specified axis (0 or 1) for 2D tensors.\n"
          "axis=0: sum rows -> [N], axis=1: sum columns -> [M]");

    // ========================================================================
    // Neural Network operations
    // ========================================================================

    // Transpose
    m.def("transpose", &ops::transpose,
          py::arg("input"),
          "Matrix transpose: input [rows, cols] -> output [cols, rows]");

    // GELU activation
    m.def("gelu", &ops::gelu,
          py::arg("input"),
          "GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))");

    // Bias add (in-place)
    m.def("bias_add_inplace", &ops::bias_add_inplace,
          py::arg("output"), py::arg("bias"),
          "Add bias to output in-place: output[batch, features] += bias[features]");

    // LayerNorm
    m.def("layernorm", &ops::layernorm,
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f,
          "Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta");

    // Softmax
    m.def("softmax", &ops::softmax,
          py::arg("input"),
          "Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))\n"
          "Applied row-wise: input [batch, features] -> output [batch, features]");

    // RMSNorm
    m.def("rmsnorm", py::overload_cast<const GPUArray&, const GPUArray&, float>(&ops::rmsnorm),
          py::arg("input"), py::arg("gamma"), py::arg("eps") = 1e-5f,
          "RMS normalization: x / sqrt(mean(x^2) + eps) * gamma\n"
          "Simpler than LayerNorm (no mean subtraction, no beta)\n"
          "input: [batch, features], gamma: [features]");

    // RMSNorm with output buffer (for CUDA Graph capture)
    m.def("rmsnorm_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, float>(&ops::rmsnorm),
          py::arg("input"), py::arg("gamma"), py::arg("out"), py::arg("eps") = 1e-5f,
          "RMS normalization with output buffer (for CUDA Graph capture)");

    // ========================================================================
    // Fused Operations (CUTLASS Epilogue Fusion)
    // ========================================================================

    // Linear + BiasGELU (fused kernel)
    m.def("linear_bias_gelu", &ops::linear_bias_gelu,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          "Fused linear + bias + GELU: output = gelu(input @ weight^T + bias)\n"
          "Uses CUTLASS TensorCore epilogue fusion for efficiency.\n"
          "input: [batch, in_features], weight: [out_features, in_features], bias: [out_features]");

    // ========================================================================
    // Additional Neural Network Operations
    // ========================================================================

    // SiLU (Swish) activation
    m.def("silu", py::overload_cast<const GPUArray&>(&ops::silu),
          py::arg("input"),
          "SiLU (Swish) activation: y = x * sigmoid(x)");

    // SiLU with output buffer (for CUDA Graph capture)
    m.def("silu_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::silu),
          py::arg("input"), py::arg("out"),
          "SiLU with output buffer (for CUDA Graph capture)");

    // Sigmoid activation
    m.def("sigmoid", py::overload_cast<const GPUArray&>(&ops::sigmoid),
          py::arg("input"),
          "Sigmoid activation: y = 1 / (1 + exp(-x))");

    m.def("sigmoid_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::sigmoid),
          py::arg("input"), py::arg("out"),
          "Sigmoid with output buffer (for CUDA Graph capture)");

    // Tanh activation
    m.def("tanh", py::overload_cast<const GPUArray&>(&ops::tanh),
          py::arg("input"),
          "Tanh activation");

    m.def("tanh_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::tanh),
          py::arg("input"), py::arg("out"),
          "Tanh with output buffer (for CUDA Graph capture)");

    // RoPE (Rotary Position Embedding) - In-place
    m.def("rope_inplace", &ops::rope_inplace,
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          "Apply RoPE to Q and K tensors in-place.\n"
          "q: [seq_len, n_heads_q, head_dim]\n"
          "k: [seq_len, n_heads_k, head_dim]\n"
          "cos, sin: [seq_len, head_dim]");

    // RoPE with FP32 cos/sin tables (higher precision for bf16/f16)
    m.def("rope_inplace_f32table", &ops::rope_inplace_f32table,
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          "Apply RoPE with FP32 cos/sin tables (higher precision).\n"
          "q: [seq_len, n_heads_q, head_dim] (bf16 or f16)\n"
          "k: [seq_len, n_heads_k, head_dim] (bf16 or f16)\n"
          "cos, sin: [seq_len, head_dim] (f32)");

    // Split fused QKV projection output into separate Q, K, V tensors
    m.def("split_qkv_batch", &ops::split_qkv_batch,
          py::arg("qkv"), py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
          py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"),
          "Split fused QKV projection [seq_len, q_dim+k_dim+v_dim] into Q, K, V.\n"
          "Output buffers must be pre-allocated for CUDA Graph compatibility.");

    // Scaled Dot-Product Attention with Causal Mask
    m.def("sdpa_causal", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, float>(&ops::sdpa_causal),
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale") = 0.0f,
          "Scaled Dot-Product Attention with causal mask.\n"
          "Q: [n_heads, q_len, head_dim]\n"
          "K: [n_heads, kv_len, head_dim]\n"
          "V: [n_heads, kv_len, head_dim]\n"
          "Output: [n_heads, q_len, head_dim]\n"
          "scale: 1/sqrt(head_dim), auto-computed if <= 0");

    // SDPA with output buffer (for CUDA Graph capture)
    m.def("sdpa_causal_", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, GPUArray&, float>(&ops::sdpa_causal),
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("out"), py::arg("scale") = 0.0f,
          "SDPA with output buffer (for CUDA Graph capture)");

    // SDPA with fixed-length KV cache support
    m.def("sdpa_causal_fixed_cache", &ops::sdpa_causal_fixed_cache,
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("out"),
          py::arg("context_len"), py::arg("scale") = 0.0f,
          "SDPA with fixed-length KV cache support.\n"
          "K/V are pre-allocated to max_seq_len, context_len specifies actual valid tokens.");

    m.def("sdpa_causal_fixed_cache_ptr", &ops::sdpa_causal_fixed_cache_ptr,
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("out"),
          py::arg("context_len_buf"), py::arg("max_kv_len"), py::arg("scale") = 0.0f,
          "SDPA with pointer-based context_len for CUDA Graph support.\n"
          "context_len_buf: GPU int32 buffer containing actual context_len.\n"
          "max_kv_len: Max context length (for shared memory allocation at graph capture).");

    // ========================================================================
    // Tensor Manipulation Operations
    // ========================================================================

    // Concat along axis 0
    m.def("concat_axis0", &ops::concat_axis0,
          py::arg("a"), py::arg("b"),
          "Concat two tensors along axis 0.\n"
          "a: [dim0_a, ...], b: [dim0_b, ...]\n"
          "Output: [dim0_a + dim0_b, ...]");

    // Repeat interleave along axis 1 (for GQA)
    m.def("repeat_interleave_axis1", &ops::repeat_interleave_axis1,
          py::arg("input"), py::arg("repeats"),
          "Repeat tensor along axis 1 (interleaved).\n"
          "input: [dim0, dim1, dim2] -> output: [dim0, dim1 * repeats, dim2]");

    // Transpose 3D: [d0, d1, d2] -> [d1, d0, d2]
    m.def("transpose_3d_021", py::overload_cast<const GPUArray&>(&ops::transpose_3d_021),
          py::arg("input"),
          "Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2]");

    // Transpose 3D with output buffer (for CUDA Graph capture)
    m.def("transpose_3d_021_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_3d_021),
          py::arg("input"), py::arg("out"),
          "Transpose 3D tensor with output buffer (for CUDA Graph capture)");

    // Transpose 4D: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
    m.def("transpose_4d_0213", py::overload_cast<const GPUArray&>(&ops::transpose_4d_0213),
          py::arg("input"),
          "Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d2, d1, d3] (swap axes 1 and 2)");

    // Transpose 4D with output buffer (for CUDA Graph capture)
    m.def("transpose_4d_0213_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_4d_0213),
          py::arg("input"), py::arg("out"),
          "Transpose 4D tensor with output buffer (for CUDA Graph capture)");

    // Transpose 3D: [d0, d1, d2] -> [d0, d2, d1] (swap last two axes)
    m.def("transpose_3d_012", py::overload_cast<const GPUArray&>(&ops::transpose_3d_012),
          py::arg("input"),
          "Transpose 3D tensor: [d0, d1, d2] -> [d0, d2, d1] (swap last two axes)");

    // Transpose 3D with output buffer (for CUDA Graph capture)
    m.def("transpose_3d_012_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_3d_012),
          py::arg("input"), py::arg("out"),
          "Transpose 3D tensor with output buffer (for CUDA Graph capture)");

    // Transpose 4D: [d0, d1, d2, d3] -> [d0, d1, d3, d2] (swap last two axes)
    m.def("transpose_4d_0132", py::overload_cast<const GPUArray&>(&ops::transpose_4d_0132),
          py::arg("input"),
          "Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d1, d3, d2] (swap last two axes)");

    // Transpose 4D with output buffer (for CUDA Graph capture)
    m.def("transpose_4d_0132_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_4d_0132),
          py::arg("input"), py::arg("out"),
          "Transpose 4D tensor with output buffer (for CUDA Graph capture)");

    // Reshape with copy
    m.def("reshape_copy", py::overload_cast<const GPUArray&, const std::vector<size_t>&>(&ops::reshape_copy),
          py::arg("input"), py::arg("new_shape"),
          "Reshape tensor with copy (ensures contiguous output).");

    // Reshape with copy into output buffer (for CUDA Graph capture)
    m.def("reshape_copy_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::reshape_copy),
          py::arg("input"), py::arg("out"),
          "Reshape with copy into output buffer (for CUDA Graph capture).");

    // ========================================================================
    // Fixed-Length KV Cache Operations (CUDA Graph Support)
    // ========================================================================

    m.def("kv_cache_update", &ops::kv_cache_update,
          py::arg("new_kv"), py::arg("cache"), py::arg("position"),
          "Update KV cache at a single position (decode step).\n"
          "new_kv: [1, num_kv_heads, head_dim]\n"
          "cache: [max_seq_len, num_kv_heads, head_dim]\n"
          "position: where to write in cache (0-indexed)");

    m.def("kv_cache_prefill", &ops::kv_cache_prefill,
          py::arg("new_kv"), py::arg("cache"), py::arg("start_pos"),
          "Prefill KV cache from sequence.\n"
          "new_kv: [seq_len, num_kv_heads, head_dim]\n"
          "cache: [max_seq_len, num_kv_heads, head_dim]\n"
          "start_pos: where to start writing in cache");

    // GQA-expanded KV cache operations (CUDA Graph optimization)
    m.def("kv_cache_update_gqa", &ops::kv_cache_update_gqa,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("position"),
          "Update GQA-expanded KV cache at single position.\n"
          "new_kv: [1, num_kv_heads, head_dim]\n"
          "cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)\n"
          "num_heads: total number of attention heads\n"
          "position: where to write in cache");

    m.def("kv_cache_prefill_gqa", &ops::kv_cache_prefill_gqa,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("start_pos"),
          "Prefill GQA-expanded KV cache from sequence.\n"
          "new_kv: [seq_len, num_kv_heads, head_dim]\n"
          "cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)\n"
          "num_heads: total number of attention heads\n"
          "start_pos: where to start writing in cache");

    // GPU position pointer variants (for CUDA Graph replay without recapture)
    m.def("kv_cache_update_gqa_ptr", &ops::kv_cache_update_gqa_ptr,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("position_buf"),
          "Update GQA-expanded KV cache reading position from GPU buffer.\n"
          "position_buf: GPUArray[1] int32 containing position value");

    // GPU-only embedding lookup (for CUDA Graph)
    m.def("embedding_lookup", &ops::embedding_lookup,
          py::arg("embed_matrix"), py::arg("out"), py::arg("token_id"),
          "Lookup embedding on GPU without CPU transfer.\n"
          "embed_matrix: [vocab_size, hidden_size]\n"
          "out: [1, hidden_size] pre-allocated buffer\n"
          "token_id: row index to copy");

    m.def("embedding_lookup_ptr", &ops::embedding_lookup_ptr,
          py::arg("embed_matrix"), py::arg("out"), py::arg("token_id_buf"),
          "Lookup embedding reading index from GPU buffer.\n"
          "token_id_buf: GPUArray[1] int32 containing token/position value");

    m.def("embedding_lookup_batch", &ops::embedding_lookup_batch,
          py::arg("embed_matrix"), py::arg("out"), py::arg("token_ids_buf"),
          py::arg("batch_size"),
          "Batch embedding lookup from GPU token ID array.\n"
          "Looks up multiple rows: out[i, :] = embed_matrix[token_ids[i], :]");

    m.def("slice_rows_range_ptr", &ops::slice_rows_range_ptr,
          py::arg("table"), py::arg("out"), py::arg("start_pos_buf"),
          py::arg("count"),
          "Slice consecutive rows from table using GPU-stored start position.\n"
          "Copies `count` rows: out[i, :] = table[start_pos + i, :]");

    // In-place addition (for CUDA Graph)
    m.def("add_inplace", &ops::add_inplace,
          py::arg("a"), py::arg("b"),
          "In-place addition: a += b");

    // In-place multiplication (for CUDA Graph)
    m.def("mul_inplace", &ops::mul_inplace,
          py::arg("a"), py::arg("b"),
          "In-place multiplication: a *= b");

    // GPU-to-GPU copy (for CUDA Graph)
    m.def("copy_to", &ops::copy_to,
          py::arg("src"), py::arg("dst"),
          "Copy src to dst on GPU");

    // ========================================================================
    // Dtype Cast Operations
    // ========================================================================

    m.def("cast_f32_to_bf16", py::overload_cast<const GPUArray&>(&ops::cast_f32_to_bf16),
          py::arg("src"),
          "Cast float32 to bfloat16 on GPU (round to nearest even)");

    m.def("cast_f32_to_bf16_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::cast_f32_to_bf16),
          py::arg("src"), py::arg("dst"),
          "Cast float32 to bfloat16 on GPU (in-place version)");

    m.def("cast_f32_to_f16", &ops::cast_f32_to_f16,
          py::arg("src"),
          "Cast float32 to float16 on GPU");

    m.def("cast_bf16_to_f32", &ops::cast_bf16_to_f32,
          py::arg("src"),
          "Cast bfloat16 to float32 on GPU");

    m.def("cast_f16_to_f32", &ops::cast_f16_to_f32,
          py::arg("src"),
          "Cast float16 to float32 on GPU");

    // ========================================================================
    // Quantization Operations (#85)
    // ========================================================================

    // Dequantize INT8 to FP16/FP32
    m.def("dequantize_int8", &ops::dequantize_int8,
          py::arg("input"), py::arg("scale"), py::arg("output_dtype"),
          "Dequantize INT8 tensor to FP16/FP32.\n"
          "output = input_int8 * scale\n"
          "input: [rows, cols] INT8, scale: [cols], output_dtype: Float16 or Float32");

    // Quantized Linear (INT8 weight x FP16 activation)
    m.def("linear_int8", [](const GPUArray& activation, const GPUArray& weight_int8,
                            const GPUArray& scale, const GPUArray* bias) {
              return ops::linear_int8(activation, weight_int8, scale, bias);
          },
          py::arg("activation"), py::arg("weight_int8"), py::arg("scale"),
          py::arg("bias") = nullptr,
          "Quantized Linear layer with INT8 weights.\n"
          "output = activation @ (weight_int8 * scale).T\n"
          "activation: [M, K] FP16, weight_int8: [N, K] INT8, scale: [N] FP16\n"
          "Dequantization happens on-the-fly (memory efficient).");

    // Quantize to INT8
    m.def("quantize_to_int8", &ops::quantize_to_int8,
          py::arg("input"),
          "Quantize FP16/FP32 tensor to INT8 with per-column scaling.\n"
          "Returns (weight_int8, scale) tuple.\n"
          "weight_int8: [rows, cols] INT8, scale: [cols] same dtype as input");

    // ========================================================================
    // Paged Attention Operations (#87)
    // ========================================================================

    m.def("paged_attention_v1", &ops::paged_attention_v1,
          py::arg("Q"), py::arg("K_cache"), py::arg("V_cache"),
          py::arg("block_tables"), py::arg("context_lens"),
          py::arg("scale") = 0.0f,
          "Paged Attention v1: single-query attention with paged KV cache.\n"
          "Q: [num_seqs, num_heads, head_dim]\n"
          "K_cache, V_cache: [num_blocks, num_kv_heads, block_size, head_dim]\n"
          "block_tables: [num_seqs, max_num_blocks_per_seq] int32\n"
          "context_lens: [num_seqs] int32\n"
          "Output: [num_seqs, num_heads, head_dim]");

    m.def("copy_to_paged_cache", &ops::copy_to_paged_cache,
          py::arg("K_new"), py::arg("V_new"),
          py::arg("K_cache"), py::arg("V_cache"),
          py::arg("slot_mapping"),
          "Copy new KV entries to paged cache (decode phase).\n"
          "K_new, V_new: [num_seqs, num_kv_heads, head_dim]\n"
          "slot_mapping: [num_seqs] int32 - physical slot indices");

    m.def("reshape_and_cache", &ops::reshape_and_cache,
          py::arg("K"), py::arg("V"),
          py::arg("K_cache"), py::arg("V_cache"),
          py::arg("slot_mapping"),
          "Reshape and copy KV from prefill format to paged cache.\n"
          "K, V: [total_tokens, num_kv_heads, head_dim]\n"
          "slot_mapping: [total_tokens] int32");

    m.def("allocate_kv_cache", &ops::allocate_kv_cache,
          py::arg("num_blocks"), py::arg("num_kv_heads"),
          py::arg("block_size"), py::arg("head_dim"),
          "Allocate KV cache blocks.\n"
          "Returns: [num_blocks, num_kv_heads, block_size, head_dim] FP16");

    // ========================================================================
    // Continuous Batching Operations (#86)
    // ========================================================================

    m.def("gather_embeddings", &ops::gather_embeddings,
          py::arg("token_ids"), py::arg("embeddings"), py::arg("total_tokens"),
          "Gather token embeddings for a batch.\n"
          "token_ids: [total_tokens] int32\n"
          "embeddings: [vocab_size, hidden_size] FP16\n"
          "Returns: [total_tokens, hidden_size] FP16");

    m.def("scatter_last_token_logits", &ops::scatter_last_token_logits,
          py::arg("logits"), py::arg("seq_start_positions"),
          py::arg("seq_lens"), py::arg("batch_size"), py::arg("vocab_size"),
          "Scatter last-token logits from batch output.\n"
          "logits: [batch_tokens, vocab_size] FP16\n"
          "Returns: [batch_size, vocab_size] FP16");

    m.def("prepare_position_ids", &ops::prepare_position_ids,
          py::arg("seq_start_positions"), py::arg("seq_context_lens"),
          py::arg("is_prefill"), py::arg("input_lens"),
          py::arg("batch_size"), py::arg("total_tokens"),
          "Prepare position IDs for rotary embeddings.\n"
          "Returns: [total_tokens] int32");

    m.def("argmax_sample", &ops::argmax_sample,
          py::arg("logits"), py::arg("batch_size"), py::arg("vocab_size"),
          "Argmax sampling from logits.\n"
          "logits: [batch_size, vocab_size] FP16\n"
          "Returns: [batch_size] int32 - sampled token IDs");

    m.def("check_eos", &ops::check_eos,
          py::arg("tokens"), py::arg("eos_token_id"),
          "Check for EOS tokens.\n"
          "tokens: [batch_size] int32\n"
          "Returns: [batch_size] int32 - 1 if EOS, 0 otherwise");

    m.def("compute_cumsum", &ops::compute_cumsum,
          py::arg("input"),
          "Compute exclusive prefix sum.\n"
          "input: [n] int32\n"
          "Returns: [n] int32");

    m.def("prepare_batch_inputs", &ops::prepare_batch_inputs,
          py::arg("token_lists"),
          "Prepare batch inputs from Python lists.\n"
          "token_lists: List of token ID lists\n"
          "Returns: (token_ids GPUArray, total_tokens count)");

    // ========================================================================
    // GPU Sampling Operations (#v0.2.10)
    // ========================================================================

    m.def("sample_greedy", &ops::sample_greedy,
          py::arg("logits"),
          "Greedy sampling (argmax) from logits.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "Returns: sampled token ID (int)");

    m.def("sample_multinomial", &ops::sample_multinomial,
          py::arg("logits"), py::arg("temperature"),
          "Multinomial sampling with temperature.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "temperature: > 0 (lower = more deterministic)\n"
          "Returns: sampled token ID (int)");

    m.def("sample_topk", &ops::sample_topk,
          py::arg("logits"), py::arg("top_k"), py::arg("temperature"),
          "Top-K sampling.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "top_k: number of top tokens to consider\n"
          "temperature: > 0\n"
          "Returns: sampled token ID (int)");

    m.def("sample_topk_to_buf", &ops::sample_topk_to_buf,
          py::arg("logits"), py::arg("result_buf"), py::arg("top_k"),
          py::arg("temperature"), py::arg("random_val"),
          "Top-K sampling (CUDA Graph compatible).\n"
          "Writes result to pre-allocated buffer, no sync/D2H.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "result_buf: pre-allocated int32 buffer [1]\n"
          "top_k: number of top tokens to consider\n"
          "temperature: > 0\n"
          "random_val: pre-generated random value [0, 1)");

    m.def("sample_topk_to_buf_ptr", &ops::sample_topk_to_buf_ptr,
          py::arg("logits"), py::arg("result_buf"), py::arg("random_val_buf"),
          py::arg("top_k"), py::arg("temperature"),
          "Top-K sampling with pointer (CUDA Graph replay compatible).\n"
          "random_val is read from GPU buffer, allowing update before replay.\n"
          "logits: [vocab_size] or [1, vocab_size] (float16 only)\n"
          "result_buf: pre-allocated int32 buffer [1]\n"
          "random_val_buf: pre-allocated float32 buffer [1]\n"
          "top_k: number of top tokens to consider\n"
          "temperature: > 0");

    m.def("sample_topp", &ops::sample_topp,
          py::arg("logits"), py::arg("top_p"), py::arg("temperature"),
          "Top-P (nucleus) sampling.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "top_p: cumulative probability threshold (0 < p <= 1)\n"
          "temperature: > 0\n"
          "Returns: sampled token ID (int)");

    m.def("sample_token_gpu", &ops::sample_token_gpu,
          py::arg("logits"),
          py::arg("temperature") = 1.0f,
          py::arg("top_k") = 0,
          py::arg("top_p") = 1.0f,
          "Unified GPU sampling API.\n"
          "Automatically selects sampling method:\n"
          "- temperature=0: greedy (argmax)\n"
          "- top_k > 0: top-k sampling\n"
          "- top_p < 1: top-p sampling\n"
          "- otherwise: multinomial with temperature\n"
          "Returns: sampled token ID (int)");

    m.def("set_sampling_seed", &ops::set_sampling_seed,
          py::arg("seed"),
          "Set random seed for reproducible GPU sampling.");

    // ========================================================================
    // Audio Processing Operations (#96)
    // ========================================================================

    m.def("audio_pcm_to_float32", &ops::audio::pcm_to_float32,
          py::arg("input"),
          "Convert int16 PCM samples to float32.\n"
          "Input: GPUArray of int16 samples\n"
          "Returns: GPUArray of float32 samples normalized to [-1.0, 1.0]");

    m.def("audio_stereo_to_mono", &ops::audio::stereo_to_mono,
          py::arg("input"),
          "Convert stereo audio to mono by averaging channels.\n"
          "Input: GPUArray of interleaved stereo samples [L,R,L,R,...]\n"
          "Returns: GPUArray of mono samples");

    m.def("audio_normalize_peak", &ops::audio::normalize_peak,
          py::arg("input"),
          "Peak normalize audio to [-1.0, 1.0] range (in-place).\n"
          "Input: GPUArray of float32 samples (modified in-place)");

    m.def("audio_normalize_rms", &ops::audio::normalize_rms,
          py::arg("input"), py::arg("target_db") = -20.0f,
          "RMS normalize audio to target dB level (in-place).\n"
          "Input: GPUArray of float32 samples (modified in-place)\n"
          "target_db: Target RMS level in dB (default -20.0)");

    m.def("audio_resample", &ops::audio::resample,
          py::arg("input"), py::arg("src_rate"), py::arg("dst_rate"),
          "Resample audio from source to target sample rate.\n"
          "Currently supports 48kHz -> 16kHz (3:1 decimation).\n"
          "Input: GPUArray of float32 samples\n"
          "src_rate: Source sample rate (e.g., 48000)\n"
          "dst_rate: Target sample rate (e.g., 16000)\n"
          "Returns: Resampled GPUArray");

    // ========================================================================
    // Audio Streaming Operations (#97)
    // ========================================================================

    m.def("audio_ring_buffer_write", &ops::audio::ring_buffer_write,
          py::arg("input"), py::arg("ring_buffer"), py::arg("write_pos"),
          "Write samples to a ring buffer with wrap-around.\n"
          "input: GPUArray of float32 samples to write\n"
          "ring_buffer: GPUArray ring buffer (modified in-place)\n"
          "write_pos: Current write position in ring buffer");

    m.def("audio_ring_buffer_read", &ops::audio::ring_buffer_read,
          py::arg("ring_buffer"), py::arg("read_pos"), py::arg("num_samples"),
          "Read samples from a ring buffer (linearized).\n"
          "ring_buffer: GPUArray ring buffer\n"
          "read_pos: Read position in ring buffer\n"
          "num_samples: Number of samples to read\n"
          "Returns: Linearized GPUArray");

    m.def("audio_apply_hann_window", &ops::audio::apply_hann_window,
          py::arg("data"),
          "Apply Hann window to audio data (in-place).\n"
          "data: GPUArray of float32 samples (modified in-place)");

    m.def("audio_overlap_add", &ops::audio::overlap_add,
          py::arg("input"), py::arg("output"), py::arg("output_offset"),
          "Overlap-add: add windowed chunk to output buffer.\n"
          "input: Windowed input chunk\n"
          "output: Output buffer (accumulated, modified in-place)\n"
          "output_offset: Offset in output buffer");

    // ========================================================================
    // Voice Activity Detection (VAD)
    // ========================================================================

    m.def("vad_compute_energy", &ops::audio::vad_compute_energy,
          py::arg("audio"), py::arg("frame_size"), py::arg("hop_size"),
          "Compute frame-level RMS energy for VAD.\n"
          "audio: Input audio samples (float32)\n"
          "frame_size: Frame size in samples\n"
          "hop_size: Hop size in samples\n"
          "Returns: GPUArray of frame energies");

    m.def("vad_compute_zcr", &ops::audio::vad_compute_zcr,
          py::arg("audio"), py::arg("frame_size"), py::arg("hop_size"),
          "Compute frame-level zero-crossing rate for VAD.\n"
          "audio: Input audio samples (float32)\n"
          "frame_size: Frame size in samples\n"
          "hop_size: Hop size in samples\n"
          "Returns: GPUArray of frame ZCR values [0, 1]");

    m.def("vad_decide", &ops::audio::vad_decide,
          py::arg("frame_energy"), py::arg("frame_zcr"),
          py::arg("energy_threshold"), py::arg("zcr_low"), py::arg("zcr_high"),
          "Apply threshold-based VAD decision.\n"
          "frame_energy: Frame energy values (float32)\n"
          "frame_zcr: Frame ZCR values (float32)\n"
          "energy_threshold: Energy threshold for speech detection\n"
          "zcr_low: Lower ZCR bound for voiced speech\n"
          "zcr_high: Upper ZCR bound\n"
          "Returns: GPUArray of int32 VAD flags (0=silence, 1=speech)");

    m.def("vad_apply_hangover", &ops::audio::vad_apply_hangover,
          py::arg("vad_input"), py::arg("hangover_frames"),
          "Apply hangover smoothing to VAD output.\n"
          "Extends speech regions by hangover_frames after speech ends.\n"
          "vad_input: Input VAD flags (int32)\n"
          "hangover_frames: Number of frames to extend\n"
          "Returns: Smoothed VAD flags (int32)");

    m.def("vad_compute_noise_floor", &ops::audio::vad_compute_noise_floor,
          py::arg("frame_energy"),
          "Compute noise floor (minimum energy) for adaptive thresholding.\n"
          "frame_energy: Frame energy values (float32)\n"
          "Returns: Minimum energy value (float)");

    // ========================================================================
    // Audio Preprocessing Operations
    // ========================================================================

    m.def("audio_preemphasis", &ops::audio::preemphasis,
          py::arg("input"), py::arg("alpha") = 0.97f,
          "Apply pre-emphasis filter (in-place).\n"
          "y[n] = x[n] - alpha * x[n-1]\n"
          "input: GPUArray of float32 samples (modified in-place)\n"
          "alpha: Pre-emphasis coefficient (default 0.97)");

    m.def("audio_deemphasis", &ops::audio::deemphasis,
          py::arg("input"), py::arg("alpha") = 0.97f,
          "Apply de-emphasis filter (in-place).\n"
          "y[n] = x[n] + alpha * y[n-1]\n"
          "input: GPUArray of float32 samples (modified in-place)\n"
          "alpha: De-emphasis coefficient (default 0.97)");

    m.def("audio_remove_dc", &ops::audio::remove_dc,
          py::arg("input"),
          "Remove DC offset from audio signal (in-place).\n"
          "Subtracts the mean value from all samples.\n"
          "input: GPUArray of float32 samples (modified in-place)");

    m.def("audio_highpass_filter", &ops::audio::highpass_filter,
          py::arg("input"), py::arg("cutoff_hz") = 20.0f, py::arg("sample_rate") = 16000,
          "Apply high-pass filter for DC removal (in-place).\n"
          "Uses single-pole IIR filter.\n"
          "input: GPUArray of float32 samples (modified in-place)\n"
          "cutoff_hz: Cutoff frequency in Hz (default 20.0)\n"
          "sample_rate: Sample rate in Hz (default 16000)");

    m.def("audio_noise_gate", &ops::audio::noise_gate,
          py::arg("input"), py::arg("threshold") = 0.01f,
          "Apply simple noise gate (in-place).\n"
          "Zeros samples with absolute value below threshold.\n"
          "input: GPUArray of float32 samples (modified in-place)\n"
          "threshold: Amplitude threshold (default 0.01)");

    m.def("audio_spectral_gate", &ops::audio::spectral_gate,
          py::arg("input"), py::arg("threshold") = 0.01f,
          py::arg("attack_samples") = 64, py::arg("release_samples") = 256,
          "Apply spectral gate for noise reduction (in-place).\n"
          "Attenuates samples in frames with energy below threshold.\n"
          "input: GPUArray of float32 samples (modified in-place)\n"
          "threshold: Energy threshold (linear scale, default 0.01)\n"
          "attack_samples: Frame size for energy computation (default 64)\n"
          "release_samples: Smoothing release (reserved, default 256)");

    m.def("audio_compute_short_term_energy", &ops::audio::compute_short_term_energy,
          py::arg("input"), py::arg("frame_size"),
          "Compute short-term energy for adaptive noise gating.\n"
          "input: GPUArray of float32 audio samples\n"
          "frame_size: Frame size in samples\n"
          "Returns: GPUArray of frame energies");

    // ========================================================================
    // Spectral Processing Operations
    // ========================================================================

    m.def("audio_stft", &ops::audio::stft,
          py::arg("input"), py::arg("n_fft") = 400, py::arg("hop_length") = 160,
          py::arg("win_length") = -1, py::arg("center") = true,
          "Compute Short-Time Fourier Transform (STFT).\n"
          "input: GPUArray of float32 audio samples\n"
          "n_fft: FFT size (must be power of 2, default 400 for Whisper)\n"
          "hop_length: Hop size (default 160 for Whisper)\n"
          "win_length: Window length (default n_fft)\n"
          "center: Whether to pad input (default true)\n"
          "Returns: Complex STFT output [n_frames, n_fft/2+1, 2] (real, imag)");

    m.def("audio_power_spectrum", &ops::audio::power_spectrum,
          py::arg("stft_output"),
          "Compute power spectrogram from STFT output.\n"
          "power = real^2 + imag^2\n"
          "stft_output: STFT output [n_frames, n_freq, 2]\n"
          "Returns: Power spectrogram [n_frames, n_freq]");

    m.def("audio_magnitude_spectrum", &ops::audio::magnitude_spectrum,
          py::arg("stft_output"),
          "Compute magnitude spectrogram from STFT output.\n"
          "magnitude = sqrt(real^2 + imag^2)\n"
          "stft_output: STFT output [n_frames, n_freq, 2]\n"
          "Returns: Magnitude spectrogram [n_frames, n_freq]");

    m.def("audio_create_mel_filterbank", &ops::audio::create_mel_filterbank,
          py::arg("n_mels"), py::arg("n_fft"), py::arg("sample_rate"),
          py::arg("f_min") = 0.0f, py::arg("f_max") = -1.0f,
          "Create Mel filterbank matrix.\n"
          "n_mels: Number of mel bands (default 80 for Whisper)\n"
          "n_fft: FFT size\n"
          "sample_rate: Sample rate in Hz\n"
          "f_min: Minimum frequency (default 0)\n"
          "f_max: Maximum frequency (default sample_rate/2)\n"
          "Returns: Mel filterbank matrix [n_mels, n_fft/2+1]");

    m.def("audio_apply_mel_filterbank", &ops::audio::apply_mel_filterbank,
          py::arg("spectrogram"), py::arg("mel_filterbank"),
          "Apply Mel filterbank to power/magnitude spectrogram.\n"
          "spectrogram: Input spectrogram [n_frames, n_fft/2+1]\n"
          "mel_filterbank: Mel filterbank [n_mels, n_fft/2+1]\n"
          "Returns: Mel spectrogram [n_frames, n_mels]");

    m.def("audio_log_mel_spectrogram", &ops::audio::log_mel_spectrogram,
          py::arg("mel_spectrogram"), py::arg("eps") = 1e-10f,
          "Compute log-mel spectrogram.\n"
          "log_mel = log(mel + eps)\n"
          "mel_spectrogram: Mel spectrogram [n_frames, n_mels]\n"
          "eps: Small constant for numerical stability (default 1e-10)\n"
          "Returns: Log-mel spectrogram [n_frames, n_mels]");

    m.def("audio_to_decibels", &ops::audio::to_decibels,
          py::arg("input"), py::arg("eps") = 1e-10f,
          "Convert to decibels.\n"
          "dB = 10 * log10(x + eps)\n"
          "input: Input array\n"
          "eps: Small constant for numerical stability (default 1e-10)\n"
          "Returns: dB values");

    m.def("audio_mfcc", &ops::audio::mfcc,
          py::arg("log_mel"), py::arg("n_mfcc") = 13,
          "Compute MFCC from log-mel spectrogram using DCT-II.\n"
          "log_mel: Log-mel spectrogram [n_frames, n_mels]\n"
          "n_mfcc: Number of MFCC coefficients (default 13)\n"
          "Returns: MFCC [n_frames, n_mfcc]");

    m.def("audio_delta_features", &ops::audio::delta_features,
          py::arg("features"), py::arg("order") = 1, py::arg("width") = 2,
          "Compute delta (differential) features.\n"
          "features: Input features [n_frames, n_features]\n"
          "order: Delta order (1 for delta, 2 for delta-delta)\n"
          "width: Window width for computation (default 2)\n"
          "Returns: Delta features [n_frames, n_features]");

    m.def("audio_whisper_mel_spectrogram", &ops::audio::whisper_mel_spectrogram,
          py::arg("input"), py::arg("n_fft") = 400, py::arg("hop_length") = 160,
          py::arg("n_mels") = 80,
          "Compute Whisper-compatible log-mel spectrogram in one call.\n"
          "Combines: STFT -> power -> mel filterbank -> log\n"
          "input: Input audio (float32, 16kHz expected)\n"
          "n_fft: FFT size (default 400)\n"
          "hop_length: Hop size (default 160)\n"
          "n_mels: Number of mel bands (default 80)\n"
          "Returns: Log-mel spectrogram [n_frames, n_mels]");

    // ========================================================================
    // Inverse STFT
    // ========================================================================

    m.def("audio_istft", &ops::audio::istft,
          py::arg("stft_output"), py::arg("hop_length") = 160,
          py::arg("win_length") = -1, py::arg("center") = true,
          py::arg("length") = -1,
          "Compute Inverse Short-Time Fourier Transform (ISTFT).\n"
          "stft_output: STFT output [n_frames, n_fft/2+1, 2] (real, imag)\n"
          "hop_length: Hop size (default 160)\n"
          "win_length: Window length (default n_fft)\n"
          "center: Whether input was padded (default true)\n"
          "length: Expected output length (optional, -1 for auto)\n"
          "Returns: Reconstructed audio signal");

    // ========================================================================
    // Griffin-Lim Algorithm
    // ========================================================================

    m.def("audio_griffin_lim", &ops::audio::griffin_lim,
          py::arg("magnitude"), py::arg("n_iter") = 32,
          py::arg("hop_length") = 160, py::arg("win_length") = -1,
          "Griffin-Lim phase reconstruction algorithm.\n"
          "Reconstructs audio from magnitude spectrogram.\n"
          "magnitude: Magnitude spectrogram [n_frames, n_fft/2+1]\n"
          "n_iter: Number of iterations (default 32)\n"
          "hop_length: Hop size (default 160)\n"
          "win_length: Window length (default n_fft * 2 - 2)\n"
          "Returns: Reconstructed audio signal");

    // ========================================================================
    // Pitch Detection
    // ========================================================================

    m.def("audio_autocorrelation", &ops::audio::autocorrelation,
          py::arg("input"), py::arg("max_lag"),
          "Compute autocorrelation of signal.\n"
          "input: Input audio samples\n"
          "max_lag: Maximum lag to compute\n"
          "Returns: Autocorrelation values [max_lag]");

    m.def("audio_detect_pitch_yin", &ops::audio::detect_pitch_yin,
          py::arg("input"), py::arg("sample_rate"),
          py::arg("f_min") = 50.0f, py::arg("f_max") = 2000.0f,
          py::arg("threshold") = 0.1f,
          "Detect pitch using YIN algorithm.\n"
          "input: Input audio samples (single frame)\n"
          "sample_rate: Sample rate in Hz\n"
          "f_min: Minimum frequency (default 50 Hz)\n"
          "f_max: Maximum frequency (default 2000 Hz)\n"
          "threshold: YIN threshold (default 0.1)\n"
          "Returns: Detected pitch in Hz (0 if unvoiced)");

    m.def("audio_detect_pitch_yin_frames", &ops::audio::detect_pitch_yin_frames,
          py::arg("input"), py::arg("sample_rate"),
          py::arg("frame_size"), py::arg("hop_size"),
          py::arg("f_min") = 50.0f, py::arg("f_max") = 2000.0f,
          py::arg("threshold") = 0.1f,
          "Detect pitch for multiple frames using YIN algorithm.\n"
          "input: Input audio samples\n"
          "sample_rate: Sample rate in Hz\n"
          "frame_size: Frame size in samples\n"
          "hop_size: Hop size in samples\n"
          "f_min: Minimum frequency (default 50 Hz)\n"
          "f_max: Maximum frequency (default 2000 Hz)\n"
          "threshold: YIN threshold (default 0.1)\n"
          "Returns: Detected pitches [n_frames] in Hz (0 if unvoiced)");

    // ========================================================================
    // Spectral Features
    // ========================================================================

    m.def("audio_spectral_centroid", &ops::audio::spectral_centroid,
          py::arg("spectrum"), py::arg("sample_rate"),
          "Compute spectral centroid (center of mass of spectrum).\n"
          "spectrum: Magnitude/power spectrogram [n_frames, n_freq]\n"
          "sample_rate: Sample rate in Hz\n"
          "Returns: Spectral centroid per frame [n_frames] in Hz");

    m.def("audio_spectral_bandwidth", &ops::audio::spectral_bandwidth,
          py::arg("spectrum"), py::arg("centroids"),
          py::arg("sample_rate"), py::arg("p") = 2,
          "Compute spectral bandwidth.\n"
          "spectrum: Magnitude/power spectrogram [n_frames, n_freq]\n"
          "centroids: Pre-computed centroids [n_frames]\n"
          "sample_rate: Sample rate in Hz\n"
          "p: Order of the bandwidth norm (default 2)\n"
          "Returns: Spectral bandwidth per frame [n_frames] in Hz");

    m.def("audio_spectral_rolloff", &ops::audio::spectral_rolloff,
          py::arg("spectrum"), py::arg("sample_rate"),
          py::arg("roll_percent") = 0.85f,
          "Compute spectral rolloff point.\n"
          "spectrum: Magnitude/power spectrogram [n_frames, n_freq]\n"
          "sample_rate: Sample rate in Hz\n"
          "roll_percent: Rolloff percentage (default 0.85 = 85%)\n"
          "Returns: Rolloff frequency per frame [n_frames] in Hz");

    m.def("audio_spectral_flatness", &ops::audio::spectral_flatness,
          py::arg("spectrum"),
          "Compute spectral flatness (Wiener entropy).\n"
          "spectrum: Magnitude/power spectrogram [n_frames, n_freq]\n"
          "Returns: Flatness per frame [n_frames] in [0, 1]");

    m.def("audio_spectral_contrast", &ops::audio::spectral_contrast,
          py::arg("spectrum"), py::arg("n_bands") = 6,
          py::arg("alpha") = 0.02f,
          "Compute spectral contrast.\n"
          "spectrum: Magnitude/power spectrogram [n_frames, n_freq]\n"
          "n_bands: Number of frequency bands (default 6)\n"
          "alpha: Percentile for peak/valley (default 0.02 = 2%)\n"
          "Returns: Spectral contrast [n_frames, n_bands]");

    m.def("audio_zero_crossing_rate", &ops::audio::zero_crossing_rate,
          py::arg("input"), py::arg("frame_size"), py::arg("hop_size"),
          "Compute zero-crossing rate.\n"
          "input: Input audio samples\n"
          "frame_size: Frame size in samples\n"
          "hop_size: Hop size in samples\n"
          "Returns: ZCR per frame [n_frames] in [0, 1]");

    // ========================================================================
    // CQT (Constant-Q Transform)
    // ========================================================================

    m.def("audio_cqt", &ops::audio::cqt,
          py::arg("input"), py::arg("sample_rate"),
          py::arg("hop_length") = 512, py::arg("f_min") = 32.7f,
          py::arg("n_bins") = 84, py::arg("bins_per_octave") = 12,
          "Compute Constant-Q Transform.\n"
          "input: Input audio samples\n"
          "sample_rate: Sample rate in Hz\n"
          "hop_length: Hop size (default 512)\n"
          "f_min: Minimum frequency (default 32.7 Hz, C1)\n"
          "n_bins: Number of CQT bins (default 84, 7 octaves)\n"
          "bins_per_octave: Bins per octave (default 12)\n"
          "Returns: Complex CQT output [n_frames, n_bins, 2]");

    m.def("audio_cqt_magnitude", &ops::audio::cqt_magnitude,
          py::arg("cqt_output"),
          "Compute CQT magnitude spectrogram.\n"
          "cqt_output: CQT output [n_frames, n_bins, 2]\n"
          "Returns: Magnitude spectrogram [n_frames, n_bins]");

    // ========================================================================
    // Chromagram
    // ========================================================================

    m.def("audio_chroma_stft", &ops::audio::chroma_stft,
          py::arg("spectrum"), py::arg("sample_rate"),
          py::arg("n_chroma") = 12, py::arg("tuning") = 0.0f,
          "Compute chromagram from STFT.\n"
          "spectrum: Power/magnitude spectrogram [n_frames, n_freq]\n"
          "sample_rate: Sample rate in Hz\n"
          "n_chroma: Number of chroma bins (default 12)\n"
          "tuning: Tuning deviation from A440 in cents (default 0)\n"
          "Returns: Chromagram [n_frames, n_chroma]");

    m.def("audio_chroma_cqt", &ops::audio::chroma_cqt,
          py::arg("cqt_mag"), py::arg("bins_per_octave") = 12,
          "Compute chromagram from CQT.\n"
          "cqt_mag: CQT magnitude [n_frames, n_bins]\n"
          "bins_per_octave: Bins per octave (must match CQT, default 12)\n"
          "Returns: Chromagram [n_frames, 12]");

    // ========================================================================
    // HPSS (Harmonic-Percussive Source Separation)
    // ========================================================================

    m.def("audio_hpss", [](const GPUArray& stft_magnitude, int kernel_size,
                           float power, float margin) {
              auto [h, p] = ops::audio::hpss(stft_magnitude, kernel_size, power, margin);
              return py::make_tuple(std::move(h), std::move(p));
          },
          py::arg("stft_magnitude"), py::arg("kernel_size") = 31,
          py::arg("power") = 2.0f, py::arg("margin") = 1.0f,
          "Harmonic-percussive source separation.\n"
          "stft_magnitude: STFT magnitude [n_frames, n_freq]\n"
          "kernel_size: Median filter kernel size (default 31)\n"
          "power: Mask power for softness (default 2.0)\n"
          "margin: Margin for separation (default 1.0)\n"
          "Returns: Tuple of (harmonic_magnitude, percussive_magnitude)");

    m.def("audio_harmonic", &ops::audio::harmonic,
          py::arg("stft_magnitude"), py::arg("kernel_size") = 31,
          py::arg("power") = 2.0f, py::arg("margin") = 1.0f,
          "Get harmonic component from HPSS.\n"
          "Returns: Harmonic magnitude [n_frames, n_freq]");

    m.def("audio_percussive", &ops::audio::percussive,
          py::arg("stft_magnitude"), py::arg("kernel_size") = 31,
          py::arg("power") = 2.0f, py::arg("margin") = 1.0f,
          "Get percussive component from HPSS.\n"
          "Returns: Percussive magnitude [n_frames, n_freq]");

    // ========================================================================
    // Time Stretch / Pitch Shift
    // ========================================================================

    m.def("audio_time_stretch", &ops::audio::time_stretch,
          py::arg("input"), py::arg("rate"),
          py::arg("n_fft") = 2048, py::arg("hop_length") = -1,
          "Time-stretch audio using phase vocoder.\n"
          "input: Input audio samples\n"
          "rate: Time stretch rate (>1 = slower, <1 = faster)\n"
          "n_fft: FFT size (default 2048)\n"
          "hop_length: Hop size (default n_fft/4)\n"
          "Returns: Time-stretched audio");

    m.def("audio_pitch_shift", &ops::audio::pitch_shift,
          py::arg("input"), py::arg("sample_rate"), py::arg("n_steps"),
          py::arg("n_fft") = 2048, py::arg("hop_length") = -1,
          "Pitch-shift audio.\n"
          "input: Input audio samples\n"
          "sample_rate: Sample rate in Hz\n"
          "n_steps: Number of semitones to shift\n"
          "n_fft: FFT size (default 2048)\n"
          "hop_length: Hop size (default n_fft/4)\n"
          "Returns: Pitch-shifted audio");

    // ========================================================================
    // cuBLASLt debug functions
    // ========================================================================

    m.def("cublaslt_is_available", &cublaslt::is_available,
          "Check if cuBLASLt is dynamically loaded and available.");

    m.def("cublaslt_get_library_path", &cublaslt::get_library_path,
          "Get the path to the loaded cuBLASLt library.");

    m.def("cublaslt_get_version", []() {
        auto [major, minor, patch] = cublaslt::get_version();
        return py::make_tuple(major, minor, patch);
    }, "Get cuBLASLt version as (major, minor, patch) tuple.");

    m.def("cublaslt_test_gemm", [](const GPUArray& a, const GPUArray& b) {
        // Test GEMM and return status code
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        GPUArray c({M, N}, a.dtype());

        cudaError_t err = cublaslt::gemm_fp16(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(c.data()),
            M, N, K, nullptr);

        return static_cast<int>(err);
    }, py::arg("a"), py::arg("b"),
       "Test cuBLASLt FP16 GEMM and return error code (0 = success).");

    m.def("cublaslt_get_last_error", &cublaslt::get_last_cublaslt_error,
          "Get last cuBLASLt status code for debugging.");

    m.def("cublaslt_get_last_step", &cublaslt::get_last_cublaslt_step,
          "Get which step failed (1=handle, 2=desc, 3-5=layout, 6=matmul).");

    m.def("cublaslt_get_handle", []() {
        auto handle = cublaslt::get_handle();
        return reinterpret_cast<uintptr_t>(handle);
    }, "Get cuBLASLt handle address for debugging (0 if not available).");

    // ========================================================================
    // Strided Batched GEMM (for batched matmul in attention)
    // ========================================================================

    m.def("gemm_strided_batched_fp32", &ops::batched_matmul_fp32,
       py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("batch_count"),
       py::arg("strideA"), py::arg("strideB"), py::arg("strideC"),
       "Strided batched GEMM: C[b] = A[b] @ B[b] for b in [0, batch_count)");

    // ========================================================================
    // FP8 GEMM for SM90 (Hopper) - per-tensor scaling
    // ========================================================================

    m.def("fp8_sm90_available", []() {
        return pygpukit_fp8_sm90_available();
    }, "Check if FP8 GEMM is available on SM90 (Hopper)");

    m.def("gemm_fp8_sm90", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm90: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm90: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm90: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm90: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_sm90(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_sm90 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "FP8 GEMM for SM90 (Hopper): D = A @ B (with FP8 quantization internally)");

    // ========================================================================
    // FP8 GEMM for SM100 (Blackwell datacenter) - blockwise scaling
    // Potential fallback for SM120 (same Blackwell architecture)
    // ========================================================================

    m.def("fp8_sm100_available", []() {
        return pygpukit_fp8_sm100_available();
    }, "Check if FP8 GEMM is available on SM100 (Blackwell datacenter)");

    m.def("gemm_fp8_sm100", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm100: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm100: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm100: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm100: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_sm100(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_sm100 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "FP8 GEMM for SM100 (Blackwell datacenter): D = A @ B (with FP8 quantization internally)");

    // ========================================================================
    // FP8 GEMM for SM120 (Blackwell GeForce) - blockwise scaling
    // NOTE: Currently disabled due to CUTLASS bug #2902
    // ========================================================================

    m.def("fp8_sm120_available", []() {
        return pygpukit_fp8_sm120_available();
    }, "Check if FP8 GEMM is available on SM120 (currently disabled due to CUTLASS bug)");

    m.def("gemm_fp8_sm120", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm120: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm120: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_sm120(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "FP8 GEMM for SM120: D = A @ B (with FP8 quantization internally)");

    // ========================================================================
    // Pure FP8 I/O GEMM for SM120 (FP8 models)
    // ========================================================================

    m.def("fp8_fp8_sm120_available", []() {
        return pygpukit_fp8_fp8_sm120_available();
    }, "Check if Pure FP8 I/O GEMM is available on SM120");

    m.def("gemm_fp8_fp8_sm120", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        // FP8 is stored as UInt8 in GPUArray
        if (A.dtype() != DataType::UInt8 || B.dtype() != DataType::UInt8 || D.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: all inputs must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        // B is expected to be in ColumnMajor format [K, N] stored as [N, K] transposed
        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_fp8_sm120(
            static_cast<const uint8_t*>(A.data()),
            static_cast<const uint8_t*>(B.data()),
            static_cast<uint8_t*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_fp8_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "Pure FP8 I/O GEMM for SM120: D = A @ B (FP8 E4M3 input/output)");

    // Tile variant helper
    auto bind_fp8_tile = [&m](const char* name, auto func, const char* doc) {
        m.def(name, [func, name](const GPUArray& A, const GPUArray& B, GPUArray& D) {
            if (A.dtype() != DataType::UInt8 || B.dtype() != DataType::UInt8 || D.dtype() != DataType::UInt8) {
                throw std::runtime_error("FP8 GEMM: all inputs must be uint8");
            }
            int M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
            if (B.shape()[0] != static_cast<size_t>(K)) throw std::runtime_error("Shape mismatch");
            cudaError_t err = func(
                static_cast<const uint8_t*>(A.data()),
                static_cast<const uint8_t*>(B.data()),
                static_cast<uint8_t*>(D.data()),
                M, N, K, 1.0f, 0.0f, nullptr);
            if (err != cudaSuccess) throw std::runtime_error(std::string(name) + " failed");
        }, py::arg("A"), py::arg("B"), py::arg("D"), doc);
    };
    bind_fp8_tile("gemm_fp8_fp8_sm120_v2", pygpukit_gemm_fp8_fp8_sm120_v2, "FP8 GEMM 128x256x64");
    bind_fp8_tile("gemm_fp8_fp8_sm120_v3", pygpukit_gemm_fp8_fp8_sm120_v3, "FP8 GEMM 256x128x64");
    bind_fp8_tile("gemm_fp8_fp8_sm120_v4", pygpukit_gemm_fp8_fp8_sm120_v4, "FP8 GEMM 128x128x64");

    // Blockwise scaled FP8 GEMM
    m.def("gemm_fp8_fp8_blockwise_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D,
        const GPUArray& scale_A, const GPUArray& scale_B
    ) {
        // FP8 is stored as UInt8 in GPUArray
        if (A.dtype() != DataType::UInt8 || B.dtype() != DataType::UInt8 || D.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: A, B, D must be uint8 (FP8 E4M3)");
        }
        if (scale_A.dtype() != DataType::Float32 || scale_B.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: scale_A, scale_B must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: A, B, D must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_fp8_blockwise_sm120(
            static_cast<const uint8_t*>(A.data()),
            static_cast<const uint8_t*>(B.data()),
            static_cast<uint8_t*>(D.data()),
            static_cast<const float*>(scale_A.data()),
            static_cast<const float*>(scale_B.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("scale_A"), py::arg("scale_B"),
       "Blockwise scaled FP8 I/O GEMM for SM120: D = (A * scale_A) @ (B * scale_B)");

    // Get scale factor sizes for FP8 blockwise GEMM
    m.def("fp8_fp8_get_scale_sizes", [](int M, int N, int K) {
        size_t sfa_size, sfb_size;
        pygpukit_fp8_fp8_get_scale_sizes(M, N, K, &sfa_size, &sfb_size);
        return py::make_tuple(sfa_size, sfb_size);
    }, py::arg("M"), py::arg("N"), py::arg("K"),
       "Get scale factor sizes for FP8 blockwise GEMM (returns (sfa_size, sfb_size))");

    // ========================================================================
    // NVF4 (4-bit) GEMM for SM120 with BF16 I/O
    // ========================================================================

    m.def("nvf4_bf16_sm120_available", []() {
        return pygpukit_nvf4_bf16_sm120_available();
    }, "Check if NVF4 BF16 GEMM is available on SM120");

    m.def("gemm_nvf4_bf16_sm120", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || B.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: all inputs must be bfloat16");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_nvf4_bf16_sm120(
            static_cast<const __nv_bfloat16*>(A.data()),
            static_cast<const __nv_bfloat16*>(B.data()),
            static_cast<__nv_bfloat16*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "NVF4 (4-bit) GEMM for SM120 with BF16 I/O: D = A @ B (BF16 -> NVF4 quantize -> GEMM -> BF16)");

    m.def("nvf4_nvf4_sm120_available", []() {
        return pygpukit_nvf4_nvf4_sm120_available();
    }, "Check if pure NVF4 GEMM is available (SM120+)");

    m.def("benchmark_gemm_nvf4_sm120", [](GPUArray& D, int M, int N, int K) {
        if (D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("benchmark_gemm_nvf4_sm120: D must be bfloat16");
        }
        if (D.ndim() != 2) {
            throw std::runtime_error("benchmark_gemm_nvf4_sm120: D must be 2D");
        }

        cudaError_t err = pygpukit_benchmark_gemm_nvf4_sm120(
            static_cast<__nv_bfloat16*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("benchmark_gemm_nvf4_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"),
       "Benchmark pure NVF4 GEMM (pre-allocated data, no quantization overhead)");

    // ========================================================================
    // NVF4 GEMV for SM120 (M=1 path)
    // ========================================================================

    m.def("gemv_nvf4_available", []() {
        return pygpukit_gemv_nvf4_available();
    }, "Check if NVF4 GEMV is available (SM120+)");

    m.def("quantize_bf16_to_nvf4", [](const GPUArray& input, GPUArray& out_data, GPUArray& out_scale) {
        if (input.dtype() != DataType::BFloat16) {
            throw std::runtime_error("quantize_bf16_to_nvf4: input must be bfloat16");
        }
        if (input.ndim() != 2) {
            throw std::runtime_error("quantize_bf16_to_nvf4: input must be 2D [K, N]");
        }

        int K = input.shape()[0];
        int N = input.shape()[1];

        cudaError_t err = pygpukit_quantize_bf16_to_nvf4(
            input.data(), out_data.data(), out_scale.data(),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("quantize_bf16_to_nvf4 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("input"), py::arg("out_data"), py::arg("out_scale"),
       "Quantize BF16 weights to NVF4 format (column-major output [K/2,N]) for SM120 W4A16 GEMV");

    m.def("quantize_bf16_to_nvf4_rowmajor", [](const GPUArray& input, GPUArray& out_data, GPUArray& out_scale) {
        // Quantize BF16 to NVF4 with row-major output layout for pure NVF4/NVF4 GEMV
        // Input: [K, N] BF16 row-major
        // Output: [N, K/2] data, [N, K/32] scale (row-major, contiguous K for coalesced access)
        if (input.dtype() != DataType::BFloat16) {
            throw std::runtime_error("quantize_bf16_to_nvf4_rowmajor: input must be bfloat16");
        }
        if (input.ndim() != 2) {
            throw std::runtime_error("quantize_bf16_to_nvf4_rowmajor: input must be 2D [K, N]");
        }

        int K = input.shape()[0];
        int N = input.shape()[1];

        cudaError_t err = pygpukit_quantize_bf16_to_nvf4_rowmajor(
            input.data(), out_data.data(), out_scale.data(),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("quantize_bf16_to_nvf4_rowmajor failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("input"), py::arg("out_data"), py::arg("out_scale"),
       "Quantize BF16 weights to NVF4 format (row-major output [N,K/2]) for pure NVF4/NVF4 GEMV");

    m.def("gemv_nvf4_bf16", [](const GPUArray& A, const GPUArray& B_data, const GPUArray& B_scale, GPUArray& C, float alpha) {
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_nvf4_bf16: A and C must be bfloat16");
        }
        if (A.ndim() != 1) {
            throw std::runtime_error("gemv_nvf4_bf16: A must be 1D [K]");
        }

        int K = A.shape()[0];
        int N = C.shape()[0];

        cudaError_t err = pygpukit_gemv_nvf4_bf16(
            A.data(), B_data.data(), B_scale.data(), C.data(),
            K, N, alpha, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_nvf4_bf16 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_data"), py::arg("B_scale"), py::arg("C"), py::arg("alpha") = 1.0f,
       "NVF4 GEMV for SM120: C[N] = alpha * A[K] @ B[K,N] (NVF4 quantized weights)");

    m.def("gemv_bf16", [](const GPUArray& A, const GPUArray& B, GPUArray& C, float alpha, float beta) {
        if (A.dtype() != DataType::BFloat16 || B.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_bf16: all inputs must be bfloat16");
        }
        if (A.ndim() != 1 || B.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_bf16: A[K], B[K,N], C[N] dimensions required");
        }

        int K = A.shape()[0];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_bf16: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_bf16: N dimension mismatch");
        }

        cudaError_t err = pygpukit_gemv_bf16(
            A.data(), B.data(), C.data(),
            K, N, alpha, beta, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_bf16 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
       "BF16 GEMV: C[N] = alpha * A[K] @ B[K,N] + beta * C[N]");

    m.def("nvf4_get_sizes", [](int K, int N) {
        size_t data_size, scale_size;
        pygpukit_nvf4_get_sizes(K, N, &data_size, &scale_size);
        return py::make_tuple(data_size, scale_size);
    }, py::arg("K"), py::arg("N"),
       "Get buffer sizes for NVF4 quantization: returns (data_size, scale_size)");

    // ========================================================================
    // Optimized FP8 GEMV (warp-level reduction, smem, vectorized)
    // NOTE: Uses [N, K] weight layout for coalesced access
    // ========================================================================

    m.def("gemv_fp8_bf16_opt", [](const GPUArray& A, const GPUArray& B_nk, const GPUArray& B_scale, GPUArray& C) {
        // A: [K] BF16 activation
        // B_nk: [N, K] uint8 FP8 weights (row = output, NOT transposed)
        // B_scale: [N/128, K/128] BF16 scale factors
        // C: [N] BF16 output
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt: A and C must be bfloat16");
        }
        if (B_nk.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_fp8_bf16_opt: B_nk must be uint8 (FP8 E4M3)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt: B_scale must be bfloat16");
        }
        if (A.ndim() != 1 || B_nk.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_fp8_bf16_opt: A[K], B_nk[N,K], C[N] dimensions required");
        }

        int K = A.shape()[0];
        int N = B_nk.shape()[0];  // Note: N is first dim in [N, K] layout

        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_fp8_bf16_opt: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_fp8_bf16_opt: N dimension mismatch");
        }

        cudaError_t err = pygpukit::ops::gemv::launch_gemv_fp8_opt(
            reinterpret_cast<const __nv_bfloat16*>(A.data()),
            reinterpret_cast<const uint8_t*>(B_nk.data()),
            reinterpret_cast<const __nv_bfloat16*>(B_scale.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_fp8_bf16_opt failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("B_scale"), py::arg("C"),
       "Optimized FP8 GEMV: C[N] = A[K] @ B_nk[N,K]^T (warp-reduce, smem, vec4)");

    m.def("gemv_fp8_bf16_opt_batched", [](const GPUArray& A, const GPUArray& B_nk, const GPUArray& B_scale, GPUArray& C) {
        // A: [M, K] BF16 activation
        // B_nk: [N, K] uint8 FP8 weights (row = output)
        // B_scale: [N/128, K/128] BF16 scale factors
        // C: [M, N] BF16 output
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: A and C must be bfloat16");
        }
        if (B_nk.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: B_nk must be uint8 (FP8 E4M3)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: B_scale must be bfloat16");
        }
        if (A.ndim() != 2 || B_nk.ndim() != 2 || C.ndim() != 2) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: A[M,K], B_nk[N,K], C[M,N] dimensions required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_nk.shape()[0];  // Note: N is first dim in [N, K] layout

        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: output shape mismatch");
        }

        cudaError_t err = pygpukit::ops::gemv::launch_gemv_fp8_opt_batched(
            reinterpret_cast<const __nv_bfloat16*>(A.data()),
            reinterpret_cast<const uint8_t*>(B_nk.data()),
            reinterpret_cast<const __nv_bfloat16*>(B_scale.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, M, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("B_scale"), py::arg("C"),
       "Optimized batched FP8 GEMV: C[M,N] = A[M,K] @ B_nk[N,K]^T (warp-reduce, smem, vec4)");

    // ========================================================================
    // W8A16 GEMM: FP8 weight x BF16 activation -> BF16 output (SM120)
    // ========================================================================

    m.def("w8a16_gemm_sm120", [](const GPUArray& A, const GPUArray& B_fp8, const GPUArray& B_scale, GPUArray& C) {
        // A: [M, K] BF16 activation
        // B_fp8: [K, N] uint8 FP8 weights
        // B_scale: [K/128, N/128] BF16 scale factors
        // C: [M, N] BF16 output
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_gemm_sm120: A and C must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_gemm_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_gemm_sm120: B_scale must be bfloat16");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || C.ndim() != 2) {
            throw std::runtime_error("w8a16_gemm_sm120: A[M,K], B_fp8[K,N], C[M,N] dimensions required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[1];
        int scale_stride_n = (N + 127) / 128;

        if (B_fp8.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_gemm_sm120: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_gemm_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_w8a16_gemm_sm120(
            A.data(), B_fp8.data(), B_scale.data(), C.data(),
            M, N, K, scale_stride_n, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_gemm_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("B_scale"), py::arg("C"),
       "W8A16 GEMM: C[M,N] = A[M,K] @ B_fp8[K,N] (FP8 weight x BF16 activation with block-wise scale)");

    // ========================================================================
    // W8A16 GEMM using CUTLASS (SM120) - quantize BF16 to FP8, use FP8xFP8 TC
    // ========================================================================

    m.def("w8a16_cutlass_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        // A: [M, K] BF16 activation (will be quantized to FP8 internally)
        // B_fp8: [N, K] FP8 E4M3 weights (transposed, ColumnMajor for CUTLASS)
        //   - CUTLASS expects ColumnMajor B[K,N], which is stored as [N,K] RowMajor in memory
        //   - Python should pass B.T.contiguous() where B is [K,N]
        // D: [M, N] BF16 output
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_cutlass_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_cutlass_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("w8a16_cutlass_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        // B_fp8 is [N, K] transposed storage
        int N = B_fp8.shape()[0];

        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_cutlass_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_cutlass_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_w8a16_cutlass_sm120(
            A.data(), B_fp8.data(), D.data(),
            M, N, K,
            1.0f, 0.0f,  // alpha=1, beta=0
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_cutlass_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "W8A16 GEMM using CUTLASS: D[M,N] = A[M,K] @ B_fp8[N,K] (B transposed for ColumnMajor, quantizes BF16->FP8 internally)");

    // W8A16 GEMM using blockwise scaling (same compilation unit as working fp8_blockwise)
    m.def("w8a16_blockwise_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        // A: [M, K] BF16 activation
        // B_fp8: [N, K] FP8 E4M3 weights (transposed for ColumnMajor)
        // D: [M, N] BF16 output
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_blockwise_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_blockwise_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("w8a16_blockwise_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];  // B is [N, K] transposed

        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_blockwise_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_blockwise_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_w8a16_blockwise_sm120(
            A.data(), B_fp8.data(), D.data(),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_blockwise_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "W8A16 GEMM using blockwise: D[M,N] = A[M,K] @ B_fp8[N,K] (same kernel as working fp8_blockwise)");

    // Optimized W8A16 GEMM: Uses fast FP8xFP8 GEMM internally + type conversions
    // Expected ~220+ TFLOPS by combining:
    // 1. BF16->FP8 quantization (~67us)
    // 2. Fast FP8xFP8 GEMM (~237 TFLOPS)
    // 3. FP8->BF16 conversion (~157us)
    m.def("w8a16_optimized_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        // A: [M, K] BF16 activation
        // B_fp8: [N, K] FP8 E4M3 weights (transposed for ColumnMajor)
        // D: [M, N] BF16 output
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_optimized_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_optimized_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("w8a16_optimized_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];  // B is [N, K] transposed

        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_optimized_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_optimized_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_w8a16_optimized_sm120(
            A.data(),
            reinterpret_cast<const uint8_t*>(B_fp8.data()),
            D.data(),
            nullptr,  // scale_A will use unity scales internally
            nullptr,  // scale_B will use unity scales internally
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_optimized_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "Optimized W8A16 GEMM: D[M,N] = A[M,K] @ B_fp8[N,K] (uses fast FP8xFP8 internally, ~220+ TFLOPS expected)");

    // ========================================================================
    // Grouped GEMM for MoE (FP8 weights x BF16 activations)
    // ========================================================================

    m.def("grouped_gemm_init_lut", []() {
        cudaError_t err = pygpukit_grouped_gemm_init_lut();
        if (err != cudaSuccess) {
            throw std::runtime_error("grouped_gemm_init_lut failed: " + std::string(cudaGetErrorString(err)));
        }
    }, "Initialize FP8->BF16 LUT for grouped GEMM");

    m.def("grouped_gemm_fp8_bf16", [](
        const GPUArray& A,              // [M, K] BF16
        const GPUArray& B_stacked,      // [num_experts, N, K] FP8
        const GPUArray& B_scale,        // [num_experts, N/128, K/128] BF16
        GPUArray& C,                    // [M, N] BF16
        const GPUArray& row_expert_ids  // [M] int32 - expert ID per row
    ) {
        // Validate dtypes
        if (A.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: A must be bfloat16");
        }
        if (B_stacked.dtype() != DataType::UInt8) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: B_stacked must be uint8 (FP8)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: B_scale must be bfloat16");
        }
        if (C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: C must be bfloat16");
        }
        if (row_expert_ids.dtype() != DataType::Int32) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: row_expert_ids must be int32");
        }

        // Validate dimensions
        if (A.ndim() != 2 || B_stacked.ndim() != 3 || C.ndim() != 2) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: invalid dimensions");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_stacked.shape()[1];

        if (B_stacked.shape()[2] != static_cast<size_t>(K)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: output shape mismatch");
        }
        if (row_expert_ids.ndim() != 1 || row_expert_ids.shape()[0] != static_cast<size_t>(M)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: row_expert_ids size mismatch");
        }

        cudaError_t err = pygpukit_grouped_gemm_fp8_bf16(
            A.data(), B_stacked.data(), B_scale.data(), C.data(),
            reinterpret_cast<const int*>(row_expert_ids.data()),
            M, N, K, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("grouped_gemm_fp8_bf16 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_stacked"), py::arg("B_scale"), py::arg("C"), py::arg("row_expert_ids"),
       "Grouped GEMM for MoE: C[M,N] = A[M,K] @ B_stacked[experts,N,K] with per-row expert IDs");

    // ========================================================================
    // Int8 GEMM via FP8 approximation (SM120)
    // SM120 has no native Int8 TensorCore, so we use FP8 as approximation
    // ========================================================================
    // Native Int8 GEMM using dp4a CUDA cores (exact computation)
    // Uses CUDA dp4a instruction for 4xInt8 dot product with Int32 accumulation
    // Slower than TensorCore but provides exact integer arithmetic
    // ========================================================================

    m.def("int8_native_gemm_available", []() {
        return pygpukit_int8_native_gemm_available();
    }, "Check if native Int8 GEMM is available (uses dp4a CUDA cores)");

    m.def("int8_native_gemm_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D
    ) {
        // A: [M, K] Int8 (RowMajor)
        // B: [N, K] Int8 (stored as transposed for ColumnMajor)
        // D: [M, N] Int32
        if (A.dtype() != DataType::Int8) {
            throw std::runtime_error("int8_native_gemm_sm120: A must be int8");
        }
        if (B.dtype() != DataType::Int8) {
            throw std::runtime_error("int8_native_gemm_sm120: B must be int8");
        }
        if (D.dtype() != DataType::Int32) {
            throw std::runtime_error("int8_native_gemm_sm120: D must be int32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("int8_native_gemm_sm120: A[M,K], B[N,K], D[M,N] required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[0];  // B is [N, K] transposed

        if (B.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("int8_native_gemm_sm120: K dimension mismatch");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("int8_native_gemm_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_int8_native_sm120(
            reinterpret_cast<const int8_t*>(A.data()),
            reinterpret_cast<const int8_t*>(B.data()),
            reinterpret_cast<int32_t*>(D.data()),
            M, N, K,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("int8_native_gemm_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "Native Int8 GEMM using dp4a: D[M,N] = A[M,K] @ B[N,K]^T with exact Int32 output");

    // ========================================================================
    // Int4 GEMM via Int8/FP8 approximation (SM120)
    // SM120 has no native Int4 TensorCore, so we unpack Int4->Int8 and use FP8
    // Input is packed: 2 signed 4-bit values per byte (low nibble first)
    // ========================================================================

    m.def("int4_gemm_available", []() {
        return pygpukit_int4_gemm_sm120_available();
    }, "Check if Int4 GEMM is available (SM120 via Int8/FP8 approximation)");

    // Int4 GEMM with Int32 output (for full precision accumulation)
    m.def("int4_gemm_int32_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D,
        float scale_A, float scale_B, float descale_D
    ) {
        // A: [M, K/2] UInt8 packed (K is unpacked dimension)
        // B: [N, K/2] UInt8 packed (stored as transposed for ColumnMajor)
        // D: [M, N] Int32
        if (A.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int32_sm120: A must be uint8 (packed int4)");
        }
        if (B.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int32_sm120: B must be uint8 (packed int4)");
        }
        if (D.dtype() != DataType::Int32) {
            throw std::runtime_error("int4_gemm_int32_sm120: D must be int32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("int4_gemm_int32_sm120: A[M,K/2], B[N,K/2], D[M,N] required");
        }

        int M = A.shape()[0];
        int K_packed = A.shape()[1];
        int K = K_packed * 2;  // Unpacked K dimension
        int N = B.shape()[0];  // B is [N, K/2] transposed

        if (B.shape()[1] != static_cast<size_t>(K_packed)) {
            throw std::runtime_error("int4_gemm_int32_sm120: K dimension mismatch");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("int4_gemm_int32_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_int4_int4_int32_sm120(
            reinterpret_cast<const uint8_t*>(A.data()),
            reinterpret_cast<const uint8_t*>(B.data()),
            reinterpret_cast<int32_t*>(D.data()),
            M, N, K,
            scale_A, scale_B, descale_D,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("int4_gemm_int32_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("scale_A") = 1.0f, py::arg("scale_B") = 1.0f, py::arg("descale_D") = 1.0f,
       "Int4 GEMM via Int8/FP8: D[M,N] = A[M,K] @ B[N,K]^T with Int32 output. Input is packed int4.");

    // Int4 GEMM with Int8 output (for quantized inference)
    m.def("int4_gemm_int8_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D,
        float scale_A, float scale_B, float descale_D
    ) {
        // A: [M, K/2] UInt8 packed (K is unpacked dimension)
        // B: [N, K/2] UInt8 packed (stored as transposed for ColumnMajor)
        // D: [M, N] Int8
        if (A.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int8_sm120: A must be uint8 (packed int4)");
        }
        if (B.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int8_sm120: B must be uint8 (packed int4)");
        }
        if (D.dtype() != DataType::Int8) {
            throw std::runtime_error("int4_gemm_int8_sm120: D must be int8");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("int4_gemm_int8_sm120: A[M,K/2], B[N,K/2], D[M,N] required");
        }

        int M = A.shape()[0];
        int K_packed = A.shape()[1];
        int K = K_packed * 2;  // Unpacked K dimension
        int N = B.shape()[0];  // B is [N, K/2] transposed

        if (B.shape()[1] != static_cast<size_t>(K_packed)) {
            throw std::runtime_error("int4_gemm_int8_sm120: K dimension mismatch");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("int4_gemm_int8_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_int4_int4_int8_sm120(
            reinterpret_cast<const uint8_t*>(A.data()),
            reinterpret_cast<const uint8_t*>(B.data()),
            reinterpret_cast<int8_t*>(D.data()),
            M, N, K,
            scale_A, scale_B, descale_D,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("int4_gemm_int8_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("scale_A") = 1.0f, py::arg("scale_B") = 1.0f, py::arg("descale_D") = 1.0f,
       "Int4 GEMM via Int8/FP8: D[M,N] = A[M,K] @ B[N,K]^T with Int8 output. Input is packed int4.");

    // ========================================================================
    // Int4 GEMV for M=1 decode (SM120)
    // Input is packed: 2 signed 4-bit values per byte (low nibble first)
    // ========================================================================

    m.def("int4_gemv_available", []() {
        return pygpukit_int4_gemv_sm120_available();
    }, "Check if Int4 GEMV is available (SM120 for M=1 decode)");

    // Int4 GEMV with Int32 output
    m.def("int4_gemv_int32_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& C,
        float scale_A, float scale_B
    ) {
        // A: [K/2] UInt8 packed (activation vector)
        // B: [N, K/2] UInt8 packed (weights, row-major)
        // C: [N] Int32
        if (A.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemv_int32_sm120: A must be uint8 (packed int4)");
        }
        if (B.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemv_int32_sm120: B must be uint8 (packed int4)");
        }
        if (C.dtype() != DataType::Int32) {
            throw std::runtime_error("int4_gemv_int32_sm120: C must be int32");
        }
        if (A.ndim() != 1) {
            throw std::runtime_error("int4_gemv_int32_sm120: A must be 1D [K/2]");
        }
        if (B.ndim() != 2) {
            throw std::runtime_error("int4_gemv_int32_sm120: B must be 2D [N, K/2]");
        }
        if (C.ndim() != 1) {
            throw std::runtime_error("int4_gemv_int32_sm120: C must be 1D [N]");
        }

        int K_packed = A.shape()[0];
        int K = K_packed * 2;  // Unpacked K dimension
        int N = B.shape()[0];

        if (B.shape()[1] != static_cast<size_t>(K_packed)) {
            throw std::runtime_error("int4_gemv_int32_sm120: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("int4_gemv_int32_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemv_int4_int4_int32_sm120(
            reinterpret_cast<const uint8_t*>(A.data()),
            reinterpret_cast<const uint8_t*>(B.data()),
            reinterpret_cast<int32_t*>(C.data()),
            K, N,
            scale_A, scale_B,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("int4_gemv_int32_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("scale_A") = 1.0f, py::arg("scale_B") = 1.0f,
       "Int4 GEMV: C[N] = A[K] . B[N,K]^T with Int32 output. Input is packed int4.");

    // ========================================================================
    // Pure FP8/FP8/FP8 GEMV (SM120)
    // A[K](FP8) x B[N,K](FP8) -> C[N](BF16 or FP8)
    // Advantage: A is FP8 (1 byte) so shared memory is halved vs W8A16
    // ========================================================================

    m.def("gemv_fp8_fp8_available", []() {
        return pygpukit_gemv_fp8_fp8_sm120_available();
    }, "Check if pure FP8/FP8 GEMV is available (SM120)");

    m.def("gemv_fp8_fp8_bf16_sm120", [](
        const GPUArray& A, const GPUArray& B_nk,
        const GPUArray& scale_A, const GPUArray& scale_B,
        GPUArray& C
    ) {
        // A: [K] FP8 E4M3 (stored as uint8)
        // B_nk: [N, K] FP8 E4M3 (stored as uint8)
        // scale_A: [K/128] FP32 blockwise scales
        // scale_B: [N/128, K/128] FP32 blockwise scales
        // C: [N] BF16 output
        if (A.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: A must be uint8 (FP8 E4M3)");
        }
        if (B_nk.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: B_nk must be uint8 (FP8 E4M3)");
        }
        if (scale_A.dtype() != DataType::Float32) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: scale_A must be float32");
        }
        if (scale_B.dtype() != DataType::Float32) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: scale_B must be float32");
        }
        if (C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: C must be bfloat16");
        }
        if (A.ndim() != 1 || B_nk.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: A[K], B_nk[N,K], C[N] dimensions required");
        }

        int K = A.shape()[0];
        int N = B_nk.shape()[0];

        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_fp8_fp8_bf16: N dimension mismatch");
        }

        cudaError_t err = pygpukit_gemv_fp8_fp8_bf16_sm120(
            reinterpret_cast<const uint8_t*>(A.data()),
            reinterpret_cast<const uint8_t*>(B_nk.data()),
            reinterpret_cast<const float*>(scale_A.data()),
            reinterpret_cast<const float*>(scale_B.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_fp8_fp8_bf16 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("scale_A"), py::arg("scale_B"), py::arg("C"),
       "Pure FP8 GEMV: C[N](BF16) = A[K](FP8) @ B_nk[N,K](FP8)^T with blockwise scaling");

    m.def("gemv_fp8_fp8_fp8_sm120", [](
        const GPUArray& A, const GPUArray& B_nk,
        const GPUArray& scale_A, const GPUArray& scale_B,
        GPUArray& C, float scale_C
    ) {
        // A: [K] FP8 E4M3 (stored as uint8)
        // B_nk: [N, K] FP8 E4M3 (stored as uint8)
        // scale_A: [K/128] FP32 blockwise scales
        // scale_B: [N/128, K/128] FP32 blockwise scales
        // C: [N] FP8 output (stored as uint8)
        if (A.dtype() != DataType::UInt8 || B_nk.dtype() != DataType::UInt8 || C.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_fp8_fp8_fp8: A, B, C must be uint8 (FP8 E4M3)");
        }
        if (scale_A.dtype() != DataType::Float32 || scale_B.dtype() != DataType::Float32) {
            throw std::runtime_error("gemv_fp8_fp8_fp8: scales must be float32");
        }
        if (A.ndim() != 1 || B_nk.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_fp8_fp8_fp8: A[K], B_nk[N,K], C[N] dimensions required");
        }

        int K = A.shape()[0];
        int N = B_nk.shape()[0];

        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_fp8_fp8_fp8: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_fp8_fp8_fp8: N dimension mismatch");
        }

        cudaError_t err = pygpukit_gemv_fp8_fp8_fp8_sm120(
            reinterpret_cast<const uint8_t*>(A.data()),
            reinterpret_cast<const uint8_t*>(B_nk.data()),
            reinterpret_cast<const float*>(scale_A.data()),
            reinterpret_cast<const float*>(scale_B.data()),
            reinterpret_cast<uint8_t*>(C.data()),
            scale_C,
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_fp8_fp8_fp8 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("scale_A"), py::arg("scale_B"), py::arg("C"), py::arg("scale_C"),
       "Pure FP8 GEMV: C[N](FP8) = A[K](FP8) @ B_nk[N,K](FP8)^T with blockwise scaling and FP8 output");

    // ========================================================================
    // Pure NVF4/NVF4/NVF4 GEMV (SM120)
    // ========================================================================

    m.def("gemv_nvf4_nvf4_available", []() {
        return pygpukit_gemv_nvf4_nvf4_sm120_available();
    }, "Check if pure NVF4/NVF4 GEMV is available (SM120)");

    m.def("gemv_nvf4_nvf4_bf16_sm120", [](
        const GPUArray& A_data, const GPUArray& A_scale,
        const GPUArray& B_data, const GPUArray& B_scale,
        GPUArray& C
    ) {
        // A_data: [K/2] packed NVF4 (2 values per byte)
        // A_scale: [K/32] UE4M3 scales
        // B_data: [N, K/2] packed NVF4 (row-major, from quantize_bf16_to_nvf4_rowmajor)
        // B_scale: [N, K/32] UE4M3 scales (row-major)
        // C: [N] BF16 output
        if (A_data.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: A_data must be uint8 (packed NVF4)");
        }
        if (A_scale.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: A_scale must be uint8 (UE4M3)");
        }
        if (B_data.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: B_data must be uint8 (packed NVF4)");
        }
        if (B_scale.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: B_scale must be uint8 (UE4M3)");
        }
        if (C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: C must be bfloat16");
        }
        if (A_data.ndim() != 1 || B_data.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: A_data[K/2], B_data[N,K/2], C[N] dimensions required");
        }

        // B_data is [N, K/2] row-major from quantize_bf16_to_nvf4_rowmajor
        int N = static_cast<int>(B_data.shape()[0]);
        int K_packed = static_cast<int>(B_data.shape()[1]);
        int K = K_packed * 2;

        if (A_data.shape()[0] != static_cast<size_t>(K_packed)) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: A_data K/2 dimension mismatch with B_data");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16: C N dimension mismatch");
        }

        cudaError_t err = pygpukit_gemv_nvf4_nvf4_bf16_sm120(
            reinterpret_cast<const uint8_t*>(A_data.data()),
            reinterpret_cast<const uint8_t*>(A_scale.data()),
            reinterpret_cast<const uint8_t*>(B_data.data()),
            reinterpret_cast<const uint8_t*>(B_scale.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_nvf4_nvf4_bf16 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A_data"), py::arg("A_scale"), py::arg("B_data"), py::arg("B_scale"), py::arg("C"),
       "Pure NVF4 GEMV: C[N](BF16) = A[K](NVF4) @ B[K,N](NVF4) with row-major B for coalesced access");

    // ========================================================================
    // FP8 GEMM auto-dispatch (selects best available backend)
    // Priority: SM120 (if enabled) > SM90 > error
    // ========================================================================

    m.def("fp8_available", []() {
        // Check all FP8 backends: SM120 (disabled), SM100, SM90
        return pygpukit_fp8_sm120_available() ||
               pygpukit_fp8_sm100_available() ||
               pygpukit_fp8_sm90_available();
    }, "Check if FP8 GEMM is available (any backend)");

    m.def("gemm_fp8", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8: D shape mismatch");
        }

        cudaError_t err;

        // Try SM120 first (when CUTLASS bug is fixed, this will be preferred)
        if (pygpukit_fp8_sm120_available()) {
            err = pygpukit_gemm_fp8_sm120(
                static_cast<const float*>(A.data()),
                static_cast<const float*>(B.data()),
                static_cast<float*>(D.data()),
                M, N, K, 1.0f, 0.0f, nullptr
            );
            if (err == cudaSuccess) return;
            // Fall through to SM100 if SM120 fails
        }

        // Try SM100 (Blackwell datacenter - potential fallback for SM120)
        if (pygpukit_fp8_sm100_available()) {
            err = pygpukit_gemm_fp8_sm100(
                static_cast<const float*>(A.data()),
                static_cast<const float*>(B.data()),
                static_cast<float*>(D.data()),
                M, N, K, 1.0f, 0.0f, nullptr
            );
            if (err == cudaSuccess) return;
            // Fall through to SM90 if SM100 fails
        }

        // Try SM90 (Hopper)
        if (pygpukit_fp8_sm90_available()) {
            err = pygpukit_gemm_fp8_sm90(
                static_cast<const float*>(A.data()),
                static_cast<const float*>(B.data()),
                static_cast<float*>(D.data()),
                M, N, K, 1.0f, 0.0f, nullptr
            );
            if (err != cudaSuccess) {
                throw std::runtime_error("gemm_fp8 (SM90) failed: " + std::string(cudaGetErrorString(err)));
            }
            return;
        }

        throw std::runtime_error("gemm_fp8: no FP8 backend available (requires SM90+)");
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "FP8 GEMM with auto backend selection: D = A @ B");

    // ========================================================================
    // MoE (Mixture of Experts) operations
    // ========================================================================

    m.def("moe_topk_with_indices", [](
        const GPUArray& logits,  // [num_tokens, num_experts]
        GPUArray& values,        // [num_tokens, k]
        GPUArray& indices,       // [num_tokens, k] int32
        int k
    ) {
        if (logits.ndim() != 2) {
            throw std::runtime_error("moe_topk_with_indices: logits must be 2D [num_tokens, num_experts]");
        }
        int num_tokens = logits.shape()[0];
        int num_experts = logits.shape()[1];

        if (values.shape()[0] != static_cast<size_t>(num_tokens) ||
            values.shape()[1] != static_cast<size_t>(k)) {
            throw std::runtime_error("moe_topk_with_indices: values shape mismatch");
        }
        if (indices.dtype() != DataType::Int32) {
            throw std::runtime_error("moe_topk_with_indices: indices must be int32");
        }

        if (logits.dtype() == DataType::Float32) {
            moe::topk_with_indices_f32(
                static_cast<const float*>(logits.data()),
                static_cast<float*>(values.data()),
                static_cast<int32_t*>(indices.data()),
                num_tokens, num_experts, k, nullptr
            );
        } else if (logits.dtype() == DataType::BFloat16) {
            moe::topk_with_indices_bf16(
                static_cast<const __nv_bfloat16*>(logits.data()),
                static_cast<__nv_bfloat16*>(values.data()),
                static_cast<int32_t*>(indices.data()),
                num_tokens, num_experts, k, nullptr
            );
        } else {
            throw std::runtime_error("moe_topk_with_indices: unsupported dtype");
        }
    }, py::arg("logits"), py::arg("values"), py::arg("indices"), py::arg("k"),
       "MoE Top-K selection: select top-k experts per token");

    m.def("moe_softmax_topk", [](GPUArray& values, int k) {
        if (values.ndim() != 2) {
            throw std::runtime_error("moe_softmax_topk: values must be 2D [num_tokens, k]");
        }
        int num_tokens = values.shape()[0];

        if (values.dtype() == DataType::Float32) {
            moe::softmax_topk_f32(
                static_cast<float*>(values.data()),
                num_tokens, k, nullptr
            );
        } else if (values.dtype() == DataType::BFloat16) {
            moe::softmax_topk_bf16(
                static_cast<__nv_bfloat16*>(values.data()),
                num_tokens, k, nullptr
            );
        } else {
            throw std::runtime_error("moe_softmax_topk: unsupported dtype");
        }
    }, py::arg("values"), py::arg("k"),
       "Softmax over top-k selected experts (in-place)");

    m.def("moe_compute_permutation", [](
        const GPUArray& expert_indices,  // [num_tokens, k] int32
        GPUArray& expert_counts,         // [num_experts] int32
        GPUArray& expert_offsets,        // [num_experts + 1] int32
        GPUArray& permute_indices,       // [num_tokens * k] int32
        GPUArray& reverse_perm,          // [num_tokens * k] int32
        int num_experts, int k
    ) {
        if (expert_indices.dtype() != DataType::Int32) {
            throw std::runtime_error("moe_compute_permutation: expert_indices must be int32");
        }
        int num_tokens = expert_indices.shape()[0];

        moe::moe_compute_permutation(
            static_cast<const int32_t*>(expert_indices.data()),
            static_cast<int32_t*>(expert_counts.data()),
            static_cast<int32_t*>(expert_offsets.data()),
            static_cast<int32_t*>(permute_indices.data()),
            static_cast<int32_t*>(reverse_perm.data()),
            num_tokens, num_experts, k, nullptr
        );
    }, py::arg("expert_indices"), py::arg("expert_counts"), py::arg("expert_offsets"),
       py::arg("permute_indices"), py::arg("reverse_perm"),
       py::arg("num_experts"), py::arg("k"),
       "Compute MoE permutation indices for token routing");

    m.def("moe_gather", [](
        const GPUArray& hidden,           // [num_tokens, hidden_size]
        const GPUArray& permute_indices,  // [num_tokens * k]
        GPUArray& gathered,               // [num_tokens * k, hidden_size]
        int k
    ) {
        if (hidden.ndim() != 2) {
            throw std::runtime_error("moe_gather: hidden must be 2D");
        }
        int num_tokens = hidden.shape()[0];
        int hidden_size = hidden.shape()[1];

        if (hidden.dtype() == DataType::Float32) {
            moe::moe_gather_f32(
                static_cast<const float*>(hidden.data()),
                static_cast<const int32_t*>(permute_indices.data()),
                static_cast<float*>(gathered.data()),
                num_tokens, hidden_size, k, nullptr
            );
        } else if (hidden.dtype() == DataType::BFloat16) {
            moe::moe_gather_bf16(
                static_cast<const __nv_bfloat16*>(hidden.data()),
                static_cast<const int32_t*>(permute_indices.data()),
                static_cast<__nv_bfloat16*>(gathered.data()),
                num_tokens, hidden_size, k, nullptr
            );
        } else {
            throw std::runtime_error("moe_gather: unsupported dtype");
        }
    }, py::arg("hidden"), py::arg("permute_indices"), py::arg("gathered"), py::arg("k"),
       "Gather hidden states according to MoE permutation");

    m.def("moe_scatter", [](
        const GPUArray& expert_outputs,   // [num_tokens * k, hidden_size]
        const GPUArray& router_weights,   // [num_tokens, k]
        const GPUArray& reverse_perm,     // [num_tokens * k]
        GPUArray& output,                 // [num_tokens, hidden_size]
        int k
    ) {
        if (output.ndim() != 2) {
            throw std::runtime_error("moe_scatter: output must be 2D");
        }
        int num_tokens = output.shape()[0];
        int hidden_size = output.shape()[1];

        if (output.dtype() == DataType::Float32) {
            moe::moe_scatter_f32(
                static_cast<const float*>(expert_outputs.data()),
                static_cast<const float*>(router_weights.data()),
                static_cast<const int32_t*>(reverse_perm.data()),
                static_cast<float*>(output.data()),
                num_tokens, hidden_size, k, nullptr
            );
        } else if (output.dtype() == DataType::BFloat16) {
            moe::moe_scatter_bf16(
                static_cast<const __nv_bfloat16*>(expert_outputs.data()),
                static_cast<const __nv_bfloat16*>(router_weights.data()),
                static_cast<const int32_t*>(reverse_perm.data()),
                static_cast<__nv_bfloat16*>(output.data()),
                num_tokens, hidden_size, k, nullptr
            );
        } else {
            throw std::runtime_error("moe_scatter: unsupported dtype");
        }
    }, py::arg("expert_outputs"), py::arg("router_weights"), py::arg("reverse_perm"),
       py::arg("output"), py::arg("k"),
       "Scatter and combine expert outputs with router weights");

    m.def("moe_expand_expert_offsets", [](
        const GPUArray& expert_offsets,    // [num_experts + 1] int32
        GPUArray& row_expert_ids,          // [M_total] int32
        int num_experts
    ) {
        if (expert_offsets.dtype() != DataType::Int32) {
            throw std::runtime_error("moe_expand_expert_offsets: expert_offsets must be int32");
        }
        if (row_expert_ids.dtype() != DataType::Int32) {
            throw std::runtime_error("moe_expand_expert_offsets: row_expert_ids must be int32");
        }
        if (expert_offsets.ndim() != 1 || expert_offsets.shape()[0] != static_cast<size_t>(num_experts + 1)) {
            throw std::runtime_error("moe_expand_expert_offsets: expert_offsets size mismatch");
        }

        int M_total = row_expert_ids.shape()[0];

        moe::expand_expert_offsets(
            reinterpret_cast<const int32_t*>(expert_offsets.data()),
            reinterpret_cast<int32_t*>(row_expert_ids.data()),
            num_experts, M_total, nullptr
        );
    }, py::arg("expert_offsets"), py::arg("row_expert_ids"), py::arg("num_experts"),
       "Expand expert_offsets to per-row expert IDs for grouped GEMM v2");
}
