/**
 * MoE (Mixture of Experts) operations
 */
#include "bindings_common.hpp"

// MoE functions - defined in ops/moe/moe.cu
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

using namespace pygpukit;

void init_moe(py::module_& m) {
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
