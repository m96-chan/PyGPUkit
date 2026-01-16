/**
 * Common header for all bindings files
 * Contains shared includes, namespaces, and forward declarations
 */
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../ops/ops.cuh"
#include "../ops/audio/audio.hpp"
#include "../jit/cublaslt_loader.hpp"
#include "../jit/cublas_loader.hpp"

namespace py = pybind11;
using namespace pygpukit;

// Forward declarations for init functions
void init_elementwise_binary(py::module_& m);
void init_elementwise_inplace(py::module_& m);
void init_elementwise_compare(py::module_& m);

void init_unary_math(py::module_& m);
void init_unary_trig(py::module_& m);

void init_reduction_basic(py::module_& m);
void init_reduction_argmax(py::module_& m);
void init_reduction_softmax(py::module_& m);

void init_tensor_cast(py::module_& m);
void init_tensor_transpose(py::module_& m);
void init_tensor_reshape(py::module_& m);
void init_tensor_repeat(py::module_& m);

void init_nn_activation(py::module_& m);
void init_nn_norm(py::module_& m);
void init_nn_attention(py::module_& m);
void init_nn_rope(py::module_& m);
void init_nn_recurrent(py::module_& m);
void init_nn_diffusion(py::module_& m);
void init_nn_conv(py::module_& m);
void init_nn_llama4(py::module_& m);

void init_embedding_lookup(py::module_& m);
void init_embedding_kv_cache(py::module_& m);

void init_gemm_generic(py::module_& m);
void init_gemm_fp8xfp8_bf16(py::module_& m);
void init_gemm_fp8xfp8_fp8(py::module_& m);
void init_gemm_fp8xbf16_bf16(py::module_& m);
void init_gemm_nvf4xbf16_bf16(py::module_& m);
void init_gemm_grouped(py::module_& m);
void init_gemm_int(py::module_& m);

void init_gemv_generic(py::module_& m);
void init_gemv_fp8xfp8_bf16(py::module_& m);
void init_gemv_nvf4xbf16_bf16(py::module_& m);

void init_sampling_basic(py::module_& m);
void init_sampling_topk(py::module_& m);
void init_sampling_seed(py::module_& m);

void init_quantize(py::module_& m);
void init_paged_attention(py::module_& m);
void init_continuous_batching(py::module_& m);
void init_audio(py::module_& m);
void init_cublaslt(py::module_& m);
void init_cublas(py::module_& m);
void init_moe(py::module_& m);
