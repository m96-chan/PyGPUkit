#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace pygpukit {
namespace ops {
namespace tf32 {

// ============================================================
// Test 1: B を row_major で読み込む
// ============================================================
__global__ void sgemm_wmma_row_row(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    using namespace nvcuda::wmma;
    
    // A: row_major, B: row_major
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
    fragment<accumulator, 16, 16, 8, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 8) {
        // A[0:16, k:k+8], stride = K
        load_matrix_sync(a_frag, A + k, K);
        // B[k:k+8, 0:16], stride = N
        load_matrix_sync(b_frag, B + k * N, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    store_matrix_sync(C, c_frag, N, mem_row_major);
}

// ============================================================
// Test 2: B を転置して col_major で読み込む
// ============================================================
__global__ void sgemm_wmma_row_col_transposed(
    const float* A, const float* B_transposed, float* C,
    int M, int N, int K
) {
    using namespace nvcuda::wmma;
    
    // B_transposed is N x K (col-major storage of K x N matrix)
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
    fragment<accumulator, 16, 16, 8, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 8) {
        load_matrix_sync(a_frag, A + k, K);
        // B_transposed[0:N, k:k+8], stride = K
        load_matrix_sync(b_frag, B_transposed + k, K);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    store_matrix_sync(C, c_frag, N, mem_row_major);
}

// ============================================================
// Launcher for row_row version
// ============================================================
inline cudaError_t launch_wmma_row_row(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    sgemm_wmma_row_row<<<1, 32, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

// ============================================================
// Debug: Dump WMMA fragment contents
// Output: A_out[32 * num_elements_a], B_out[32 * num_elements_b]
// Each thread dumps its fragment elements
// ============================================================
__global__ void debug_dump_fragments(
    const float* A, const float* B,
    float* A_out, float* B_out,
    int K, int N
) {
    using namespace nvcuda::wmma;

    int lane = threadIdx.x;
    if (lane >= 32) return;

    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;

    // Load first K-tile only
    load_matrix_sync(a_frag, A, K);
    load_matrix_sync(b_frag, B, N);

    // Dump A fragment (4 elements per thread for 16x16x8)
    for (int i = 0; i < a_frag.num_elements; i++) {
        A_out[lane * a_frag.num_elements + i] = a_frag.x[i];
    }

    // Dump B fragment (4 elements per thread for 16x16x8)
    for (int i = 0; i < b_frag.num_elements; i++) {
        B_out[lane * b_frag.num_elements + i] = b_frag.x[i];
    }
}

inline cudaError_t launch_dump_fragments(
    const float* A, const float* B,
    float* A_out, float* B_out,
    int K, int N,
    cudaStream_t stream = 0
) {
    debug_dump_fragments<<<1, 32, 0, stream>>>(A, B, A_out, B_out, K, N);
    return cudaGetLastError();
}

} // namespace tf32
} // namespace ops
} // namespace pygpukit