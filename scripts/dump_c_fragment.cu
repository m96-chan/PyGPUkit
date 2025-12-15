#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

// PTX mma.sync m16n8k8 uses:
// A: 16x8 (row major)
// B: 8x8 (col major, transposed)
// C: 16x8

__global__ void dump_c_fragment_ptx(
    float* C_out  // 32 threads * 4 elements
) {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    // Initialize accumulators to identifiable values
    // We'll set acc[i] = thread * 10 + i so we can track where each value ends up
    float acc0 = lane * 10 + 0;
    float acc1 = lane * 10 + 1;
    float acc2 = lane * 10 + 2;
    float acc3 = lane * 10 + 3;

    // Output without doing mma - just to see the initial mapping
    C_out[lane * 4 + 0] = acc0;
    C_out[lane * 4 + 1] = acc1;
    C_out[lane * 4 + 2] = acc2;
    C_out[lane * 4 + 3] = acc3;
}

// Test store_matrix_sync with WMMA to see C fragment mapping
using namespace nvcuda::wmma;

__global__ void dump_c_fragment_wmma(
    float* C_mat,  // 16x16 output matrix
    float* C_frag_out,  // 32 threads * 8 elements
    int N
) {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    fragment<accumulator, 16, 16, 8, float> c_frag;

    // Initialize each element with identifiable value
    // c_frag.x[i] = lane * 10 + i
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = lane * 10 + i;
    }

    // Store to matrix using WMMA
    store_matrix_sync(C_mat, c_frag, N, mem_row_major);

    // Also dump raw fragment
    for (int i = 0; i < c_frag.num_elements; i++) {
        C_frag_out[lane * c_frag.num_elements + i] = c_frag.x[i];
    }
}

int main() {
    const int M = 16, N = 16;

    printf("=== C Fragment Mapping Analysis ===\n\n");

    // Test 1: PTX accumulator positions
    printf("=== Test 1: PTX m16n8k8 accumulator positions ===\n");
    printf("For PTX mma.sync.m16n8k8, C is 16x8\n");
    printf("Each thread has 4 accumulators: acc0, acc1, acc2, acc3\n\n");

    float *d_C_ptx;
    cudaMalloc(&d_C_ptx, 32 * 4 * sizeof(float));

    dump_c_fragment_ptx<<<1, 32>>>(d_C_ptx);
    cudaDeviceSynchronize();

    float h_C_ptx[32 * 4];
    cudaMemcpy(h_C_ptx, d_C_ptx, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Thread | acc[0]     | acc[1]     | acc[2]     | acc[3]     | Pattern\n");
    printf("-------|------------|------------|------------|------------|---------\n");
    for (int t = 0; t < 32; t++) {
        printf("  %2d   |", t);
        for (int i = 0; i < 4; i++) {
            printf(" %10.0f |", h_C_ptx[t * 4 + i]);
        }
        // Pattern analysis
        int row_base = t / 4;
        int col_base = t % 4;
        printf(" row=%d, col=%d\n", row_base, col_base);
    }

    cudaFree(d_C_ptx);

    // Test 2: WMMA accumulator -> matrix mapping
    printf("\n=== Test 2: WMMA 16x16x8 accumulator fragment ===\n");
    printf("Each thread has 8 elements in accumulator fragment\n\n");

    float *d_C_mat, *d_C_frag;
    cudaMalloc(&d_C_mat, M * N * sizeof(float));
    cudaMalloc(&d_C_frag, 32 * 8 * sizeof(float));
    cudaMemset(d_C_mat, 0, M * N * sizeof(float));

    dump_c_fragment_wmma<<<1, 32>>>(d_C_mat, d_C_frag, N);
    cudaDeviceSynchronize();

    float h_C_mat[M * N], h_C_frag[32 * 8];
    cudaMemcpy(h_C_mat, d_C_mat, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_frag, d_C_frag, 32 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Raw C fragment per thread (8 elements each):\n");
    printf("Thread | c[0]  | c[1]  | c[2]  | c[3]  | c[4]  | c[5]  | c[6]  | c[7]  |\n");
    printf("-------|-------|-------|-------|-------|-------|-------|-------|-------|\n");
    for (int t = 0; t < 32; t++) {
        printf("  %2d   |", t);
        for (int i = 0; i < 8; i++) {
            printf(" %5.0f |", h_C_frag[t * 8 + i]);
        }
        printf("\n");
    }

    printf("\n\nC matrix after store_matrix_sync (16x16):\n");
    printf("     ");
    for (int j = 0; j < N; j++) printf("%6d ", j);
    printf("\n");
    for (int i = 0; i < M; i++) {
        printf("%2d: ", i);
        for (int j = 0; j < N; j++) {
            float val = h_C_mat[i * N + j];
            printf("%6.0f ", val);
        }
        printf("\n");
    }

    printf("\n\nDecoding C matrix -> (thread, element) mapping:\n");
    printf("C[row][col] = thread * 10 + element\n\n");
    printf("     ");
    for (int j = 0; j < N; j++) printf("  col%d ", j);
    printf("\n");
    for (int i = 0; i < M; i++) {
        printf("%2d: ", i);
        for (int j = 0; j < N; j++) {
            float val = h_C_mat[i * N + j];
            int thread = (int)val / 10;
            int elem = (int)val % 10;
            printf("t%d.%d  ", thread, elem);
        }
        printf("\n");
    }

    cudaFree(d_C_mat);
    cudaFree(d_C_frag);

    return 0;
}
