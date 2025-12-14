#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda::wmma;

__global__ void debug_dump_fragments(
    const float* A, const float* B,
    float* A_out, float* B_out,
    int K, int N
) {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;

    load_matrix_sync(a_frag, A, K);
    load_matrix_sync(b_frag, B, N);

    // Dump A fragment
    for (int i = 0; i < a_frag.num_elements; i++) {
        A_out[lane * a_frag.num_elements + i] = a_frag.x[i];
    }

    // Dump B fragment
    for (int i = 0; i < b_frag.num_elements; i++) {
        B_out[lane * b_frag.num_elements + i] = b_frag.x[i];
    }
}

int main() {
    const int M = 16, N = 16, K = 8;

    // Create simple test matrices with identifiable values
    // A[i][j] = i * 10 + j (row * 10 + col)
    // B[i][j] = i * 100 + j
    float h_A[M * K], h_B[K * N];

    printf("=== Input Matrices ===\n\n");
    printf("A (16x8) - A[row][col] = row*10 + col:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = i * 10 + j;
            printf("%5.0f ", h_A[i * K + j]);
        }
        printf("\n");
    }

    printf("\nB (8x16) - B[row][col] = row*100 + col:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = i * 100 + j;
            printf("%5.0f ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Allocate device memory
    float *d_A, *d_B, *d_A_out, *d_B_out;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_A_out, 32 * 4 * sizeof(float));  // 32 threads * 4 elements
    cudaMalloc(&d_B_out, 32 * 4 * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    debug_dump_fragments<<<1, 32>>>(d_A, d_B, d_A_out, d_B_out, K, N);
    cudaDeviceSynchronize();

    // Copy back
    float h_A_out[32 * 4], h_B_out[32 * 4];
    cudaMemcpy(h_A_out, d_A_out, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_out, d_B_out, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== WMMA Fragment Mapping ===\n\n");
    printf("A fragment (16x8 matrix_a, row_major):\n");
    printf("Thread | a[0]       | a[1]       | a[2]       | a[3]       | Decoded positions\n");
    printf("-------|------------|------------|------------|------------|-----------------\n");
    for (int t = 0; t < 32; t++) {
        printf("  %2d   |", t);
        for (int i = 0; i < 4; i++) {
            printf(" %10.0f |", h_A_out[t * 4 + i]);
        }
        // Decode: value = row*10 + col
        printf(" ");
        for (int i = 0; i < 4; i++) {
            int val = (int)h_A_out[t * 4 + i];
            int row = val / 10;
            int col = val % 10;
            printf("A[%d][%d] ", row, col);
        }
        printf("\n");
    }

    printf("\nB fragment (8x16 matrix_b, row_major):\n");
    printf("Thread | b[0]       | b[1]       | b[2]       | b[3]       | Decoded positions\n");
    printf("-------|------------|------------|------------|------------|-----------------\n");
    for (int t = 0; t < 32; t++) {
        printf("  %2d   |", t);
        for (int i = 0; i < 4; i++) {
            printf(" %10.0f |", h_B_out[t * 4 + i]);
        }
        // Decode: value = row*100 + col
        printf(" ");
        for (int i = 0; i < 4; i++) {
            int val = (int)h_B_out[t * 4 + i];
            int row = val / 100;
            int col = val % 100;
            printf("B[%d][%d] ", row, col);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_A_out);
    cudaFree(d_B_out);

    return 0;
}
