/**
 * FP8 Block-Scale MMA Implementation for SM120
 *
 * This file provides the implementation that will be compiled as part of
 * the native module when building for SM120.
 */

#include "fp8_block_scale_mma_sm120.cuh"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>

// Require CUDA 13.x for SM120 support
#if __CUDACC_VER_MAJOR__ >= 13

// The kernels are defined in the header file (fp8_block_scale_mma_sm120.cuh)
// This file just ensures the header is compiled into the native module

#endif  // __CUDACC_VER_MAJOR__ >= 13

// Host-callable functions (always compiled)
extern "C" {

/**
 * Check if FP8 block-scale MMA is available (SM120+)
 */
bool pygpukit_fp8_block_scale_mma_available() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return false;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return false;

    // SM120+ required
    return (prop.major >= 12);
}

/**
 * Get device compute capability
 */
int pygpukit_get_sm_version() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return 0;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return 0;

    return prop.major * 10 + prop.minor;
}

}  // extern "C"

// Test kernel for FP8 block-scale MMA (requires CUDA 13.x)
#if __CUDACC_VER_MAJOR__ >= 13

__global__ void fp8_block_scale_mma_test_kernel(
    const uint32_t* __restrict__ A_packed,  // [num_warps, 4] A fragments
    const uint32_t* __restrict__ B_packed,  // [num_warps, 2] B fragments
    float* __restrict__ D,                   // [num_warps, 4] output
    uint8_t scale_a,
    uint8_t scale_b
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Load fragments for this warp (simplified - in real code would be from smem)
    // Note: This is a simplified test - real usage needs proper fragment loading

    uint32_t a[4];
    uint32_t b[2];
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d[4];

    // For testing, all threads in warp use same data
    a[0] = A_packed[warp_id * 4 + 0];
    a[1] = A_packed[warp_id * 4 + 1];
    a[2] = A_packed[warp_id * 4 + 2];
    a[3] = A_packed[warp_id * 4 + 3];

    b[0] = B_packed[warp_id * 2 + 0];
    b[1] = B_packed[warp_id * 2 + 1];

    // Execute MMA (the function has internal __CUDA_ARCH__ check)
    pygpukit::ops::matmul::fp8_mma_sm120::mma_fp8_block_scale_16x8x32(
        d[0], d[1], d[2], d[3],
        a[0], a[1], a[2], a[3],
        b[0], b[1],
        c[0], c[1], c[2], c[3],
        scale_a, scale_b
    );

    // Store output for lane 0 (for verification)
    if (lane_id == 0) {
        D[warp_id * 4 + 0] = d[0];
        D[warp_id * 4 + 1] = d[1];
        D[warp_id * 4 + 2] = d[2];
        D[warp_id * 4 + 3] = d[3];
    }
}

#endif  // __CUDACC_VER_MAJOR__ >= 13

extern "C" {

/**
 * Run a simple test of the FP8 block-scale MMA instruction.
 * Returns 0 on success, non-zero on failure.
 */
int pygpukit_fp8_block_scale_mma_test() {
#if __CUDACC_VER_MAJOR__ >= 13
    int sm = pygpukit_get_sm_version();
    if (sm < 120) {
        printf("FP8 block-scale MMA test: SM%d not supported (need SM120+)\n", sm);
        return -1;
    }

    printf("FP8 block-scale MMA test on SM%d\n", sm);

    // Allocate test data
    uint32_t* d_A;
    uint32_t* d_B;
    float* d_D;

    cudaError_t err;
    err = cudaMalloc(&d_A, 4 * sizeof(uint32_t));
    if (err != cudaSuccess) return -2;
    err = cudaMalloc(&d_B, 2 * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_A); return -2; }
    err = cudaMalloc(&d_D, 4 * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return -2; }

    // Initialize with simple test values
    // A: 16 FP8 values = 0x01, 0x02, ..., 0x10 (small positive values)
    // B: 8 FP8 values = 0x01, 0x02, ..., 0x08
    uint32_t h_A[4] = {0x04030201, 0x08070605, 0x0C0B0A09, 0x100F0E0D};
    uint32_t h_B[2] = {0x04030201, 0x08070605};

    cudaMemcpy(d_A, h_A, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // UE8M0 scale = 127 means scale factor = 1.0 (2^(127-127) = 2^0 = 1)
    uint8_t scale_a = 127;
    uint8_t scale_b = 127;

    // Launch test kernel (declared above with __CUDACC_VER_MAJOR__ >= 13 guard)
    fp8_block_scale_mma_test_kernel<<<1, 32>>>(d_A, d_B, d_D, scale_a, scale_b);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_D);
        return -3;
    }

    // Read results
    float h_D[4];
    cudaMemcpy(h_D, d_D, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("MMA output: [%.4f, %.4f, %.4f, %.4f]\n", h_D[0], h_D[1], h_D[2], h_D[3]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);

    return 0;
#else
    printf("FP8 block-scale MMA test requires CUDA 13.x\n");
    return -4;
#endif
}

}  // extern "C"
