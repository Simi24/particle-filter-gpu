// test_simple_kernel.cu
// Simple test to verify basic CUDA kernel execution

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void simple_test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (float)idx + 42.0f;
    }
}

int main() {
    const int n = 1000;
    const int bytes = n * sizeof(float);
    
    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Initialize to zeros
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    
    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    printf("Launching kernel with %d blocks, %d threads\n", blocks, threads);
    
    simple_test_kernel<<<blocks, threads>>>(d_data, n);
    
    // Check for errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(launch_error));
        return 1;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaError_t exec_error = cudaGetLastError();
    if (exec_error != cudaSuccess) {
        printf("ERROR: Kernel execution failed: %s\n", cudaGetErrorString(exec_error));
        return 1;
    }
    
    // Copy back and check results
    float* h_data = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n");
    
    // Check if kernel worked
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_data[i] != (float)i + 42.0f) {
            success = false;
            break;
        }
    }
    
    printf("Test %s\n", success ? "PASSED" : "FAILED");
    
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    
    return success ? 0 : 1;
}