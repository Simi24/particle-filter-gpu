// Temporary debug version of weighted_average that just computes unweighted average
#include <stdio.h>
#include <stdlib.h>
#include "particle_filter_config.h"

__global__ void simple_average_kernel(
    const Particle* __restrict__ particles,
    int n,
    float* __restrict__ out_x,
    float* __restrict__ out_y
) {
    __shared__ float sdata_x[256];
    __shared__ float sdata_y[256];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    
    // Each thread sums multiple elements
    while (i < n) {
        sum_x += particles[i].x;
        sum_y += particles[i].y;
        i += blockDim.x * gridDim.x;
    }
    
    sdata_x[tid] = sum_x;
    sdata_y[tid] = sum_y;
    __syncthreads();
    
    // Simple reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_x[tid] += sdata_x[tid + s];
            sdata_y[tid] += sdata_y[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out_x[blockIdx.x] = sdata_x[0];
        out_y[blockIdx.x] = sdata_y[0];
    }
}

void debug_simple_average(
    const Particle* d_particles,
    int n,
    float* est_x,
    float* est_y,
    cudaStream_t stream = 0
) {
    int blocks = min(256, (n + 255) / 256);
    
    float *d_block_x, *d_block_y;
    CUDA_CHECK(cudaMalloc(&d_block_x, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_y, blocks * sizeof(float)));
    
    simple_average_kernel<<<blocks, 256, 0, stream>>>(
        d_particles, n, d_block_x, d_block_y
    );
    
    // Sum the block results
    float total_x = 0, total_y = 0;
    float* h_blocks_x = (float*)malloc(blocks * sizeof(float));
    float* h_blocks_y = (float*)malloc(blocks * sizeof(float));
    
    CUDA_CHECK(cudaMemcpyAsync(h_blocks_x, d_block_x, blocks * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_blocks_y, d_block_y, blocks * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    for (int i = 0; i < blocks; i++) {
        total_x += h_blocks_x[i];
        total_y += h_blocks_y[i];
    }
    
    *est_x = total_x / n;
    *est_y = total_y / n;
    
    free(h_blocks_x);
    free(h_blocks_y);
    CUDA_CHECK(cudaFree(d_block_x));
    CUDA_CHECK(cudaFree(d_block_y));
}