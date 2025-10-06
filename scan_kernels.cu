// scan_kernels.cu
// Optimized implementation of parallel prefix sum (scan) to replace Thrust
// Uses Blelloch scan algorithm with work-efficient approach

#include <stdio.h>
#include <stdlib.h>
#include "particle_filter_config.h"

/* =================================================== */
/* INCLUSIVE SCAN (PREFIX SUM) IMPLEMENTATION          */
/* =================================================== */

/**
 * Simple inclusive scan using shared memory - more reliable implementation
 * Based on the classic reduction pattern
 */
__device__ void block_inclusive_scan_simple(float* sdata, int tid, int n) {
    // First, copy to shared memory if needed (assume already done)
    __syncthreads();
    
    // Perform inclusive scan
    for (int stride = 1; stride < n; stride *= 2) {
        __syncthreads();
        if (tid >= stride && tid < n) {
            sdata[tid] += sdata[tid - stride];
        }
    }
    __syncthreads();
}

/**
 * Kernel 1: Performs scan on blocks and outputs block sums
 * Each block processes THREADS_PER_BLOCK * 2 elements
 */
__global__ void scan_blocks_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ block_sums,
    int n
) {
    __shared__ float temp[THREADS_PER_BLOCK * 2];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * THREADS_PER_BLOCK * 2;
    
    // Load data into shared memory with bounds checking
    int idx1 = block_offset + tid;
    int idx2 = block_offset + tid + THREADS_PER_BLOCK;
    
    temp[tid] = (idx1 < n) ? input[idx1] : 0.0f;
    temp[tid + THREADS_PER_BLOCK] = (idx2 < n) ? input[idx2] : 0.0f;
    
    __syncthreads();
    
    // Determine actual number of elements to scan in this block
    int elements_in_block = min(THREADS_PER_BLOCK * 2, n - block_offset);
    if (elements_in_block <= 0) return;
    
    // Perform block-level inclusive scan
    block_inclusive_scan_simple(temp, tid, elements_in_block);
    
    // Write results back to global memory with bounds checking
    if (idx1 < n) output[idx1] = temp[tid];
    if (idx2 < n) output[idx2] = temp[tid + THREADS_PER_BLOCK];
    
    // Last thread in block writes the block sum
    if (tid == 0 && block_sums != NULL) {
        int last_idx = min(THREADS_PER_BLOCK * 2 - 1, elements_in_block - 1);
        if (last_idx >= 0) {
            block_sums[blockIdx.x] = temp[last_idx];
        }
    }
}

/**
 * Kernel 2: Adds scanned block sums to each block's elements
 * Avoids warp divergence by having all threads in a block work together
 */
__global__ void add_block_sums_kernel(
    float* __restrict__ data,
    const float* __restrict__ block_sums,
    int n
) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * THREADS_PER_BLOCK * 2;
    
    // Check if this block has any valid elements
    if (block_offset >= n) return;
    
    // Load the cumulative sum from previous blocks
    float block_sum = (blockIdx.x > 0) ? block_sums[blockIdx.x - 1] : 0.0f;
    
    // Add to both elements this thread is responsible for
    int idx1 = block_offset + tid;
    int idx2 = block_offset + tid + THREADS_PER_BLOCK;
    
    if (idx1 < n) data[idx1] += block_sum;
    if (idx2 < n) data[idx2] += block_sum;
}

/**
 * Host function: Performs inclusive scan on device array
 * Replaces thrust::inclusive_scan
 * 
 * @param d_input: Device input array
 * @param d_output: Device output array
 * @param n: Number of elements
 * @param stream: CUDA stream for async execution
 */
void inclusive_scan(
    const float* d_input,
    float* d_output,
    int n,
    cudaStream_t stream = 0
) {
    if (n <= 0) return;
    
    // Calculate grid dimensions
    int elements_per_block = THREADS_PER_BLOCK * 2;
    int num_blocks = GRID_SIZE(n, elements_per_block);
    
    if (num_blocks == 1) {
        // Single block - no need for second pass
        scan_blocks_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
            d_input, d_output, NULL, n
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    
    // Allocate temporary storage for block sums
    float* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));
    
    // Phase 1: Scan each block
    scan_blocks_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_input, d_output, d_block_sums, n
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Phase 2: Scan the block sums recursively if needed
    if (num_blocks > 1) {
        float* d_scanned_block_sums;
        CUDA_CHECK(cudaMalloc(&d_scanned_block_sums, num_blocks * sizeof(float)));
        
        // Recursively scan block sums
        inclusive_scan(d_block_sums, d_scanned_block_sums, num_blocks, stream);
        
        // Phase 3: Add scanned block sums to all blocks
        add_block_sums_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_output, d_scanned_block_sums, n
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        CUDA_CHECK(cudaFree(d_scanned_block_sums));
    }
    
    CUDA_CHECK(cudaFree(d_block_sums));
}
