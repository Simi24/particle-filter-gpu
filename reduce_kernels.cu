// reduce_kernels.cu
// Optimized reduction kernels to replace Thrust reduce operations
// Uses warp shuffle, shared memory, and unwrapping for maximum performance

#include <stdio.h>
#include <stdlib.h>
#include "particle_filter_config.h"

/* =================================================== */
/* REDUCTION KERNELS                                   */
/* =================================================== */

/**
 * Optimized reduction kernel using:
 * - Warp shuffle instructions for final warp reduction (no shared mem needed)
 * - Sequential addressing to avoid bank conflicts
 * - Loop unrolling for better instruction-level parallelism
 * - Each thread processes multiple elements to reduce kernel launches
 * 
 * Template allows for different operations (sum, max, etc.)
 */
template<typename T, typename Op>
__global__ void reduce_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    T identity
) {
    extern __shared__ T sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int grid_size = blockDim.x * 2 * gridDim.x;
    
    T sum = identity;
    
    // Grid-stride loop: each thread accumulates multiple elements
    // This reduces the number of blocks needed
    while (i < n) {
        sum = Op()(sum, input[i]);
        if (i + blockDim.x < n) {
            sum = Op()(sum, input[i + blockDim.x]);
        }
        i += grid_size;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory with sequential addressing
    // Unrolled for blocks of 256 threads
    if (blockDim.x >= 512) {
        if (tid < 256) { sdata[tid] = Op()(sdata[tid], sdata[tid + 256]); }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) { sdata[tid] = Op()(sdata[tid], sdata[tid + 128]); }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) { sdata[tid] = Op()(sdata[tid], sdata[tid + 64]); }
        __syncthreads();
    }
    
    // Final warp reduction using shuffle instructions (no __syncthreads needed)
    if (tid < 32) {
        volatile T* smem = sdata;
        if (blockDim.x >= 64) smem[tid] = Op()(smem[tid], smem[tid + 32]);
        
        // Last warp uses shuffle instructions - much faster than shared memory
        T val = smem[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            T other = __shfl_down_sync(0xffffffff, val, offset);
            val = Op()(val, other);
        }
        
        if (tid == 0) output[blockIdx.x] = val;
    }
}

/* =================================================== */
/* OPERATION FUNCTORS                                  */
/* =================================================== */

// Functor for sum operation
struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a + b;
    }
};

// Functor for sum of squares operation (for ESS calculation)
struct SumSquaresOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a + b * b;
    }
};

/* =================================================== */
/* HOST INTERFACE FUNCTIONS                            */
/* =================================================== */

/**
 * Performs sum reduction on device array
 * Replaces thrust::reduce
 * 
 * @param d_input: Device input array
 * @param n: Number of elements
 * @param stream: CUDA stream for async execution
 * @return: Sum of all elements
 */
float reduce_sum(const float* d_input, int n, cudaStream_t stream = 0) {
    // Calculate optimal grid size
    // Use fewer blocks but more work per thread for better efficiency
    int threads = REDUCTION_THREADS;
    int blocks = min(256, GRID_SIZE(n, threads * 2));
    
    // Allocate temporary storage for block results
    float* d_block_results;
    CUDA_CHECK(cudaMalloc(&d_block_results, blocks * sizeof(float)));
    
    // First reduction
    reduce_kernel<float, SumOp><<<blocks, threads, threads * sizeof(float), stream>>>(
        d_input, d_block_results, n, 0.0f
    );
    
    float result;
    if (blocks == 1) {
        // Single block - copy result directly
        CUDA_CHECK(cudaMemcpyAsync(&result, d_block_results, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        // Multiple blocks - need second reduction
        float* d_final_result;
        CUDA_CHECK(cudaMalloc(&d_final_result, sizeof(float)));
        
        reduce_kernel<float, SumOp><<<1, threads, threads * sizeof(float), stream>>>(
            d_block_results, d_final_result, blocks, 0.0f
        );
        
        CUDA_CHECK(cudaMemcpyAsync(&result, d_final_result, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_final_result));
    }
    
    CUDA_CHECK(cudaFree(d_block_results));
    return result;
}

/**
 * Performs sum of squares reduction
 * Used for ESS (Effective Sample Size) calculation
 * Replaces thrust::transform_reduce with square operation
 * 
 * @param d_input: Device input array
 * @param n: Number of elements
 * @param stream: CUDA stream for async execution
 * @return: Sum of squares of all elements
 */
float reduce_sum_squares(const float* d_input, int n, cudaStream_t stream = 0) {
    int threads = REDUCTION_THREADS;
    int blocks = min(256, GRID_SIZE(n, threads * 2));
    
    float* d_block_results;
    CUDA_CHECK(cudaMalloc(&d_block_results, blocks * sizeof(float)));
    
    reduce_kernel<float, SumSquaresOp><<<blocks, threads, threads * sizeof(float), stream>>>(
        d_input, d_block_results, n, 0.0f
    );
    
    float result;
    if (blocks == 1) {
        CUDA_CHECK(cudaMemcpyAsync(&result, d_block_results, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        float* d_final_result;
        CUDA_CHECK(cudaMalloc(&d_final_result, sizeof(float)));
        
        reduce_kernel<float, SumOp><<<1, threads, threads * sizeof(float), stream>>>(
            d_block_results, d_final_result, blocks, 0.0f
        );
        
        CUDA_CHECK(cudaMemcpyAsync(&result, d_final_result, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_final_result));
    }
    
    CUDA_CHECK(cudaFree(d_block_results));
    return result;
}

/* =================================================== */
/* WEIGHTED REDUCTION FOR STATE ESTIMATION             */
/* =================================================== */

/**
 * Optimized kernel for computing weighted average in a single pass
 * Reduces memory bandwidth requirements
 */
__global__ void weighted_reduce_kernel(
    const Particle* __restrict__ particles,
    const float* __restrict__ weights,
    int n,
    float* __restrict__ out_x,
    float* __restrict__ out_y
) {
    __shared__ float sdata_x[REDUCTION_THREADS];
    __shared__ float sdata_y[REDUCTION_THREADS];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int grid_size = blockDim.x * 2 * gridDim.x;
    
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    
    // Grid-stride loop for coalesced memory access
    while (i < n) {
        float w = weights[i];
        sum_x += particles[i].x * w;
        sum_y += particles[i].y * w;
        
        if (i + blockDim.x < n) {
            float w2 = weights[i + blockDim.x];
            sum_x += particles[i + blockDim.x].x * w2;
            sum_y += particles[i + blockDim.x].y * w2;
        }
        i += grid_size;
    }
    
    sdata_x[tid] = sum_x;
    sdata_y[tid] = sum_y;
    __syncthreads();
    
    // Reduction with unrolling
    if (blockDim.x >= 512) {
        if (tid < 256) {
            sdata_x[tid] += sdata_x[tid + 256];
            sdata_y[tid] += sdata_y[tid + 256];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            sdata_x[tid] += sdata_x[tid + 128];
            sdata_y[tid] += sdata_y[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) {
            sdata_x[tid] += sdata_x[tid + 64];
            sdata_y[tid] += sdata_y[tid + 64];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) {
        volatile float* smem_x = sdata_x;
        volatile float* smem_y = sdata_y;
        if (blockDim.x >= 64) {
            smem_x[tid] += smem_x[tid + 32];
            smem_y[tid] += smem_y[tid + 32];
        }
        
        float val_x = smem_x[tid];
        float val_y = smem_y[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_x += __shfl_down_sync(0xffffffff, val_x, offset);
            val_y += __shfl_down_sync(0xffffffff, val_y, offset);
        }
        
        if (tid == 0) {
            out_x[blockIdx.x] = val_x;
            out_y[blockIdx.x] = val_y;
        }
    }
}

/**
 * Computes weighted average of particle positions
 * Returns result through output parameters
 */
void weighted_average(
    const Particle* d_particles,
    const float* d_weights,
    int n,
    float* est_x,
    float* est_y,
    cudaStream_t stream = 0
) {
    int threads = REDUCTION_THREADS;
    int blocks = min(256, GRID_SIZE(n, threads * 2));
    
    float *d_block_x, *d_block_y;
    CUDA_CHECK(cudaMalloc(&d_block_x, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_y, blocks * sizeof(float)));
    
    weighted_reduce_kernel<<<blocks, threads, 0, stream>>>(
        d_particles, d_weights, n, d_block_x, d_block_y
    );
    
    if (blocks == 1) {
        CUDA_CHECK(cudaMemcpyAsync(est_x, d_block_x, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(est_y, d_block_y, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        // Perform final reduction on the block results
        float *d_final_x, *d_final_y;
        CUDA_CHECK(cudaMalloc(&d_final_x, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_final_y, sizeof(float)));
        
        // Final reduction for x
        reduce_kernel<float, SumOp><<<1, threads, threads * sizeof(float), stream>>>(
            d_block_x, d_final_x, blocks, 0.0f
        );
        
        // Final reduction for y  
        reduce_kernel<float, SumOp><<<1, threads, threads * sizeof(float), stream>>>(
            d_block_y, d_final_y, blocks, 0.0f
        );
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpyAsync(est_x, d_final_x, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(est_y, d_final_y, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        CUDA_CHECK(cudaFree(d_final_x));
        CUDA_CHECK(cudaFree(d_final_y));
    }
    
    CUDA_CHECK(cudaFree(d_block_x));
    CUDA_CHECK(cudaFree(d_block_y));
}