// particle_filter_config.h
// Configuration and data structures for the particle filter

#ifndef PARTICLE_FILTER_CONFIG_H
#define PARTICLE_FILTER_CONFIG_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

/* =================================================== */
/* SIMULATION PARAMETERS                               */
/* =================================================== */
#define N_PARTICLES      100000000
#define N_TIMESTEPS      100
#define DT               0.1f
#define PROCESS_NOISE    0.3f
#define MEASUREMENT_NOISE 1.0f
#define INIT_NOISE       0.5f

/* =================================================== */
/* CONSTANT MEMORY DECLARATIONS                        */
/* =================================================== */
__constant__ float c_dt = DT;
__constant__ float c_process_noise = PROCESS_NOISE;
__constant__ float c_measurement_noise = MEASUREMENT_NOISE;
__constant__ int c_n_particles = N_PARTICLES;

/* =================================================== */
/* TEXTURE MEMORY DECLARATIONS                         */
/* =================================================== */
// Texture objects are now managed per-particle-filter instance

/* =================================================== */
/* CUDA OPTIMIZATION PARAMETERS                        */
/* =================================================== */
// Optimal thread count for modern GPUs (multiple of 32 for warp alignment)
#define THREADS_PER_BLOCK 256
#define WARP_SIZE        32

// Number of warps per block
#define WARPS_PER_BLOCK  (THREADS_PER_BLOCK / WARP_SIZE)

// For reduction kernels - must be power of 2
#define REDUCTION_THREADS 256

// Number of CUDA streams for concurrent execution
#define NUM_STREAMS      4

// Macro to compute grid size
#define GRID_SIZE(n, block_size) (((n) + (block_size) - 1) / (block_size))

/* =================================================== */
/* ERROR CHECKING MACRO                                */
/* =================================================== */
#define CUDA_CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

/* =================================================== */
/* DATA STRUCTURES                                     */
/* =================================================== */

// Particle structure - aligned for coalesced memory access
typedef struct __align__(16) {
    float x;   // Position x
    float y;   // Position y
    float vx;  // Velocity x
    float vy;  // Velocity y
} Particle;

// Structure for tracking results
typedef struct {
    float time;
    float true_x, true_y;
    float obs_x, obs_y;
    float est_x, est_y;
    float error;
} Result;

/* =================================================== */
/* INLINE DEVICE FUNCTIONS                             */
/* =================================================== */

// Fast square function
__device__ __forceinline__ float square(float x) {
    return x * x;
}

// Warp-level reduction using shuffle instructions (no shared memory)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif // PARTICLE_FILTER_CONFIG_H