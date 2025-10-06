// particle_filter_kernels.cu
// Core particle filter CUDA kernels with full optimizations

#include <stdio.h>
#include <stdlib.h>
#include "particle_filter_config.h"

/* =================================================== */
/* TRAJECTORY GENERATION                               */
/* =================================================== */

/**
 * Generates true trajectory (ground truth)
 * Piecewise motion model for realistic tracking scenario
 */
__host__ __device__ void generate_trajectory(
    float t,
    float* x,
    float* y
) {
    const float speed = 3.0f;
    
    if (t < 2.0f) {
        // Straight line motion
        *x = 0.0f + speed * t;
        *y = 0.0f;
    } else if (t < 4.0f) {
        // Accelerating turn
        float t_local = t - 2.0f;
        *x = 6.0f + speed * t_local;
        *y = 0.5f * t_local * t_local;
    } else if (t < 6.0f) {
        // Diagonal motion
        float t_local = t - 4.0f;
        *x = 12.0f + speed * 0.7f * t_local;
        *y = 2.0f + speed * 0.7f * t_local;
    } else if (t < 8.0f) {
        // Descending turn
        float t_local = t - 6.0f;
        *x = 16.2f + speed * 0.5f * t_local;
        *y = 10.8f - speed * 0.8f * t_local;
    } else {
        // Final straight segment
        float t_local = t - 8.0f;
        *x = 19.2f + speed * t_local;
        *y = 4.4f + speed * 0.2f * t_local;
    }
}

extern "C" {

/* =================================================== */
/* PARTICLE INITIALIZATION                             */
/* =================================================== */

/**
 * Initializes particles with random perturbations around initial state
 * Optimized for:
 * - Coalesced memory access (threads access consecutive particles)
 * - Efficient random number generation (cuRAND)
 * 
 * @param particles: Output particle array
 * @param rand_states: cuRAND state for each particle
 * @param n: Number of particles
 * @param init_x, init_y: Initial position
 */
__global__ void init_particles_kernel(
    Particle* __restrict__ particles,
    curandState* __restrict__ rand_states,
    int n,
    float init_x,
    float init_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Guard against out-of-bounds access
    if (idx >= n) return;
    
    // Initialize random state for this particle first
    curand_init(1234ULL + idx, 0, 0, &rand_states[idx]);
    
    // Initialize particle with small random noise around initial position
    float noise_x = curand_normal(&rand_states[idx]) * INIT_NOISE;
    float noise_y = curand_normal(&rand_states[idx]) * INIT_NOISE;
    float noise_vx = curand_normal(&rand_states[idx]) * 0.1f;  // Small velocity noise
    float noise_vy = curand_normal(&rand_states[idx]) * 0.1f;
    
    particles[idx].x = init_x + noise_x;
    particles[idx].y = init_y + noise_y;
    particles[idx].vx = 3.0f + noise_vx;  // Initial velocity around 3.0
    particles[idx].vy = 0.0f + noise_vy;  // Initial vy around 0.0
}

/* =================================================== */
/* PREDICTION STEP                                     */
/* =================================================== */

/**
 * Predicts particle states forward in time using motion model
 * Optimized for:
 * - Coalesced memory access
 * - Minimal register usage
 * - No warp divergence (all threads execute same path)
 * 
 * @param particles: Input/output particle array
 * @param rand_states: Random number generator states
 * @param n: Number of particles
 * @param dt: Time step
 * @param process_noise: Process noise standard deviation
 */
__global__ void predict_kernel(
    Particle* __restrict__ particles,
    curandState* __restrict__ rand_states,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Load random state to local memory for efficiency
    curandState local_state = rand_states[idx];

    // Load particle to registers
    Particle p = particles[idx];

    // Constant velocity model with Gaussian noise (using constant memory)
    float noise_scale = c_process_noise * c_dt;
    p.x += p.vx * c_dt + curand_normal(&local_state) * noise_scale;
    p.y += p.vy * c_dt + curand_normal(&local_state) * noise_scale;

    // Velocity random walk
    float vel_noise_scale = c_process_noise * 0.2f * c_dt;
    p.vx += curand_normal(&local_state) * vel_noise_scale;
    p.vy += curand_normal(&local_state) * vel_noise_scale;

    // Write back to global memory
    particles[idx] = p;
    rand_states[idx] = local_state;
}

/* =================================================== */
/* UPDATE WEIGHTS (MEASUREMENT UPDATE)                 */
/* =================================================== */

/**
 * Updates particle weights based on observation likelihood
 * Optimized for:
 * - Coalesced reads/writes
 * - Fast math operations (no divergence)
 * - Minimal memory transactions
 * 
 * @param particles: Input particle array
 * @param weights: Output weights array
 * @param n: Number of particles
 * @param obs_x, obs_y: Observed position
 * @param measurement_noise: Measurement noise standard deviation
 */
__global__ void update_weights_kernel(
    const Particle* __restrict__ particles,
    float* __restrict__ weights,
    int n,
    float obs_x,
    float obs_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Load particle position to registers
    float px = particles[idx].x;
    float py = particles[idx].y;

    // Compute squared distance to observation
    float dx = px - obs_x;
    float dy = py - obs_y;
    float dist_sq = dx * dx + dy * dy;

    // Gaussian likelihood with small epsilon to avoid zero weights (using constant memory)
    float variance = c_measurement_noise * c_measurement_noise;
    float likelihood = __expf(-dist_sq / (2.0f * variance)) + 1e-10f;

    // Store weight
    weights[idx] = likelihood;
}

/* =================================================== */
/* WEIGHT NORMALIZATION                                */
/* =================================================== */

/**
 * Normalizes weights and cumulative weights
 * Optimized for coalesced memory access
 * 
 * @param weights: Input/output normalized weights
 * @param cumulative_weights: Input/output normalized cumulative weights
 * @param n: Number of particles
 * @param total_weight: Sum of all weights
 */
__global__ void normalize_weights_kernel(
    float* __restrict__ weights,
    float* __restrict__ cumulative_weights,
    int n,
    float total_weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Avoid division by zero
    float norm_factor = (total_weight > 1e-10f) ? (1.0f / total_weight) : (1.0f / n);
    
    // Normalize both arrays in a single kernel launch
    weights[idx] *= norm_factor;
    cumulative_weights[idx] *= norm_factor;
}


/* =================================================== */
/* OPTIMIZED RESAMPLING WITH SHARED MEMORY             */
/* =================================================== */

/**
 * Advanced resampling kernel using shared memory caching
 * Further optimized for large particle counts
 * 
 * Loads cumulative weights into shared memory in blocks
 * to reduce global memory accesses during binary search
 */
__global__ void resample_optimized_kernel(
    const Particle* __restrict__ particles_in,
    Particle* __restrict__ particles_out,
    cudaTextureObject_t tex_weights_obj,
    int n,
    float random_offset
) {
    __shared__ float s_cum_weights[THREADS_PER_BLOCK + 1];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Compute systematic sample position
    float position = random_offset + (float)idx / n;
    
    // Collaborative loading of cumulative weights to shared memory
    // Each block loads a window of cumulative weights
    int block_start = blockIdx.x * blockDim.x;
    if (tid < blockDim.x && block_start + tid < n) {
        s_cum_weights[tid] = tex1Dfetch<float>(tex_weights_obj, block_start + tid);
    }
    if (tid == 0 && block_start + blockDim.x < n) {
        s_cum_weights[blockDim.x] = tex1Dfetch<float>(tex_weights_obj, block_start + blockDim.x);
    }
    __syncthreads();
    
    // Fast path: check if position is in current block's range
    int selected_idx;
    if (block_start > 0 && position < tex1Dfetch<float>(tex_weights_obj, block_start - 1)) {
        // Need to search in previous blocks - use binary search on global memory
        int left = 0;
        int right = block_start - 1;
        selected_idx = 0;

        while (left <= right) {
            int mid = (left + right) >> 1;
            if (tex1Dfetch<float>(tex_weights_obj, mid) < position) {
                left = mid + 1;
            } else {
                selected_idx = mid;
                right = mid - 1;
            }
        }
    } else if (block_start + blockDim.x < n &&
               position >= s_cum_weights[blockDim.x - 1]) {
        // Need to search in later blocks
        int left = block_start + blockDim.x;
        int right = n - 1;
        selected_idx = left;

        while (left <= right) {
            int mid = (left + right) >> 1;
            if (tex1Dfetch<float>(tex_weights_obj, mid) < position) {
                left = mid + 1;
            } else {
                selected_idx = mid;
                right = mid - 1;
            }
        }
    } else {
        // Position is in current block - search in shared memory
        int left = 0;
        int right = min(blockDim.x - 1, n - block_start - 1);
        int local_idx = 0;
        
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (s_cum_weights[mid] < position) {
                left = mid + 1;
            } else {
                local_idx = mid;
                right = mid - 1;
            }
        }
        selected_idx = block_start + local_idx;
    }
    
    // Copy selected particle (coalesced write)
    particles_out[idx] = particles_in[selected_idx];
}

/* =================================================== */
/* UTILITY KERNELS                                     */
/* =================================================== */

/**
 * Sets all weights to a uniform value
 * Used after resampling to reset particle weights
 */
__global__ void set_uniform_weights_kernel(
    float* __restrict__ weights,
    int n,
    float value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] = value;
    }
}

} // extern "C"