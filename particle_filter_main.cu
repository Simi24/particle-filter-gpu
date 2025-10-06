// particle_filter_main.cu
// Main program for optimized CUDA particle filter
// Integrates all components with multi-stream execution

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "particle_filter_config.h"

// GPU Timer structure
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GPUTimer;

// Forward declarations for external functions
extern void inclusive_scan(const float*, float*, int, cudaStream_t);
extern float reduce_sum(const float*, int, cudaStream_t);
extern float reduce_sum_squares(const float*, int, cudaStream_t);
extern void weighted_average(const Particle*, const float*, int, float*, float*, cudaStream_t);
extern void debug_simple_average(const Particle*, int, float*, float*, cudaStream_t);

// Forward declarations for kernel launches
extern "C" {
    __global__ void init_particles_kernel(Particle*, curandState*, int, float, float);
    __global__ void predict_kernel(Particle*, curandState*, int);
    __global__ void update_weights_kernel(const Particle*, float*, int, float, float);
    __global__ void normalize_weights_kernel(float*, float*, int, float);
    __global__ void resample_kernel(const Particle*, Particle*, const float*, int, float);
    __global__ void resample_optimized_kernel(const Particle*, Particle*, cudaTextureObject_t, int, float);
    __global__ void set_uniform_weights_kernel(float*, int, float);
}

// Utility functions
extern float randn_host(float, float);
extern FILE* safe_fopen(const char*, const char*);
extern void write_result(FILE*, const Result*);
extern void print_device_info();
extern void gpu_timer_create(GPUTimer*);
extern void gpu_timer_start(GPUTimer*);
extern float gpu_timer_stop(GPUTimer*);
extern void gpu_timer_destroy(GPUTimer*);
extern float compute_rmse(const float*, int);
extern float compute_mean(const float*, int);
extern float compute_stddev(const float*, int, float);

// Trajectory generation (also in kernels file)
extern void generate_trajectory(float, float*, float*);

/* =================================================== */
/* PARTICLE FILTER STATE STRUCTURE                     */
/* =================================================== */

typedef struct {
    // Device memory
    Particle* d_particles[2];      // Double buffer for resampling
    float* d_weights;
    float* d_cumulative_weights;
    curandState* d_rand_states;

    // Texture object for weights
    cudaTextureObject_t tex_weights_obj;

    // Pinned host memory for async transfers
    float* h_est_x_pinned;
    float* h_est_y_pinned;

    // CUDA streams for concurrent execution
    cudaStream_t streams[NUM_STREAMS];

    // Configuration
    int n_particles;
    int current_buffer;

    // Grid configuration
    int threads_per_block;
    int blocks_per_grid;
} ParticleFilterState;

/* =================================================== */
/* INITIALIZATION                                      */
/* =================================================== */

/**
 * Initializes the particle filter state
 * Allocates all GPU and pinned host memory
 * Creates CUDA streams for concurrent execution
 */
void initialize_particle_filter(ParticleFilterState* pf, int n_particles) {
    pf->n_particles = n_particles;
    pf->current_buffer = 0;
    pf->threads_per_block = THREADS_PER_BLOCK;
    pf->blocks_per_grid = GRID_SIZE(n_particles, THREADS_PER_BLOCK);
    
    printf("Initializing Particle Filter...\n");
    printf("Particles: %d\n", n_particles);
    printf("Blocks: %d, Threads per block: %d\n", 
           pf->blocks_per_grid, pf->threads_per_block);
    
    // Allocate device memory
    size_t particle_size = n_particles * sizeof(Particle);
    size_t float_size = n_particles * sizeof(float);
    size_t state_size = n_particles * sizeof(curandState);
    
    CUDA_CHECK(cudaMalloc(&pf->d_particles[0], particle_size));
    CUDA_CHECK(cudaMalloc(&pf->d_particles[1], particle_size));
    CUDA_CHECK(cudaMalloc(&pf->d_weights, float_size));
    CUDA_CHECK(cudaMalloc(&pf->d_cumulative_weights, float_size));
    CUDA_CHECK(cudaMalloc(&pf->d_rand_states, state_size));
    
    // Allocate pinned host memory for faster transfers
    CUDA_CHECK(cudaMallocHost(&pf->h_est_x_pinned, sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&pf->h_est_y_pinned, sizeof(float)));
    
    // Create CUDA streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&pf->streams[i]));
    }
    
    printf("Memory allocated successfully\n");
    printf("Total GPU memory used: %.2f MB\n",
           (2 * particle_size + 2 * float_size + state_size) / (1024.0 * 1024.0));
}

/**
 * Initializes particles around initial position
 */
void initialize_particles(ParticleFilterState* pf, float init_x, float init_y) {
    printf("Launching init_particles_kernel with %d blocks, %d threads\n", 
           pf->blocks_per_grid, pf->threads_per_block);
    
    init_particles_kernel<<<pf->blocks_per_grid, pf->threads_per_block>>>(
        pf->d_particles[0],
        pf->d_rand_states,
        pf->n_particles,
        init_x,
        init_y
    );
    
    // Check for kernel launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(launch_error));
        exit(EXIT_FAILURE);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check for kernel execution errors
    cudaError_t exec_error = cudaGetLastError();
    if (exec_error != cudaSuccess) {
        printf("ERROR: Kernel execution failed: %s\n", cudaGetErrorString(exec_error));
        exit(EXIT_FAILURE);
    }
    
    printf("Particles initialized at (%.2f, %.2f)\n", init_x, init_y);
    
    // Debug: Check first few particles
    Particle* h_particles = (Particle*)malloc(10 * sizeof(Particle));
    CUDA_CHECK(cudaMemcpy(h_particles, pf->d_particles[0], 10 * sizeof(Particle), cudaMemcpyDeviceToHost));
    printf("DEBUG: First 3 particles after init:\n");
    for (int i = 0; i < 3; i++) {
        printf("  Particle %d: x=%.6f, y=%.6f, vx=%.6f, vy=%.6f\n", 
               i, h_particles[i].x, h_particles[i].y, h_particles[i].vx, h_particles[i].vy);
    }
    free(h_particles);
}

/* =================================================== */
/* PARTICLE FILTER STEP                                */
/* =================================================== */

/**
 * Executes one complete particle filter iteration
 * Uses streams for overlapping computation where possible
 * 
 * @return: Estimated state (x, y) and error
 */
Result particle_filter_step(
    ParticleFilterState* pf,
    float time,
    float obs_x,
    float obs_y,
    float true_x,
    float true_y
) {
    // Select current particle buffer
    int curr_buf = pf->current_buffer;
    Particle* d_particles_curr = pf->d_particles[curr_buf];
    Particle* d_particles_next = pf->d_particles[1 - curr_buf];

    // Assign streams for different phases
    cudaStream_t predict_stream = pf->streams[0];
    cudaStream_t weight_stream = pf->streams[1];
    cudaStream_t scan_stream = pf->streams[2];
    cudaStream_t resample_stream = pf->streams[3];
    
    // ============ PREDICTION STEP ============
    predict_kernel<<<pf->blocks_per_grid, pf->threads_per_block, 0, predict_stream>>>(
        d_particles_curr,
        pf->d_rand_states,
        pf->n_particles
    );

    // Wait for prediction to complete before weight update (data dependency)
    CUDA_CHECK(cudaStreamSynchronize(predict_stream));
    
    // Debug: Check particles after prediction (only for first timestep)
    if (time < 0.01f) {  // Only for t=0
        Particle* h_particles = (Particle*)malloc(3 * sizeof(Particle));
        CUDA_CHECK(cudaMemcpy(h_particles, d_particles_curr, 3 * sizeof(Particle), cudaMemcpyDeviceToHost));
        printf("DEBUG: First 3 particles after prediction at t=%.1f:\n", time);
        for (int i = 0; i < 3; i++) {
            printf("  Particle %d: x=%.6f, y=%.6f, vx=%.6f, vy=%.6f\n", 
                   i, h_particles[i].x, h_particles[i].y, h_particles[i].vx, h_particles[i].vy);
        }
        free(h_particles);
    }
    
    // ============ UPDATE WEIGHTS ============
    update_weights_kernel<<<pf->blocks_per_grid, pf->threads_per_block, 0, weight_stream>>>(
        d_particles_curr,
        pf->d_weights,
        pf->n_particles,
        obs_x,
        obs_y
    );
    
    // ============ COMPUTE CUMULATIVE WEIGHTS ============
    // Wait for weight update to complete before scan
    CUDA_CHECK(cudaStreamSynchronize(weight_stream));

    // Perform inclusive scan (prefix sum) on weights
    inclusive_scan(
        pf->d_weights,
        pf->d_cumulative_weights,
        pf->n_particles,
        scan_stream
    );

    // Get total weight (last element of cumulative sum)
    float total_weight;
    CUDA_CHECK(cudaMemcpyAsync(
        &total_weight,
        pf->d_cumulative_weights + pf->n_particles - 1,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        scan_stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(scan_stream));
    
    // ============ NORMALIZE WEIGHTS ============
    normalize_weights_kernel<<<pf->blocks_per_grid, pf->threads_per_block, 0, scan_stream>>>(
        pf->d_weights,
        pf->d_cumulative_weights,
        pf->n_particles,
        total_weight
    );
    
    // ============ COMPUTE ESS (Effective Sample Size) ============
    // Can overlap with state estimation
    float sum_weights_squared = reduce_sum_squares(
        pf->d_weights,
        pf->n_particles,
        scan_stream
    );
    float ess = 1.0f / sum_weights_squared;
    
    // ============ RESAMPLING (if needed) ============
    float ess_threshold = pf->n_particles / 2.0f;
    if (ess < ess_threshold) {
        // Wait for normalization to complete before resampling
        CUDA_CHECK(cudaStreamSynchronize(scan_stream));

        // Create texture object for weight access
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = pf->d_cumulative_weights;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = pf->n_particles * sizeof(float);

        cudaTextureDesc texDesc = {};
        texDesc.readMode = cudaReadModeElementType;

        CUDA_CHECK(cudaCreateTextureObject(&pf->tex_weights_obj, &resDesc, &texDesc, NULL));

        // Generate random offset for systematic resampling
        float random_offset = ((float)rand() / RAND_MAX) / pf->n_particles;

        // Use optimized resampling kernel with shared memory
        resample_optimized_kernel<<<pf->blocks_per_grid, pf->threads_per_block, 0, resample_stream>>>(
            d_particles_curr,
            d_particles_next,
            pf->tex_weights_obj,
            pf->n_particles,
            random_offset
        );

        // Destroy texture object
        CUDA_CHECK(cudaDestroyTextureObject(pf->tex_weights_obj));

        // Swap buffers
        pf->current_buffer = 1 - curr_buf;

        // Reset weights to uniform after resampling
        float uniform_weight = 1.0f / pf->n_particles;
        set_uniform_weights_kernel<<<pf->blocks_per_grid, pf->threads_per_block, 0, resample_stream>>>(
            pf->d_weights, pf->n_particles, uniform_weight
        );
    }
    
    // ============ STATE ESTIMATION ============
    float est_x = -999.0f, est_y = -999.0f;  // Initialize to obvious wrong values to detect bugs

    // Use proper weighted average for state estimation
    // Can overlap with ESS calculation and resampling decision
    weighted_average(
        pf->d_particles[pf->current_buffer],
        pf->d_weights,
        pf->n_particles,
        &est_x,
        &est_y,
        resample_stream
    );
    
    // ============ COMPUTE ERROR ============
    float dx = est_x - true_x;
    float dy = est_y - true_y;
    float error = sqrtf(dx * dx + dy * dy);
    
    // Create result
    Result result;
    result.time = time;
    result.true_x = true_x;
    result.true_y = true_y;
    result.obs_x = obs_x;
    result.obs_y = obs_y;
    result.est_x = est_x;
    result.est_y = est_y;
    result.error = error;
    
    return result;
}

/* =================================================== */
/* CLEANUP                                             */
/* =================================================== */

void cleanup_particle_filter(ParticleFilterState* pf) {
    // Free device memory
    CUDA_CHECK(cudaFree(pf->d_particles[0]));
    CUDA_CHECK(cudaFree(pf->d_particles[1]));
    CUDA_CHECK(cudaFree(pf->d_weights));
    CUDA_CHECK(cudaFree(pf->d_cumulative_weights));
    CUDA_CHECK(cudaFree(pf->d_rand_states));
    
    // Free pinned host memory
    CUDA_CHECK(cudaFreeHost(pf->h_est_x_pinned));
    CUDA_CHECK(cudaFreeHost(pf->h_est_y_pinned));
    
    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(pf->streams[i]));
    }
}

/* =================================================== */
/* MAIN PROGRAM                                        */
/* =================================================== */

int main(int argc, char** argv) {
    // Seed random number generator
    srand(time(NULL));
    
    printf("\n");
    printf("============================================\n");
    printf("  OPTIMIZED CUDA PARTICLE FILTER\n");
    printf("============================================\n");
    
    // Print device information
    print_device_info();
    
    // Open output files
    FILE* results_file = safe_fopen("results_gpu_optimized.csv", "w");
    fprintf(results_file, "time,true_x,true_y,obs_x,obs_y,est_x,est_y,error\n");
    
    // Initialize particle filter
    ParticleFilterState pf;
    initialize_particle_filter(&pf, N_PARTICLES);
    
    // Get initial position from trajectory
    float init_x, init_y;
    generate_trajectory(0.0f, &init_x, &init_y);
    initialize_particles(&pf, init_x, init_y);
    
    // Start timing
    GPUTimer timer;
    gpu_timer_create(&timer);
    gpu_timer_start(&timer);
    
    printf("\n");
    printf("============================================\n");
    printf("  RUNNING SIMULATION\n");
    printf("============================================\n");
    printf("Timesteps: %d, dt: %.2f\n\n", N_TIMESTEPS, DT);
    
    // Arrays for statistics
    float* errors = (float*)malloc(N_TIMESTEPS * sizeof(float));
    
    // ============ MAIN SIMULATION LOOP ============
    for (int t = 0; t < N_TIMESTEPS; ++t) {
        float time = t * DT;
        
        // Generate true trajectory and noisy observation
        float true_x, true_y;
        generate_trajectory(time, &true_x, &true_y);
        
        float obs_x = true_x + randn_host(0.0f, MEASUREMENT_NOISE);
        float obs_y = true_y + randn_host(0.0f, MEASUREMENT_NOISE);
        
        // Execute particle filter step
        Result result = particle_filter_step(
            &pf, time, obs_x, obs_y, true_x, true_y
        );
        
        // Store error
        errors[t] = result.error;
        
        // Write result to file
        write_result(results_file, &result);
        
        // Print progress
        if (t % 10 == 0 || t == N_TIMESTEPS - 1) {
            printf("t=%3d: true=(%.2f,%.2f) obs=(%.2f,%.2f) est=(%.2f,%.2f) error=%.3f\n",
                   t, true_x, true_y, obs_x, obs_y, result.est_x, result.est_y, result.error);
        }
    }
    
    // Stop timing
    float total_time = gpu_timer_stop(&timer);
    gpu_timer_destroy(&timer);
    
    printf("\n");
    printf("============================================\n");
    printf("  SIMULATION COMPLETE\n");
    printf("============================================\n");
    
    // Compute and display statistics
    float mean_error = compute_mean(errors, N_TIMESTEPS);
    float rmse = compute_rmse(errors, N_TIMESTEPS);
    float stddev = compute_stddev(errors, N_TIMESTEPS, mean_error);
    
    printf("Performance Statistics:\n");
    printf("Total simulation time: %.3f seconds\n", total_time);
    printf("Average time per step: %.3f ms\n", 
           (total_time * 1000.0f) / N_TIMESTEPS);
    printf("Throughput: %.1f particles/ms\n",
           (N_PARTICLES * N_TIMESTEPS) / (total_time * 1000.0f));
    
    printf("\nAccuracy Statistics:\n");
    printf("Mean error: %.3f\n", mean_error);
    printf("RMSE: %.3f\n", rmse);
    printf("Standard deviation: %.3f\n", stddev);
    printf("Final error: %.3f\n", errors[N_TIMESTEPS - 1]);
    
    // Cleanup
    cleanup_particle_filter(&pf);
    fclose(results_file);
    free(errors);
    
    printf("\nResults saved to: results_gpu_optimized.csv\n");
    printf("============================================\n");
    
    return 0;
}