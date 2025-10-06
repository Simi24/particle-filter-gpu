// utils.cu
// Utility functions for host-side operations

#include "particle_filter_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* =================================================== */
/* HOST RANDOM NUMBER GENERATION                       */
/* =================================================== */

/**
 * Box-Muller transform for generating Gaussian random numbers on host
 * Used for generating noisy observations
 */
float randn_host(float mean, float stddev) {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return mean + stddev * spare;
    }
    
    has_spare = 1;
    float u, v, s;
    
    do {
        u = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        v = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return mean + stddev * u * s;
}

/* =================================================== */
/* FILE I/O UTILITIES                                  */
/* =================================================== */

/**
 * Opens a file with error checking
 */
FILE* safe_fopen(const char* filename, const char* mode) {
    FILE* file = fopen(filename, mode);
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    return file;
}

/**
 * Writes result to CSV file
 */
void write_result(FILE* file, const Result* result) {
    fprintf(file, "%f,%f,%f,%f,%f,%f,%f,%f\n",
            result->time,
            result->true_x, result->true_y,
            result->obs_x, result->obs_y,
            result->est_x, result->est_y,
            result->error);
}



/* =================================================== */
/* DEVICE INFORMATION                                  */
/* =================================================== */

/**
 * Prints GPU device information
 * Useful for understanding performance characteristics
 */
void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("\n=== GPU Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Shared Memory per Block: %.2f KB\n", 
           prop.sharedMemPerBlock / 1024.0);
    printf("==============================\n\n");
}

/* =================================================== */
/* PERFORMANCE TIMING                                  */
/* =================================================== */

/**
 * GPU timer using CUDA events
 */
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GPUTimer;

void gpu_timer_create(GPUTimer* timer) {
    CUDA_CHECK(cudaEventCreate(&timer->start));
    CUDA_CHECK(cudaEventCreate(&timer->stop));
}

void gpu_timer_start(GPUTimer* timer) {
    CUDA_CHECK(cudaEventRecord(timer->start));
}

float gpu_timer_stop(GPUTimer* timer) {
    CUDA_CHECK(cudaEventRecord(timer->stop));
    CUDA_CHECK(cudaEventSynchronize(timer->stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, timer->start, timer->stop));
    return milliseconds / 1000.0f; // Convert to seconds
}

void gpu_timer_destroy(GPUTimer* timer) {
    CUDA_CHECK(cudaEventDestroy(timer->start));
    CUDA_CHECK(cudaEventDestroy(timer->stop));
}

/* =================================================== */
/* STATISTICS UTILITIES                                */
/* =================================================== */

/**
 * Computes RMSE from array of errors
 */
float compute_rmse(const float* errors, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += errors[i] * errors[i];
    }
    return sqrtf(sum / n);
}

/**
 * Computes mean of array
 */
float compute_mean(const float* values, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += values[i];
    }
    return sum / n;
}

/**
 * Computes standard deviation
 */
float compute_stddev(const float* values, int n, float mean) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq / n);
}