#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Funzione di utility per gestire gli errori CUDA
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore CUDA: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// __global__ indica una funzione (kernel) che viene eseguita sulla GPU
__global__ void sum_vectors(float *result, const float *a, const float *b, int n) {
    // Calcola l'indice globale univoco per ogni thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Assicura che il thread non acceda a memoria fuori dai limiti dell'array
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// Funzione main eseguita sulla CPU (host)
int main(void) {
    int N = 1000000; // Numero di elementi nei vettori
    size_t size = N * sizeof(float);

    // 1. Allocazione della memoria sulla CPU (host)
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        fprintf(stderr, "Errore malloc!\n");
        return 1;
    }

    // Inizializzazione dei dati sulla CPU
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)i * 2.0f;
    }

    // 2. Allocazione della memoria sulla GPU (device)
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, size), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, size), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, size), "cudaMalloc d_c");

    // 3. Copia dei dati dalla CPU alla GPU
    printf("Copia dei dati da Host a Device...\n");
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "cudaMemcpy h_a -> d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "cudaMemcpy h_b -> d_b");

    // 4. Configurazione ed esecuzione del kernel
    int threadsPerBlock = 256;
    // Calcola il numero di blocchi necessari per coprire tutti gli elementi
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Lancio del kernel CUDA...\n");
    sum_vectors<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, N);

    // Controlla eventuali errori durante il lancio del kernel
    checkCudaError(cudaGetLastError(), "Errore nel lancio del kernel");
    // Sincronizza per essere sicuri che il kernel sia terminato
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // 5. Copia del risultato dalla GPU alla CPU
    printf("Copia del risultato da Device a Host...\n");
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_c -> h_c");

    // 6. Verifica del risultato
    printf("Verifica del risultato (primi 5 elementi):\n");
    for (int i = 0; i < 5; i++) {
        printf("h_c[%d] = %f (atteso: %f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // 7. Liberazione della memoria
    free(h_a);
    free(h_b);
    free(h_c);
    checkCudaError(cudaFree(d_a), "cudaFree d_a");
    checkCudaError(cudaFree(d_b), "cudaFree d_b");
    checkCudaError(cudaFree(d_c), "cudaFree d_c");

    printf("Operazione completata con successo!\n");

    return 0;
}