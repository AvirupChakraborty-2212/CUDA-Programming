#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
 // Keep the standard header

int main()
{
    int N = 128;
    size_t size = N * N * sizeof(float);

    // 1. Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 2. Initialize array
    for(int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }    

    // 3. Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 4. Copy host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 5. Initialize cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle); // Address of handle to initialize

    // 6. Set parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 7. Call SGEMM
    // Using the trick: (A*B)^T = B^T * A^T to handle Row-Major in Col-Major cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);

    // 8. Copy device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Output at [0]: %f\n", h_C[0]);

    // 9. Cleanup
    cublasDestroy(handle); // FIXED: No '&' here
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}