#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
// cuda kernel
__global__ void  matMul(const float *A, const float *B, float*C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (row < N && col < N)
    {
        float acc = 0.0f;
        for(int k = 0; k<N; k++)
        {
            acc += A[row*N + k] * B[k*N + col];
        }
        C[row * N + col] = acc;
    }
} 

// main function
int main()
{
    int N = 128;
    size_t size = N * N * sizeof(float);

    //allocate host
    float *h_A = (float *)malloc(size); //weak
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    //allocate device
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void**)&d_A, size); //weak
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    //init array
    for(int i = 0 ; i<N*N; i++)
    {        
        h_A[i]= rand() / (float)RAND_MAX; //weak
        h_B[i]= rand() / (float)RAND_MAX;        
    }
    //copy host to dev
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    //call cuda kernel
    dim3 threads_per_block(32,32);
    dim3 num_blocks ((N + threads_per_block.x-1)/threads_per_block.x, (N + threads_per_block.y-1)/threads_per_block.y);
    matMul<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    printf("test passed \n");
    //copy dev to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Done! Result at [0]: %f\n", h_C[0]);
    //free mem
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
