#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
// cuda kernel
__global__ void  matMul(const float *A, const float *B, float*C, int M, int N, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    

    if (row < M && col < N)
    {
        float acc = 0.0f;
        for(int k = 0; k<K; k++)
        {
            acc += A[row*K + k] * B[k*N + col];
        }
        C[row * N + col] = acc;
    }
} 

// main function
int main()
{
    int M = 64;
    int N = 128;
    int K = 32;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * N * sizeof(float);


    //allocate host
    float *h_A = (float *)malloc(sizeA); //weak
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    //allocate device
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void**)&d_A, sizeA); //weak
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);
    //init array
    for(int i = 0 ; i<M*K; i++)
    {        
        h_A[i]= rand() / (float)RAND_MAX; //weak
            
    }
    for(int i = 0 ; i<K*N; i++)
    {        
        h_B[i]= rand() / (float)RAND_MAX; //weak
            
    }
    //copy host to dev
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    //call cuda kernel
    dim3 threads_per_block(32,32);
    dim3 num_blocks ((M + threads_per_block.x-1)/threads_per_block.x, (N + threads_per_block.y-1)/threads_per_block.y);
    matMul<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N ,K);
    printf("test passed \n");
    //copy dev to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

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
