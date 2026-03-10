#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// matmul kernel coalaced

__global__ void matmul(float *X, float *W, float *Out, int B, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row<B && col<N)
    {
        float acc = 0.0f;
        for(int k=0; k<N; k++)
        {
            acc += X[row * N + k] * W[k * N + col];
        }

        Out[row * N + col] = acc;
    }
}

// relu kernel

__global__ void relu(float *Y, int B, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * N + col;

    if(row < B && col < N)
    {
        Y[idx] = fmaxf(0.0f, Y[idx]);
    }
}

int main()
{
    // size
    int B = 32;
    int N = 32;
    int S = 4;
    int o_s = B/S;
    size_t size_x = B * N * sizeof(float);
    size_t size_w = N * N * sizeof(float);

    // allocate host memory
    float *h_X = (float *)malloc(size_x);
    float *h_W1 = (float *)malloc(size_w);
    float *h_W2 = (float *)malloc(size_w);
    float *h_Z = (float *)malloc(size_x);

    // initialize arrays 
    for(int i=0; i< N*N; i++)
    {
        h_W1[i] = rand()/(float)RAND_MAX;
        h_W2[i] = rand()/(float)RAND_MAX;
    }

    for(int i=0; i< B*N; i++)
    {
        h_X[i] = rand()/(float)RAND_MAX;        
    }

    // initialize device memory
    float *d_X, *d_W1, *d_W2, *d_Z, *d_Y;
    cudaMalloc((void **)&d_X, size_x); 
    cudaMalloc((void **)&d_W1, size_w);
    cudaMalloc((void **)&d_W2, size_w);
    cudaMalloc((void **)&d_Y, size_x);
    cudaMalloc((void **)&d_Z, size_x);

    // copy host to device
    cudaMemcpy(d_X, h_X, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, size_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, size_w, cudaMemcpyHostToDevice);

    // streams
    cudaStream_t streams[4];
    for(int i=0; i<S; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);


    // launch kernel

    cudaEventRecord(start);

    dim3 block(32,32);
    dim3 grid( (N + 31 -1) / 32, (o_s + 31 -1) / 32);

    for(int i=0; i<S; i++)
    {
        int offset = i * o_s * N;
        matmul<<<grid, block, 0, streams[i]>>>( d_X + offset, d_W1, d_Y + offset, o_s, N );
        relu<<<grid, block, 0, streams[i]>>>( d_Y + offset, B, N );
        matmul<<<grid, block, 0, streams[i]>>>( d_Y + offset, d_W2, d_Z + offset, o_s, N );
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Elapsed time: %f ms\n", ms);

    // copy device to host
    cudaMemcpy(h_Z, d_Z, size_x, cudaMemcpyDeviceToHost);

    // free memory
    for(int i=0; i<S; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_X);cudaFree(d_W1);cudaFree(d_W2);cudaFree(d_Y);cudaFree(d_Z);
    free(h_X);free(h_W1);free(h_W2);free(h_Z);


    return 0;
}