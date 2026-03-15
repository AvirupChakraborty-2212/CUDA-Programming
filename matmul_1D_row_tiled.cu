#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define ele 3
#define T 4

__global__ void matmul(float *A, float *B, float *C, int N)
{
    extern __shared__ float s[];
    float *sA = s;
    float *sB = s + T*T;
    float acc[ele];
    for(int e=0; e<ele; e++)
    {
        acc[e] = 0.0f;
    }
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int globalRow = blockDim.y * blockIdx.y + threadIdx.y;
    int baseGlobalCol = blockIdx.x * blockDim.x * ele;
    int num_tiles = (N + T - 1) / T;
    
    for(int i=0; i<num_tiles; i++)
    {
        int aRow = globalRow;
        int aCol = i * T + localCol;
        int bRow = i * T + localRow;

        if(aRow<N && aCol<N)
        {
            sA[localRow * T + localCol] = A[aRow * N + aCol];
        }
        else
        {
            sA[localRow * T + localCol] = 0.0f;
        }

        for(int e=0; e<ele; e++)
        {
            int globalCol = baseGlobalCol + localCol + e * T;
            int bCol = globalCol;
            
            if(bRow<N && bCol<N)
            {
                sB[localRow * T + localCol + e*T] = B[bRow * N + bCol];
            }
            else
            {
                sB[localRow * T + localCol + e*T] = 0.0f;
            }
        }

        __syncthreads();

        int limit = T;

        for(int i=0; i<limit; i++)
        {
            for(int e=0; e<ele; e++)
            {
                acc[e] += sA[localRow * T + i] * sB[i * T + localCol + e * T];
            }
        }

        __syncthreads();
        
    }

    for(int e=0; e<ele; e++)
        {
            int globalCol = baseGlobalCol + localCol + e * T;
            if(globalRow<N && globalCol<N)
            {
                C[globalRow * N + globalCol] = acc[e];
            }            
        }
}

int main()
{
    int N = 128;
    size_t size = N * N * sizeof(float);
    size_t shared_mem = ((T * T) + (T * T * ele)) * sizeof(float) ;

    //Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    //Initialize array
    for(int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }    

    //Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    //Copy host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //call cuda kernel
    dim3 threads_per_block(4,4);
    dim3 num_blocks ( (N+ (T*ele) -1) / (T*ele), (N+T-1) / T );

    matmul<<<num_blocks, threads_per_block, shared_mem>>>(d_A, d_B, d_C, N);

    //Copy device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Output at [0]: %f\n", h_C[0]);

    //Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}