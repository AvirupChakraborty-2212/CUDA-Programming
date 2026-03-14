#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define T 4


__global__ void matmul(float *A, float *B, float*C, int N)
{
    extern __shared__ float s[];
    float *sA = s;    
    
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (N+T-1)/T;

    for(int i=0; i<num_tiles; i++)

    {
        int aRow = globalRow;
        int aCol = i*T+localCol;        
        int bCol = globalCol;

        if(aRow < N && aCol < N)
        {
            sA[localRow * T + localCol] = A[aRow * N + aCol];
        }
        else
        {
            sA[localRow * T + localCol] = 0.0f;
        }

        __syncthreads();

        int limit = T;        

        for(int k=0; k<limit; k++)
        {
            int bRow = i*T+k;
            float valA = sA[localRow * T + k];
            float valB;
            if(bRow<N && bCol<N)
            {
                valB = B[bRow * N + bCol];
            } 
            else
            {
                valB = 0.0f;
            }               
            acc += valA * valB;
        }

        __syncthreads();
    }    
    
    if(globalRow < N && globalCol < N)
    {
        C[globalRow * N + globalCol] = acc;
    }
     
}

int main()
{   
    int N = 128;
    size_t size = N*N*sizeof(float);
    size_t shared_size = T*T*sizeof(float);

    // allocate memory host
    float *h_A, *h_B, *h_C;
    cudaMallocHost((void**)&h_A, size);
    cudaMallocHost((void**)&h_B, size);
    cudaMallocHost((void**)&h_C, size);

    // initialize host array
    for(int i = 0; i < N*N; i++)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // allocate memory device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // initialize and launch kernel
    dim3 block(4,4);
    dim3 grid( (N + block.x - 1) / block.x , (N + block.y - 1) / block.y);

    matmul<<<grid, block, shared_size>>>(d_A, d_B, d_C, N);

    // copy device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print
    printf("result at 0 : %f\n", h_C[0]);

    // free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}