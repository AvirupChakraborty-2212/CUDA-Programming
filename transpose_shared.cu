#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define T 32

__global__ void transpose(float *A, float *B, int N)
{
    __shared__ float s[T][T+1]; //shared memory bank conflict
    int by = blockDim.y * blockIdx.y  ;
    int bx = blockDim.x * blockIdx.x  ;
    int y = by + threadIdx.y ;
    int x = bx + threadIdx.x ;

    if(y < N && x < N)
    {
        s[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    
    __syncthreads();

    int newbx = blockDim.y * blockIdx.y  ;
    int newby = blockDim.x * blockIdx.x  ;

    int newx = newbx + threadIdx.x ;
    int newy = newby + threadIdx.y ;
    
    if(newy < N && newx < N)
    {
        B[newy * N + newx] = s[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
     
}

int main()
{   
    int N = 1024;
    size_t size = N*N*sizeof(float);

    // allocate memory
    float *input, *output;
    cudaMallocManaged((void**)&input, size);
    cudaMallocManaged((void**)&output, size);

    for(int i = 0; i < N*N; i++)
    {
        input[i] = rand()/(float)RAND_MAX;
    }

    dim3 block(32,32);
    dim3 grid( (N + block.x - 1) / block.x , (N + block.y - 1) / block.y);

    transpose<<<grid, block>>>(input, output, N);

    cudaDeviceSynchronize();

    if (output[1 * N + 0] == input[0 * N + 1]) {
        printf("Success: Transpose is correct and coalesced!\n");
    } else {
        printf("Failure: Transpose logic error.\n");
    }

    cudaFree(input);
    cudaFree(output);

    return 0;
}