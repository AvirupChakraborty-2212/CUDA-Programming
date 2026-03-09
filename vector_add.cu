#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// cuda kernel
__global__ void  vecAdd(const float *A, const float *B, float*C, int num_ele)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_ele)
    {
        C[idx] = A[idx]+B[idx];
    }
} 

// main function
int main()
{

int num_ele = 50;
size_t size = num_ele * sizeof(float);

// allocate host memory
float *h_A = (float *)malloc(size);
float *h_B = (float *)malloc(size);
float *h_C = (float *)malloc(size);

// initialize

for(int i=0; i<num_ele; i++){
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
}


// allocate device memory
float *d_A = NULL;
float *d_B = NULL;
float *d_C = NULL;
cudaMalloc((void**)&d_A, size);
cudaMalloc((void**)&d_B, size);
cudaMalloc((void**)&d_C, size);

// copy host to device

cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// call cuda kernel

int threads_per_block = 256;
int num_blocks = (num_ele+threads_per_block-1)/threads_per_block;

vecAdd<<<num_blocks , threads_per_block>>>(d_A, d_B, d_C, num_ele);

// copy device to host

cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

for(int i=0; i<num_ele; i++)
{
    if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5){
        fprintf(stderr, "Failed at %d\n", i);
        exit(EXIT_FAILURE);
    }
}

printf("Test passed \n");

//free memory

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

free(h_A);
free(h_B);
free(h_C);

return 0;
}
