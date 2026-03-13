// vec add using cuda managed memory

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

// allocate unified memory

float *d_A = NULL;
float *d_B = NULL;
float *d_C = NULL;
cudaMallocManaged((void**)&d_A, size);
cudaMallocManaged((void**)&d_B, size);
cudaMallocManaged((void**)&d_C, size);


// initialize

for(int i=0; i<num_ele; i++){
    d_A[i] = rand()/(float)RAND_MAX;
    d_B[i] = rand()/(float)RAND_MAX;
}


// call cuda kernel

int threads_per_block = 256;
int num_blocks = (num_ele+threads_per_block-1)/threads_per_block;

vecAdd<<<num_blocks , threads_per_block>>>(d_A, d_B, d_C, num_ele);

cudaDeviceSynchronize();

printf("value at 0 %f: \n", d_C[0]);


printf("Test passed \n");

//free memory

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

return 0;
}
