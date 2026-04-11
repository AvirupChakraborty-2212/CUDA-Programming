#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

#define Br 32
#define Bc 32
#define B 8
#define NH 16
#define D 64
//#define N 1024


__global__ void flash_attn(float *Q, float *K, float *V, float *O, float *M, float *L, int N, float scale)
{
    //no. of tiles
    int Tr = N/Br; 
    int Tc = N/Bc;

    int b = blockIdx.x;
    int h = blockIdx.y;
    int tx = threadIdx.x;

    //base offsets
    int qkv = b*NH*N*D + h*N*D;
    int lm = b*NH*N + h*N;

    extern __shared__ float smem[];
    float *Qi = smem; // Br * D
    float *Kj = Qi + (Br * D); // Bc * D
    float *Vj = Kj + (Bc * D); // Bc * D
    float *S = Vj + (Bc * D); // Br * Bc

    for(int j=0; j<Tc; j++)
    {

        // load data

        for(int k=0; k<D; k++)
        {
            int kvidx = qkv + (j * Bc + tx) * D + k;
            Kj[tx * D + k] = K[kvidx];
            Vj[tx * D + k] = V[kvidx];
        }
        __syncthreads();

        for(int i=0; i<Tr; i++)
        {
            float q_local[D];
            for(int k=0; k<D; k++)
            {
                int qidx = qkv + (i * Br + tx) * D + k;
                q_local[k] = Q[qidx];
            }

            float l_prev = L[lm + (i * Br + tx)];
            float m_prev = M[lm + (i * Br + tx)];

            float m_tile = -FLT_MAX;

            // compute QK^T, scale sum and max tile 

            for(int a=0; a<Bc; a++) // a=col
            {
                float sum = 0.0f;
                for(int k=0; k<D; k++)
                {
                    sum += q_local[k] * Kj[a * D + k]; //
                }
                sum *= scale;
                S[tx * Bc + a] = sum;
                if(sum>m_tile)
                {
                    m_tile = sum;
                }
            }

            // exp and acccumulate sum
            float l_tile = 0.0f;
            for(int a=0; a<Bc; a++)
            {
                S[tx * Bc + a] = expf(S[tx * Bc + a] - m_tile);
                l_tile += S[tx * Bc + a];
            }

            // online softmax and output rescaling

            float m_new = fmaxf(m_tile, m_prev); 
            float alpha = expf(m_prev - m_new);
            float beta =  expf(m_tile - m_new);
            float l_new = l_prev * alpha + l_tile * beta;

            // load output from HBM
            for(int k=0; k<D; k++)
            {
                float out_prev = O[qkv + (i * Br + tx) * D + k];

                // pvsum
                float pvsum = 0.0f;
                for(int a=0; a<Bc; a++)
                {
                    pvsum += S[tx * Bc + a] * Vj[a * D + k];
                }
                float out_new =  (1.0f/l_new) * (out_prev * alpha * l_prev + pvsum * beta);
                O[qkv + (i * Br + tx) * D + k] = out_new;
            }

            M[lm + (i * Br + tx)] = m_new;
            L[lm + (i * Br + tx)] = l_new;
        }
        __syncthreads();
    }
    
}


int main()
{
    // size
    int N = 1024;
    size_t smem_size = (Br*D + 2*Bc*D + Br*Bc) * sizeof(float);
    size_t qkvo_size = (size_t)B*NH*N*D*sizeof(float);
    size_t lm_size = (size_t)B*NH*N*sizeof(float);
    

    // host memory
    float *h_Q = (float *)malloc(qkvo_size);
    float *h_K = (float *)malloc(qkvo_size);
    float *h_V = (float *)malloc(qkvo_size);
    float *h_O = (float *)malloc(qkvo_size);
    float *h_L = (float *)malloc(lm_size);
    float *h_M = (float *)malloc(lm_size);

    // device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_M;
    cudaMalloc(&d_Q, qkvo_size);
    cudaMalloc(&d_K, qkvo_size);
    cudaMalloc(&d_V, qkvo_size);
    cudaMalloc(&d_O, qkvo_size);
    cudaMalloc(&d_L, lm_size);
    cudaMalloc(&d_M, lm_size);

    // initialize
    for(int i=0; i<B*NH*N*D; i++)
    {
        h_Q[i]=rand()/(float)RAND_MAX;
        h_K[i]=rand()/(float)RAND_MAX;
        h_V[i]=rand()/(float)RAND_MAX;
        h_O[i]=0.0f;
    }
    for(int j=0; j<B*NH*N; j++)
    {
        h_L[j]= 0.0f;
        h_M[j]= -FLT_MAX;
    }

    float scale = 1.0f/sqrt(D);

    // copy host to device
    cudaMemcpy(d_Q, h_Q, qkvo_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkvo_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkvo_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L, lm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, lm_size, cudaMemcpyHostToDevice);

    // block and grid
    dim3 block(Br);
    dim3 grid(B,NH);

    // cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // kernel launch
    flash_attn<<<grid, block, smem_size>>>(d_Q, d_K, d_V, d_O, d_M, d_L, N, scale);
    
    // cuda event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Elapsed Time: %f ms \n", ms);

    // copy device to host
    cudaMemcpy(h_O, d_O, qkvo_size, cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(d_M);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_L);
    free(h_M); 

    return 0;
}
