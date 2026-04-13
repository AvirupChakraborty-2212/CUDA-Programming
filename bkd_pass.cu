//define

#define N 1024
#define IN 512
#define OUT 512
#define H1 2048
#define H2 2048
#define H3 2048
#define lr 0.001f

// input dim N * IN
// output dim IN * OUT

// kernels

    // forward pass
    
        // matmul
__global__ void matmul(float *X, float *W, float *b, float *Z, int N, int IN, int OUT)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < OUT)
    {
        float sum = 0.0f;
        for(int i=0; i<IN; i++)
        {
            sum += X[row * IN + i] * W[i * OUT  + col];
        }
        Z[row * OUT + col] = sum + b[col];
    }
}

        // relu
__global__ void relu(float *Z, float *A, int N, int OUT)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * OUT + col;

    if(row < N && col < OUT)
    {
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

    // MSE loss = dZ4
__global__ void mse_loss(float *Z, float *Y, float *dZ, int N, int OUT)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * OUT + col;
    float dim_S = (float)N*OUT;

    if(row<N && col<OUT)
    {
        dZ[idx] = 2*(Z[idx] - Y[idx])*(1.0f/dim_S);
    }
}

    // backward pass

        // matmul transpose dA = dZ * W^T
__global__ void matmul_transpose_dA(float *dZ, float *W, float *dA, int N, int IN, int OUT)
{
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;

    if(row < N && col < IN)
    {
        float sum = 0.0f;
        for(int i=0; i<OUT; i++)
        {
            sum += dZ[row * OUT + i] * W[col * OUT + i];
        }
        dA[row * IN + col] = sum;
    }
}

        // matmul transpose dW = A^T * dZ
__global__ void matmul_transpose_dW(float *A, float *dZ, float *dW, int N, int IN, int OUT)
{
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;

    if(row < IN && col < OUT)
    {
        float sum = 0.0f;
        for(int i=0; i<N; i++)
        {
            sum += A[i * IN + row] * dZ[i * OUT + col];
        }
        dW[row * OUT + col] = sum;
    }
}
        // db = sum_rows(dZ) // 1-D grid
__global__ void db_sum(float *dZ, float *db, int N, int OUT)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(col < OUT)
    {
        float sum = 0.0f;
        for(int i=0; i<N; i++)
        {
            sum+=dZ[i * OUT + col];
        }
        db[col] = sum;
    }
}

        // relu backward -- input to prev layers 
__global__ void relu_bwd(float *dZ, float *dA, float *Z, int N, int OUT)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * OUT + col;

    if(row < N && col < OUT)
    {
        dZ[idx] =(Z[idx] > 0.0f) ? dA[idx] : 0.0f;
    }
}        

    // SGD update // 2D grid but 1D update for bias
__global__ void sgd(float *W, float *b, float *db, float *dW, int IN, int H, float lr)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * H + col;

    if(row < IN && col < H)
    {
        W[idx] = W[idx] - (lr * dW[idx]);
    }

    if(row == 0 && col < H)
    {
        b[col] = b[col] - (lr * db[col]);
    }
}

//calling functions

int main()
{

    // size

    size_t size_x = (size_t)N * IN * sizeof(float);

    size_t size_w1 = (size_t)IN * H1 * sizeof(float);
    size_t size_w2 = (size_t)H1 * H2 * sizeof(float);
    size_t size_w3 = (size_t)H2 * H3 * sizeof(float);
    size_t size_w4 = (size_t)H3 * OUT* sizeof(float);

    size_t size_b1 = (size_t)H1 * sizeof(float);
    size_t size_b2 = (size_t)H2 * sizeof(float);
    size_t size_b3 = (size_t)H3 * sizeof(float);
    size_t size_b4 = (size_t)OUT* sizeof(float);

    size_t size_za1 =(size_t)N * H1* sizeof(float);
    size_t size_za2 =(size_t)N * H2* sizeof(float);
    size_t size_za3 =(size_t)N * H3* sizeof(float);

    size_t size_z4 = (size_t)N * OUT* sizeof(float);

    // allocate host memory

    float *h_X = (float *)malloc(size_x);

    float *h_W1 = (float *)malloc(size_w1);
    float *h_W2 = (float *)malloc(size_w2);
    float *h_W3 = (float *)malloc(size_w3);
    float *h_W4 = (float *)malloc(size_w4);

    float *h_b1 = (float *)malloc(size_b1);
    float *h_b2 = (float *)malloc(size_b2);
    float *h_b3 = (float *)malloc(size_b3);
    float *h_b4 = (float *)malloc(size_b4);

    /**float *h_Z1 = (float *)malloc(size_za1);
    float *h_Z2 = (float *)malloc(size_za2);
    float *h_Z3 = (float *)malloc(size_za3);**/
    float *h_Z4 = (float *)malloc(size_z4);

    float *h_y = (float *)malloc(size_z4);

    /**float *h_A1 = (float *)malloc(size_za1);
    float *h_A2 = (float *)malloc(size_za2);
    float *h_A3 = (float *)malloc(size_za3);**/

    // allocate device memory

    float *d_X;
    float *d_W1;
    float *d_W2;
    float *d_W3;
    float *d_W4; 
    float *d_b1;
    float *d_b2;
    float *d_b3;
    float *d_b4;
    float *d_Z1;
    float *d_Z2;
    float *d_Z3;
    float *d_Z4;
    float *d_A1;
    float *d_A2;
    float *d_A3;
    float *d_Y;

    //bkd

    float *d_dZ1;
    float *d_dZ2;
    float *d_dZ3;
    float *d_dZ4;

    float *d_dW1;
    float *d_dW2;
    float *d_dW3;
    float *d_dW4;

    float *d_dA1;
    float *d_dA2;
    float *d_dA3;

    float *d_db1;
    float *d_db2;
    float *d_db3;
    float *d_db4;

    //fwd 

    cudaMalloc(&d_X, size_x);

    cudaMalloc(&d_W1, size_w1);
    cudaMalloc(&d_W2, size_w2);
    cudaMalloc(&d_W3, size_w3);
    cudaMalloc(&d_W4, size_w4);

    cudaMalloc(&d_b1, size_b1);
    cudaMalloc(&d_b2, size_b2);
    cudaMalloc(&d_b3, size_b3);
    cudaMalloc(&d_b4, size_b4);

    cudaMalloc(&d_Z1, size_za1);
    cudaMalloc(&d_Z2, size_za2);
    cudaMalloc(&d_Z3, size_za3);
    cudaMalloc(&d_Z4, size_z4);

    cudaMalloc(&d_Y, size_z4);

    cudaMalloc(&d_A1, size_za1);
    cudaMalloc(&d_A2, size_za2);
    cudaMalloc(&d_A3, size_za3);

    // bkd

    cudaMalloc(&d_dW1, size_w1);
    cudaMalloc(&d_dW2, size_w2);
    cudaMalloc(&d_dW3, size_w3);
    cudaMalloc(&d_dW4, size_w4);

    cudaMalloc(&d_db1, size_b1);
    cudaMalloc(&d_db2, size_b2);
    cudaMalloc(&d_db3, size_b3);
    cudaMalloc(&d_db4, size_b4);

    cudaMalloc(&d_dZ1, size_za1);
    cudaMalloc(&d_dZ2, size_za2);
    cudaMalloc(&d_dZ3, size_za3);
    cudaMalloc(&d_dZ4, size_z4);

    cudaMalloc(&d_dA1, size_za1);
    cudaMalloc(&d_dA2, size_za2);
    cudaMalloc(&d_dA3, size_za3);
    
    // cuda mem info

    // initialize
 /**   for(int i=0; i< N*IN; i++)
    {
        h_X[i] = RAND()/(float)RAND_MAX;
    }
    for(int j=0; j< N*IN; j++)
    {
        h_X[i] = RAND()/(float)RAND_MAX;
    }
    for(int k=0; k< N*IN; k++)
    {
        h_X[i] = RAND()/(float)RAND_MAX;
    }
    for(int i=0; i< N*IN; i++)
    {
        h_X[i] = RAND()/(float)RAND_MAX;
    }
    for(int i=0; i< N*IN; i++)
    {
        h_X[i] = RAND()/(float)RAND_MAX;
    }**/


    // copy to device 
    cudaMemcpy(d_X, h_X, size_x, cudaMemcpyHostToDevice);

    cudaMemcpy(d_W1, h_W1, size_w1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, size_w2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, h_W3, size_w3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W4, h_W4, size_w4, cudaMemcpyHostToDevice);

    cudaMemcpy(d_b1, h_b1, size_b1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, size_b2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, size_b3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4, h_b4, size_b4, cudaMemcpyHostToDevice);

    cudaMemcpy(d_Y, h_y, size_z4, cudaMemcpyHostToDevice);

    // cuda event

    // grids blocks
    dim3 block(16,16);
    dim3 grid1((H1+15)/16, (N+15)/16);
    dim3 grid2((H2+15)/16, (N+15)/16);
    dim3 grid3((H3+15)/16, (N+15)/16);
    //dim3 grid4((H4+15)/16, (N+15)/16);
    dim3 grid5((OUT+15)/16, (N+15)/16);

    // kernel launch

        // fwd pass

        // checkpoint A2 Z4

        matmul<<<grid1, block>>>(d_X, d_W1, d_b1, d_Z1, N, IN, H1);
        relu<<<grid1, block>>>(d_Z1, d_A1, N, H1);

        cudaFree(d_Z1); // free Z1 after use

        matmul<<<grid2, block>>>(d_A1, d_W2, d_b2, d_Z2, N, H1, H2);
        relu<<<grid2, block>>>(d_Z2, d_A2, N, H2);

        cudaFree(d_A1); cudaFree(d_Z2);

        matmul<<<grid3, block>>>(d_A2, d_W3, d_b3, d_Z3, N, H2, H3);
        relu<<<grid3, block>>>(d_Z3, d_A3, N, H3);

        cudaFree(d_Z3);

        matmul<<<grid5, block>>>(d_A3, d_W4, d_b4, d_Z4, N, H3, OUT);

        cudaFree(d_A3);

        // bkd pass

        //layer 4
        mse_loss<<<grid5, block>>>(d_Z4, d_Y, d_dZ4,  N, OUT);
        dim3 grid6((H3+15)/16, (N+15)/16);
        matmul_transpose_dA<<<grid6, block>>>(d_dZ4, d_W4, d_dA3,  N,  H3,  OUT);

        dim3 grid7((OUT+15)/16, (H3+15)/16);
        
        cudaMalloc(&d_A3, size_za3);
        cudaMalloc(&d_Z3, size_za3);
        matmul<<<grid3, block>>>(d_A2, d_W3, d_b3, d_Z3, N, H2, H3);
        relu<<<grid3, block>>>(d_Z3, d_A3, N, H3);
        matmul_transpose_dW<<<grid7, block>>>(d_A3, d_dZ4, d_dW4,  N,  H3,  OUT); 
        cudaFree(d_A3); 

        db_sum<<<(OUT+255)/256 , 256>>>(d_dZ4, d_db4,  N,  OUT);

        //layer 3       
        relu_bwd<<<grid3, block>>>(d_dZ3, d_dA3, d_Z3,  N,  H3);
        cudaFree(d_Z3);

        dim3 grid8((H2+15)/16, (N+15)/16);
        matmul_transpose_dA<<<grid8, block>>>(d_dZ3, d_W3, d_dA2,  N,  H2,  H3);        
        dim3 grid9((H3+15)/16, (H2+15)/16);
        matmul_transpose_dW<<<grid9, block>>>(d_A2, d_dZ3, d_dW3,  N,  H2,  H3);     

        db_sum<<<(H3+255)/256 , 256>>>(d_dZ3, d_db3,  N,  H3);

        //layer 2
        cudaMalloc(&d_A1, size_za1);
        cudaMalloc(&d_Z1, size_za1);
        matmul<<<grid1, block>>>(d_X, d_W1, d_b1, d_Z1, N, IN, H1);
        relu<<<grid1, block>>>(d_Z1, d_A1, N, H1);

        matmul<<<grid2, block>>>(d_A1, d_W2, d_b2, d_Z2, N, H1, H2);

        relu_bwd<<<grid2, block>>>(d_dZ2, d_dA2, d_Z2,  N,  H2);
        cudaFree(d_Z2);
        dim3 grid10((H1+15)/16, (N+15)/16);
        matmul_transpose_dA<<<grid10, block>>>(d_dZ2, d_W2, d_dA1,  N,  H1,  H2);
        dim3 grid11((H2+15)/16, (H1+15)/16);
        matmul_transpose_dW<<<grid11, block>>>(d_A1, d_dZ2, d_dW2,  N,  H1,  H2);  
        cudaFree(d_A1);       
        db_sum<<<(H2+255)/256 , 256>>>(d_dZ2, d_db2,  N,  H2);

        //layer 1

        relu_bwd<<<grid1, block>>>(d_dZ1, d_dA1, d_Z1,  N,  H1);       
        cudaFree(d_Z1);
        dim3 grid14((H1+15)/16, (IN+15)/16);
        matmul_transpose_dW<<<grid14, block>>>(d_X, d_dZ1, d_dW1,  N,  IN,  H1);        
        db_sum<<<(H1+255)/256 , 256>>>(d_dZ1, d_db1,  N,  H1);

        //sgd
        sgd<<<grid7, block>>>(d_W4, d_b4, d_db4, d_dW4,  H3,  OUT,  lr);
        sgd<<<grid9, block>>>(d_W3, d_b3, d_db3, d_dW3,  H2,  H3,  lr);
        sgd<<<grid11, block>>>(d_W2, d_b2, d_db2, d_dW2,  H1,  H2,  lr);
        sgd<<<grid14, block>>>(d_W1, d_b1, d_db1, d_dW1,  IN,  H1,  lr);


    // copy to host

    cudaMemcpy(h_Z4, d_Z4, size_z4, cudaMemcpyDeviceToHost);

    // cuda event

    // free 


    
    return 0;
}