/*
#define BM 128 // Block Rows
#define BN 128 // Block Cols
#define BK 8   // The "Inner" K dimension for tiling

#define T 16   // We use a 16x16 block (256 threads)
#define ele 8  // Each thread calculates an 8x8 patch

__global__ void matmul_warp_tiled(float *A, float *B, float *C, int N) {
    // 1. SHARED MEMORY (The Block's Desk)
    __shared__ float sA[BK][BM]; 
    __shared__ float sB[BK][BN];

    // 2. IDENTIFY THE TEAMS (Warps)
    // Linear ID from 0 to 255
    int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    
    // Which team (0-7)?
    int teamId = tid / 32; 
    // Which seat in the team (0-31)?
    int seatId = tid % 32; 

    // 3. MAP TEAMS TO AREAS
    // We arrange the 8 teams in a 2x4 grid
    int teamRow = (teamId / 4) * 64; // Team Zone starts at Row 0 or 64
    int teamCol = (teamId % 4) * 32; // Team Zone starts at Col 0, 32, 64, or 96

    // 4. MAP INDIVIDUALS WITHIN THE TEAM
    // Each team (32 people) is arranged as a 4x8 grid
    int localRowInTeam = (seatId / 8) * ele; 
    int localColInTeam = (seatId % 8) * ele;

    // 5. REGISTERS (The Worker's Hands)
    float acc[ele][ele]; 
    for(int i=0; i<ele; i++) 
        for(int j=0; j<ele; j++) acc[i][j] = 0.0f;

    // --- MAIN LOOP ---
    for (int k_tile = 0; k_tile < N; k_tile += BK) {
        
        // COLLABORATIVE LOAD (Everyone fills the Shared Desk)
        // [Each thread loads a few values to fill sA and sB...]
        __syncthreads();

        // MATH LOOP
        for (int k = 0; k < BK; k++) {
            // Register Caching (Pull from Desk to Hands)
            float regA[ele];
            float regB[ele];

            for (int i = 0; i < ele; i++) {
                // I only look at my TEAM'S row zone on the desk
                regA[i] = sA[k][teamRow + localRowInTeam + i];
            }
            for (int j = 0; j < ele; j++) {
                // I only look at my TEAM'S col zone on the desk
                regB[j] = sB[k][teamCol + localColInTeam + j];
            }

            // PURE REGISTER MATH
            for (int i = 0; i < ele; i++) {
                for (int j = 0; j < ele; j++) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // --- FINAL WRITE ---
    // Calculate final global position and write the 64 results
    // GlobalRow = BlockStart + TeamRow + MyRowInsideTeam


// 1. Calculate the starting corner for the entire Block
int blockRow_start = blockIdx.y * BM;
int blockCol_start = blockIdx.x * BN;

// 2. Loop through my 8x8 private patch (Registers -> Global)
for (int i = 0; i < ele; i++) {
    for (int j = 0; j < ele; j++) {
        
        // Final Row = Block Start + Team Zone Start + My Spot + loop index
        int finalRow = blockRow_start + teamRow + localRowInTeam + i;
        
        // Final Col = Block Start + Team Zone Start + My Spot + loop index
        int finalCol = blockCol_start + teamCol + localColInTeam + j;

        // Boundary Guard (Make sure we don't write outside the N x N floor)
        if (finalRow < N && finalCol < N) {
            C[finalRow * N + finalCol] = acc[i][j];
        }
    }
}
}
*/