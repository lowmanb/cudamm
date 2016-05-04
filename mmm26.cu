#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<iostream>
#include<string.h>

using namespace std;

//----------------------------------- Structures and Globals---------------------------------------------

typedef struct {
    int dimension1;
    int dimension2;
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory
float *A, *B, *C, *C_CPU;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;

//----------------------------------- host function definitions -----------------------------------------

void allocateAndInitializeAB();
void computeCpuMMM();
__global__ void computeGPUMMM(float* A, float* B, float* C, int size);
void copyMatricesToGPU();
void copyResultFromGPU();
void compareHostAndGpuOutput();
void die(const char *error);
void check_error(cudaError e);

//----------------------------------- CUDA function definitions -----------------------------------------
#define BLOCK_DIM 26
clock_t start, endt, mem_transfer, compute;

//-------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 100;
    A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
    B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
    B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;
    C_MD.dimension1 = A_MD.dimension1;
    C_MD.dimension2 = B_MD.dimension2;

    // pad dimensions to next multiple of BLOCK_DIM
    if (A_MD.dimension1 % BLOCK_DIM != 0)
        A_MD.dimension1 = ((int)A_MD.dimension1/(3*BLOCK_DIM) + 1) * 3*BLOCK_DIM;
    if (A_MD.dimension2 % BLOCK_DIM != 0)
        A_MD.dimension2 = ((int)A_MD.dimension2/(3*BLOCK_DIM) + 1) * 3*BLOCK_DIM; 
    if (B_MD.dimension1 % BLOCK_DIM != 0)
        B_MD.dimension1 = ((int)B_MD.dimension1/(3*BLOCK_DIM) + 1) * 3*BLOCK_DIM;
    if (B_MD.dimension2 % BLOCK_DIM != 0)
        B_MD.dimension2 = ((int)B_MD.dimension2/(3*BLOCK_DIM) + 1) * 3*BLOCK_DIM;
    if (C_MD.dimension1 % BLOCK_DIM != 0)
        C_MD.dimension1 = ((int)C_MD.dimension1/(3*BLOCK_DIM) + 1) * 3*BLOCK_DIM;
    if (C_MD.dimension2 % BLOCK_DIM != 0)
        C_MD.dimension2 = ((int)C_MD.dimension2/(3*BLOCK_DIM) + 1) * 3*BLOCK_DIM;

    printf("Matrix A is %d-by-%d\n", A_MD.dimension1, A_MD.dimension2);
    printf("Matrix B is %d-by-%d\n", B_MD.dimension1, B_MD.dimension2);
    printf("Matrix C is %d-by-%d\n", C_MD.dimension1, C_MD.dimension2);

    allocateAndInitializeAB();

    // matrix matrix multiplication in the CPU
    /*
    start = clock();
    computeCpuMMM();
    endt = clock();
    double elapsed = (endt - start) / (double) CLOCKS_PER_SEC;
    printf("Computation time in the CPU: %f seconds\n", elapsed);
    */

    // MMM on the GPU
    start = clock();
    copyMatricesToGPU();
    endt = clock();
    mem_transfer = endt - start;

    dim3 blocks(A_MD.dimension1/(3*BLOCK_DIM), A_MD.dimension1/(3*BLOCK_DIM));
    dim3 threads(BLOCK_DIM, BLOCK_DIM);

    start = clock();
    computeGPUMMM<<<blocks, threads>>>(A_GPU, B_GPU, C_GPU, A_MD.dimension1);
    cudaThreadSynchronize();
    endt = clock();
    compute = endt - start;

    check_error(cudaGetLastError());
    start = clock();
    copyResultFromGPU();
    endt = clock();
    mem_transfer += endt - start;
    printf("Memory Transfer time in the GPU: %f seconds\n", mem_transfer / (double) CLOCKS_PER_SEC);
    printf("Computation time in the GPU: %f seconds\n", compute / (double) CLOCKS_PER_SEC);
    printf("Total time in the GPU: %f seconds\n", (mem_transfer + compute) / (double) CLOCKS_PER_SEC);

    /*
    for (int i=0; i < C_MD.dimension1; i++) {
       printf("\n");
        for (int j=0; j<C_MD.dimension2; j++)
            printf("%.2f ", C_CPU[i*C_MD.dimension2+j]);
    }
    */

    //printf("Comparing answers...\n");
    //compareHostAndGpuOutput();

    return 0;
}

// allocate and initialize A and B using a random number generator
void allocateAndInitializeAB() {

    size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
    A = (float*) malloc(sizeofA);

    srand(5);
    //srand(time(NULL));
    for (int i = 0; i < A_MD.dimension1; i++) {
        for (int j = 0; j < A_MD.dimension2; j++) {
            int index = i * A_MD.dimension2 + j;
            A[index] = (rand() % 1000) * 0.001;
        }
    }

    size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
    B = (float*) malloc(sizeofB);
    for (int i = 0; i < B_MD.dimension1; i++) {
        for (int j = 0; j < B_MD.dimension2; j++) {
            int index = i * B_MD.dimension2 + j;
            B[index] = (rand() % 1000) * 0.001;
        }
    }
}

// allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {

    size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
    check_error(cudaMalloc((void **) &A_GPU, sizeofA));
    check_error(cudaMemcpy(A_GPU, A, sizeofA, cudaMemcpyHostToDevice));

    size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
    check_error(cudaMalloc((void **) &B_GPU, sizeofB));
    check_error(cudaMemcpy(B_GPU, B, sizeofB, cudaMemcpyHostToDevice));

    size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
    check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
    size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
    C_CPU = (float*) malloc(sizeofC);
    check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
}

// do a straightforward matrix-matrix multiplication in the CPU
// notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are
// not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {

    // allocate the result matrix for the CPU computation
    size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
    C = (float*) malloc(sizeofC);

    // compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
    for (int i = 0; i < A_MD.dimension1; i++) {
        int a_i = i * A_MD.dimension2;
        int c_i = i * C_MD.dimension2;
        for (int j = 0; j < B_MD.dimension2; j++) {
            int c_index = c_i + j;
            C[c_index] = 0;
            for (int k = 0; k < B_MD.dimension1; k++) {
                int a_index = a_i + k;
                int b_index = k * B_MD.dimension2 + j;
                C[c_index] += A[a_index] * B[b_index];
            }
        }
    }
}

__global__ void computeGPUMMM(float* A, float* B, float* C, int size) {

    // get the row and column of current thread
    // this row and column is relative to the top left
    // tile in this block's region
    int row = blockIdx.y * 3*BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * 3*BLOCK_DIM + threadIdx.x;

    // allocate this blocks quadrant (9X the block area)
    __shared__ float AA[3*BLOCK_DIM][3*BLOCK_DIM];
    __shared__ float BB[3*BLOCK_DIM][3*BLOCK_DIM];

    // sums for each point this thread responsible for
    float sum0 = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;
    float sum4 = 0;
    float sum5 = 0;
    float sum6 = 0;
    float sum7 = 0;
    float sum8 = 0;

    int kk;
    int k;
    int i;
    int j;
    
    // ^  15 local variables should fit into the 32 available registers

    // as all matrix qudrants necessary for this block's computation will not 
    // fit into shared memory, the entire computation is blocked by kk
    for (kk = 0; kk < size; kk += 3*BLOCK_DIM) {

        // load the approprate data from memory
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                AA[threadIdx.y + i*BLOCK_DIM][threadIdx.x + j*BLOCK_DIM] =
                    A[(i*BLOCK_DIM + row) * size + threadIdx.x + kk + j*BLOCK_DIM];
                BB[threadIdx.y + i*BLOCK_DIM][threadIdx.x + j*BLOCK_DIM] =
                    B[(kk + threadIdx.y + i*BLOCK_DIM) * size + col + j*BLOCK_DIM];
            }
        }

        // since we stride across memory written by different warps, a sync is needed
        __syncthreads();

        // compute the partial dot products for all four quadrants
        for (k = 0; k < 3*BLOCK_DIM; k++) {
            sum0 += AA[threadIdx.y][k] * BB[k][threadIdx.x]; 
            sum1 += AA[threadIdx.y][k] * BB[k][threadIdx.x + BLOCK_DIM]; 
            sum2 += AA[threadIdx.y][k] * BB[k][threadIdx.x + 2*BLOCK_DIM]; 
            sum3 += AA[threadIdx.y + BLOCK_DIM][k] * BB[k][threadIdx.x]; 
            sum4 += AA[threadIdx.y + BLOCK_DIM][k] * BB[k][threadIdx.x + BLOCK_DIM]; 
            sum5 += AA[threadIdx.y + BLOCK_DIM][k] * BB[k][threadIdx.x + 2*BLOCK_DIM]; 
            sum6 += AA[threadIdx.y + 2*BLOCK_DIM][k] * BB[k][threadIdx.x]; 
            sum7 += AA[threadIdx.y + 2*BLOCK_DIM][k] * BB[k][threadIdx.x + BLOCK_DIM]; 
            sum8 += AA[threadIdx.y + 2*BLOCK_DIM][k] * BB[k][threadIdx.x + 2*BLOCK_DIM]; 
        }

        // sync is needed before writing to AA & BB
        __syncthreads();
    }

    // write the final dot products to C
    C[row * size + col] = sum0;
    C[row * size + col + BLOCK_DIM] = sum1;
    C[row * size + col + 2*BLOCK_DIM] = sum2;
    C[(row + BLOCK_DIM) * size + col] = sum3;
    C[(row + BLOCK_DIM) * size + col + BLOCK_DIM] = sum4;
    C[(row + BLOCK_DIM) * size + col + 2*BLOCK_DIM] = sum5;
    C[(row + 2*BLOCK_DIM) * size + col] = sum6;
    C[(row + 2*BLOCK_DIM) * size + col + BLOCK_DIM] = sum7;
    C[(row + 2*BLOCK_DIM) * size + col + 2*BLOCK_DIM] = sum8;

}

// function to determine if the GPU computation is done correctly by comparing the output from the GPU with that
// from the CPU
void compareHostAndGpuOutput() {
    int totalElements = C_MD.dimension1 * C_MD.dimension2;
    int missmatchCount = 0;
    for (int i = 0; i < totalElements; i++) {
        if (fabs(C[i] - C_CPU[i]) > 0.01) {
            missmatchCount++;
            //printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
        }
    }
    if (missmatchCount > 0) {
        printf("Computation is incorrect: outputs do not match in %d indexes\n", missmatchCount);
    } else {
        printf("Computation is correct: CPU and GPU outputs match\n");
    }
}

// Prints the specified error message and then exits
void die(const char *error) {
    printf("%s", error);
    exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
    if (e != cudaSuccess) {
        printf("\nCUDA error: %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

/*
void optimizedCpuMMM() {
    // allocate the result matrix for the CPU computation
    size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
    C = (float*) malloc(sizeofC);
    memset(C, 0f, sizeofC);

    // compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
    int BLOCK_SIZE = 50;
    for (int jj = 0; jj < B_MD.dimension2; jj += BLOCK_SIZE) {
        int j_limit = jj + BLOCK_SIZE;
        for (int j = 0; j < B_MD.dimension1; j++) {
            int c_index = j;
            for (int ii = 0; i < A_MD.dimension1; ii += BLOCK_SIZE) {
                int i_limit = ii + BLOCK_SIZE;
                for (int i = 0; i < A_MD.dimension2; i++) {
                    int a_i = i * A_MD.dimension2;
                    int c_i += i * C_MD.dimension2;
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        int a_index = a_i + k + ;
                        int b_index = k * B_MD.dimension2 + j;
                        C[c_index] += A[a_index] * B[b_index];
                    }
                }
            }
        }
    }
}
*/
