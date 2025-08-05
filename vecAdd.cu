#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


void vecAdd_h(float* A_h, float* B_h, float* C_h, int n)
{
   for (int i =0; i < n; ++i)
   {
       C_h[i] = A_h[i] + B_h[i];
   } 
}

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) 
    {
        C[i] = A[i] + B[i];
    }
}

void vecAdd_d(float* A_h, float* B_h, float* C_h, int n)
{
    // now doing cuda here
    float *A_d;
    float *B_d;
    float *C_d;
    int size = n * sizeof(float);


    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);


    // transfer mem and launch kernel
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // move out back to host 
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    // define bs here for like stuff ts

    #define N 5  // define vector size
    float A[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}; // input vector A
    float B[N] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f}; // input vector B
    float C[N]; // output vector C

    vecAdd_h(A, B, C, N);

    printf("CPU Result of A + B = C:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%.2f ", C[i]);
    }
    printf("\n");


    vecAdd_d(A, B, C, N);

    printf("GPU Result of A + B = C:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%.2f ", C[i]);
    }
    printf("\n");


}

