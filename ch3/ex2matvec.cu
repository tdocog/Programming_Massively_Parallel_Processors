#include <stdio.h>
#include <cuda_runtime.h>

__global__
void matmulKernel(const float *A, const float *B, float *C, int N)
{
    int thrd= blockIdx.x * blockDim.x + threadIdx.x; 

    if (thrd < N)
    {
        float dot = 0.0f; 
        
        for (int i = 0; i < N; i++)
        {
            dot += A[thrd*N + i] * B[i];
        }
        C[thrd] = dot;
    }
}

int main()
{
    int N = 1024;
    size_t bytes = N*N*sizeof(float);

    // Host allocation
    float *A_h = (float*)malloc(bytes);
    float *B_h = (float*)malloc(bytes/N);
    float *C_h = (float*)malloc(bytes/N);

    // Initialize matrices
    for (int i = 0; i < N*N; i++)
    {
        A_h[i] = 1.0f;
        if (i < N) B_h[i] = 2.0f;
    }

    // Device allocation
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, bytes);
    cudaMalloc((void**)&B_d, bytes/N);
    cudaMalloc((void**)&C_d, bytes/N);

    // Copy to device
    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes/N, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 1);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1);
    matmulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);

    // Copy result back
    cudaMemcpy(C_h, C_d, bytes/N, cudaMemcpyDeviceToHost);

    // Print a sample result
    printf("GPU result sample C[0] = %f\n", C_h[0]);

    // Cleanup
    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}

