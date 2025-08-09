#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__
void matmulKernel(const float * A, const float * B, float * C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (col < N && row < N)
    {
        float dot = 0.0f; 
        
        for (int i = 0; i < N; i++)
        {
            dot += A[row*N + i] * B[i*N + col];
        }
        C[row*N + col] = dot;
    }

}

void matmulCPU(const float *A, const float *B, float *C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float dot = 0.0f;
            for (int i = 0; i < N; i++) {
                dot += A[row*N + i] * B[i*N + col];
            }
            C[row*N + col] = dot;
        }
    }
}

int main()
{
    int N = 1024;
    size_t bytes = N*N*sizeof(float);

    // alloc on the hsot
    float * A_h = (float *) malloc(bytes);
    float * B_h = (float *) malloc(bytes);
    float * C_h = (float *) malloc(bytes);

    // fill in the mat
    for (int i = 0; i < N*N; i++)
    {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    // Device alloc
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, bytes);
    cudaMalloc((void**)&B_d, bytes);
    cudaMalloc((void**)&C_d, bytes);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // --- Host to Device ---
    cudaEventRecord(start);
    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("[Timing] Host → Device: %.4f ms\n", ms);

    // --- Kernel ---
    dim3 dimBlock(16,16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    cudaEventRecord(start);
    matmulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("[Timing] Kernel execution: %.4f ms\n", ms);

    // --- Device to Host ---
    cudaEventRecord(start);
    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("[Timing] Device → Host: %.4f ms\n", ms);

    printf("C[0] = %f\n", C_h[0]);

     auto cpuStart = std::chrono::high_resolution_clock::now();
         float *C_cpu = (float*)malloc(bytes);
    matmulCPU(A_h, B_h, C_cpu, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = cpuEnd - cpuStart;

    printf("[Timing] CPU matmul: %.4f ms\n", cpuDuration.count());
    printf("CPU result sample C[0] = %f\n", C_cpu[0]);

    // Cleanup
    free(A_h); free(B_h); free(C_h);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
