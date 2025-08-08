#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include <cuda_runtime.h>
#include "stb_image.h"
#include <stdio.h>

__global__ 
void rgb2grayKernel(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        // gray output flattened offset
        int grayOff = row*width + col;
        
        // 3 bc rgb is 3 channels
        int rgbOff = grayOff*3;
        unsigned char r = Pin[rgbOff];
        unsigned char g = Pin[rgbOff + 1];
        unsigned char b = Pin[rgbOff + 2];

        Pout[grayOff] = 0.21f * r + 0.71f * g + 0.07f * b;
    }

}  

int main()
{
    int width, height, channels;
    unsigned char * Pin_h = stbi_load("Grace_Hopper.jpg", &width, &height, &channels, 0);
    unsigned char * Pout_h = (unsigned char*) malloc(width*height);

    if (Pin_h == NULL) 
    {
        printf("Error: could not load image\n");
        return 1;
    }

    printf("Loaded image with width %d, height %d, channels %d\n", width, height, channels);

    //alloc device mem
    unsigned char * Pin_d;
    unsigned char * Pout_d;

    cudaMalloc((void**) &Pin_d, width*height*channels * sizeof(char));
    cudaMalloc((void**) &Pout_d, width*height*sizeof(char));

    // mem copy
    cudaMemcpy(Pin_d, Pin_h, width*height*channels*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // call kernel
    dim3 dimBlock(16,16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
        
    rgb2grayKernel<<< dimGrid , dimBlock >>>(Pout_d, Pin_d, width, height);

    //mem copy again
    cudaMemcpy(Pout_h, Pout_d, width * height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // clean up device memory
    cudaFree(Pin_d);
    cudaFree(Pout_d);

    // write grayscale out
    stbi_write_png("output.png", width, height, 1, Pout_h, width);

    // clean up
    stbi_image_free(Pin_h); 
    free(Pout_h);
    Pout_h = NULL;

    return 0; 
}

