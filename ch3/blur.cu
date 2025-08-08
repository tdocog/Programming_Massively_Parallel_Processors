#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include <cuda_runtime.h>
#include "stb_image.h"
#include <stdio.h>

 //      int grayOff = row*width + col;
        
        // 3 bc rgb is 3 channels
//        int rgbOff = grayOff*3;
 
__global__ 
void blurKernel(unsigned char* Pout, unsigned char* Pin, int width, int height, int radius)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    // we need to iterate from col - rad to col + rad and same for row
    if (row < height && col < width)
    {
        int count = 0;
        int r_sum = 0, g_sum = 0, b_sum = 0;

        for (int blurRow = -radius; blurRow < radius + 1; blurRow++)
        {
            for (int blurCol = -radius; blurCol < radius + 1; blurCol++)
            {
                int currRow = row + blurRow;
                int currCol = col + blurCol;

                if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width)
                {
                    r_sum += Pin[(currRow * width + currCol) * 3];
                    g_sum += Pin[(currRow * width + currCol) * 3 + 1];
                    b_sum += Pin[(currRow * width + currCol) * 3 + 2];
                    ++count;
                }
            }
        }
        Pout[(row * width + col) * 3 + 0] = (unsigned char) (r_sum / count);
        Pout[(row * width + col) * 3 + 1] = (unsigned char) (g_sum / count);
        Pout[(row * width + col) * 3 + 2] = (unsigned char) (b_sum / count);
    }
}  

int main()
{
    int width, height, channels;
    unsigned char * Pin_h = stbi_load("Grace_Hopper.jpg", &width, &height, &channels, 0);
    unsigned char * Pout_h = (unsigned char*) malloc(width*height*3);

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
    cudaMalloc((void**) &Pout_d, channels*width*height*sizeof(char));

    // mem copy
    cudaMemcpy(Pin_d, Pin_h, width*height*channels*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // call kernel
    dim3 dimBlock(16,16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
        
    blurKernel<<< dimGrid , dimBlock >>>(Pout_d, Pin_d, width, height, 3);

    //mem copy again
    cudaMemcpy(Pout_h, Pout_d, 3 * width * height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // clean up device memory
    cudaFree(Pin_d);
    cudaFree(Pout_d);

    // write grayscale out
    stbi_write_png("output.png", width, height, 3, Pout_h, width*3);

    // clean up
    stbi_image_free(Pin_h); 
    free(Pout_h);
    Pout_h = NULL;

    return 0; 
}


