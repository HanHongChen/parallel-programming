#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16
#pragma nvcc optimize("Xptxas", "O3")

__device__ int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, int resX, int resY,
                                int maxIterations, float stepX, float stepY, int* result_ptr) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(j >= resX || i >= resY) return; 
    float x = lowerX + j * stepX;
    float y = lowerY + i * stepY;

    int idx = resX * i + j;
    result_ptr[idx] = mandel(x, y, maxIterations);

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{   //upperX = x1, upperY = y1, lowerX = x0, lowerY = y0,
    //img = output, resX = width, resY = height, maxIterations = maxIterations
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    int* result_ptr;
    cudaMalloc((void**) &result_ptr, size);

    //設定GPU及memory
    // + BLOCK_SIZE - 1 是為了確保當stepX非BLOCK_SIZE整數倍時還能夠確保整個向量都能被分配到block
    //注意最後是/BLOCK_SIZE
    int block_x = (resX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_y = (resY + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid(block_x, block_y);
    mandelKernel<<<blocks_per_grid, threads_per_block>>>(lowerX, lowerY, resX, resY, maxIterations, stepX, stepY, result_ptr);
    
    cudaMemcpy(img, result_ptr, size, cudaMemcpyDeviceToHost);
    cudaFree(result_ptr);
}
