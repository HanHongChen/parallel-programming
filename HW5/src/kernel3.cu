#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16
#define GROUP_SIZE 2

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

    int j = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int i = (blockIdx.y * blockDim.y + threadIdx.y) * GROUP_SIZE;

    if(j >= resX || i >= resY) return; 

    for(int k = i; k < i + GROUP_SIZE; k++){
      if(k >= resY)break;

      for(int l = j; l < j + GROUP_SIZE; l++){
        if(l >= resX) break;

        float x = lowerX + l * stepX;
        float y = lowerY + k * stepY;
        int idx = resX * k  + l ;
        result_ptr[idx] = mandel(x, y, maxIterations);  
      }
    }

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{   //upperX = x1, upperY = y1, lowerX = x0, lowerY = y0,
    //img = output, resX = width, resY = height, maxIterations = maxIterations
    float stepX = (upperX - lowerX) / resX;//x軸要負責的
    float stepY = (upperY - lowerY) / resY;//y軸要負責的

    int size = resX * resY * sizeof(int);
    int* result_ptr;
    int* result_pin;
    size_t pitch;

    //cudaMallocPitch是為了產生一段aligned的記憶體空間
    cudaMallocPitch((void**) &result_ptr, &pitch, resX * sizeof(int), resY);
    //cudaHostAlloc是為了將記憶體pinned住
    cudaHostAlloc(&result_pin, size, cudaHostAllocMapped);

    //設定GPU及memory
    // + BLOCK_SIZE - 1 是為了確保當stepX非BLOCK_SIZE整數倍時還能夠確保整個向量都能被分配到block
    //注意最後是/BLOCK_SIZE
    int block_x = (resX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_y = (resY + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads_per_block(BLOCK_SIZE / GROUP_SIZE, BLOCK_SIZE / GROUP_SIZE);
    dim3 blocks_per_grid(block_x, block_y);
    mandelKernel<<<blocks_per_grid, threads_per_block>>>(lowerX, lowerY, resX, resY, maxIterations, stepX, stepY, result_ptr);

    cudaMemcpy(result_pin, result_ptr, size, cudaMemcpyDeviceToHost);
    memcpy(img, result_pin, size);
    cudaFree(result_ptr);
    cudaFreeHost(result_pin);

}
