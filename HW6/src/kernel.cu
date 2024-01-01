#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolution(float *inp_data,
                            float *oup_dat,
                            float *fil_dat,
                            int imageHeight,
                            int imageWidth,
                            int filterWidth,
                            int halffilterSize){
    int gidY = blockIdx.x * blockDim.x + threadIdx.x;
    int gidX = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    for(int k = -halffilterSize; k <= halffilterSize; k++){
        for(int l = -halffilterSize; l <= halffilterSize; l++){
            int y = gidY + k;
            int x = gidX + l;

            if (x < 0 || x >= imageWidth || y < 0 || y >=  imageHeight) {
                // padding 0
                continue;
            } else {
                // 計算 convolution
                sum += inp_data[y * imageWidth + x] * fil_dat[(k + halffilterSize) * filterWidth + l + halffilterSize];
            }
        }
    }
    oup_dat[gidY * imageWidth + gidX] = sum;
}
//似乎增加這個編譯才會成功
extern "C" {
	void hostFEcuda(int filterWidth, float *filter, int imageHeight, int imageWidth,
                 float *inputImage, float *outputImage);
}
#define BLOCK_SIZE 16
void hostFEcuda(int filterWidth, float *filter, int imageHeight, int imageWidth,
                 float *inputImage, float *outputImage){

    int graphSize = imageHeight * imageWidth * sizeof(float);
    int filterSize = filterWidth * filterWidth * sizeof(float);
    float *result_ptr;
    float *filter_ptr;
    float *input_ptr;

    cudaMalloc((void**) &result_ptr, graphSize);
    cudaMalloc((void**) &input_ptr, graphSize);
    cudaMalloc((void**) &filter_ptr, filterSize);

    cudaMemcpy(input_ptr, inputImage, graphSize, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_ptr, filter, filterSize, cudaMemcpyHostToDevice);

    int blk_x = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blk_y = (imageWidth  + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid(blk_x, blk_y);

    convolution<<<blocks_per_grid, threads_per_block>>>(input_ptr, result_ptr, filter_ptr, imageHeight, imageWidth, filterWidth, filterWidth / 2);
    cudaMemcpy(outputImage, result_ptr, graphSize, cudaMemcpyDeviceToHost);
    cudaFree(result_ptr);
    cudaFree(input_ptr);
    cudaFree(filter_ptr);
}