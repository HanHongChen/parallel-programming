#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

//有cl_device_id、cl_context、cl_program=>已經初始化過
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    
    // printf("進入了hostFE\n");
    cl_int status;
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    size_t imageSize = imageHeight * imageWidth * sizeof(float);
    int halfFilterWidth = filterWidth / 2;
    //設定Command Queue
    cl_command_queue myqueue = clCreateCommandQueue(*context, *device, 0, &status);
    // checkOpenCLError(status, "clCreateCommandQueue");
    // printf("建立commandQ\n");
    //設定Buffer
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        imageSize, inputImage, &status);
    // checkOpenCLError(status, "clCreateBuffer for input");
    // printf("建立inputBuffer\n");

    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
        imageSize, NULL, &status);
    // checkOpenCLError(status, "clCreateBuffer for output");
    // printf("建立outputBuffer\n");

    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR,
        filterSize, filter, &status);
    // checkOpenCLError(status, "clCreateBuffer for filter");
    // printf("建立filterBUffer\n");

    //設定kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    // checkOpenCLError(status, "clCreateKernel");
    // printf("建立kernel\n");

    //設定kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 4, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
    clSetKernelArg(kernel, 6, sizeof(int), &halfFilterWidth);
    //execute kernel
    size_t globalSize[2] = {imageWidth, imageHeight };
    // size_t globalGroupSize[2] = {imageHeight * imageWidth / 4, 1};
    // int grid_size = imageHeight * imageWidth / 4;
    // size_t globalSize[2] = { grid_size, 1 };
    clEnqueueNDRangeKernel(myqueue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    // printf("enqueueNDRangeKernel\n");
    
    clFinish(myqueue);
    // printf("finish\n");
    clEnqueueReadBuffer(myqueue, outputBuffer, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);
    // printf("readBuffer\n");
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    // printf("釋放mem\n");
    clReleaseKernel(kernel);
    // printf("釋放kernel\n");
}

//     __kernel void convolution(__global float *inputImage,
                            // __global float *outputImage,
                            // __global float *filter,
                            // int filterWidth, 
                            // int imageHeight, 
                            // int imageWidth)
//     int halffilterSize = filterWidth / 2;
//     const int gidX= get_global_id(0);
//     const int gidY = get_global_id(1);

//     float sum = 0.0;

//     for(int k = -halffilterSize; k <= halffilterSize; k++){

//         for(l = -halffilterSize; l <= halffilterSize; l++){
//             int currentX = gidX + i;
//             int currentY = gidY + j;

//             // Check boundaries
//             if (currentX >= 0 && currentX < imageWidth && currentY >= 0 && currentY < imageHeight)
//             {
//                 int inputIndex = currentY * imageWidth + currentX;
//                 int filterIndex = (j + halfFilterSize) * filterWidth + (i + halfFilterSize);

//                 sum += inputImage[inputIndex] * filter[filterIndex];
//             }

//         }
//     }
//     outputImage[gidY * imageWidth + gidX] = sum;