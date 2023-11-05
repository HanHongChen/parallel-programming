#include <stdio.h>
#include <thread>
#include <string.h>

#include "CycleTimer.h"
#include <time.h>
typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

extern void writePPMImage(
    int* data,
    int width, int height,
    const char *filename,
    int maxIterations);

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{
    // clock_t start = clock();
    int q = args->height / args->numThreads;
    int startRow = args->threadId * q;
    int totalRows = q + (args->threadId == args->numThreads - 1 ? 0 : args->height % args->numThreads); 
    
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                     args->width, args->height, startRow, totalRows,
                     args->maxIterations, args->output);
    // clock_t end = clock();
    // double execTime = ((double)(end - start))/CLOCKS_PER_SEC;
    // printf("Thread %d, height = %d, width = %d,  execTime = %f,\n\
    //         x0 = %f, x1 = %f, y0 = %f, y1 = %f, execRow = %d~%d, \n", 
    //         args->threadId, args->height, args->width, execTime,
    //         args->x0, args->x1, args->y0, args->y1, startRow, startRow + totalRows);
    // if(args->threadId == 1) printf("---------------\n");
    
    // writePPMImage(args->output, args->width, args->height, "mandelbrot-thread1.ppm", args->maxIterations);
    // printf("Hello world from thread %d\n", args->threadId);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
