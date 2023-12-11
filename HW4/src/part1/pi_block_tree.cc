#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#pragma GCC optimize("Ofast", "unroll-loops")

double monteCarlo(unsigned int seed, long long int tosses){
    long long int numberInCircle = 0;
    for(int i = 0; i < tosses; i++){
        double x = (float) rand_r(&seed) / RAND_MAX*2 - 1;//-1 + (float) (rand_r(&seed)) / ( (float) (RAND_MAX/(1-(-1))));
        double y = (float) rand_r(&seed) / RAND_MAX*2 - 1;//-1 + (float) (rand_r(&seed)) / ( (float) (RAND_MAX/(1-(-1))));
        double distanceSquared = x * x + y * y;

        if(distanceSquared <= 1) numberInCircle++;
    }
    return 4 * numberInCircle / ((double) tosses);
}
int main(int argc, char **argv)
{
    unsigned int seed = time(NULL);
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int toss_per_rank = tosses / world_size;
    long long int remain = tosses % world_size;

    // TODO: binary tree redunction
    double part_result;
    if(world_rank > 0)
        part_result += monteCarlo(seed, toss_per_rank);
    else
        part_result += monteCarlo(seed, toss_per_rank + remain);

    for(int stride = 1; stride < world_size; stride *= 2){
        if(world_rank % (stride * 2) == 0){
            int brother =  world_rank + stride;
            if(brother < world_size){
                double received_pi;
                MPI_Recv(&received_pi, 1, MPI_DOUBLE, brother, 0, MPI_COMM_WORLD, &status);
                part_result += received_pi;
            }
        } else {
            int parent = world_rank - stride;
            MPI_Send(&part_result, 1, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD);
            break;
        }
    }
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = part_result / world_size;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
