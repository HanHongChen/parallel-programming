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
        double x = (float) rand_r(&seed) / RAND_MAX*2 - 1;
        double y = (float) rand_r(&seed) / RAND_MAX*2 - 1;
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

    // TODO: init MPI
    pi_result = 0.0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int toss_per_rank = tosses / world_size;
    long long int remain = tosses % world_size;
    if (world_rank > 0)
    {
        // TODO: handle workers
        double part_result = 0.0;
        part_result += monteCarlo(seed, toss_per_rank);
        // printf("我是%d，我算出來的結果是%f\n", world_rank,pi_result);
        MPI_Send((&part_result), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        pi_result += monteCarlo(seed, toss_per_rank + remain);
        // printf("我是%d，我算出來的結果是%f\n", world_rank,pi_result);
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        
        for(int i = 1; i < world_size; i++){
            double pi_result_others;
            MPI_Recv((&pi_result_others), 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            pi_result += pi_result_others;
        }
        pi_result /= world_size;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
