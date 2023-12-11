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

    // TODO: MPI init
    pi_result = 0.0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int toss_per_rank = tosses / world_size;
    long long int remain = tosses % world_size;
    double received_data[world_size - 1];
    if (world_rank > 0)
    {
        // TODO: MPI workers
        double part_result;
        part_result = monteCarlo(seed, toss_per_rank);
        MPI_Send((&part_result), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size - 1]; 
        for(int i = 1; i < world_size; i++){
            MPI_Irecv(&received_data[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }

        pi_result = monteCarlo(seed, toss_per_rank + remain);

        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        for(int i = 1; i < world_size; i++){
            pi_result += received_data[i];
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
