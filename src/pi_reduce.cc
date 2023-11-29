#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
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
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int toss_per_rank = tosses / world_size;
    long long int remain = tosses % world_size;
    double local_data;

    if(world_rank > 0){
        local_data = monteCarlo(seed, toss_per_rank);
    }else{
        local_data = monteCarlo(seed, toss_per_rank + remain);
    }


    // TODO: use MPI_Reduce
    MPI_Reduce(&local_data, &pi_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0)
    {
        // TODO: PI result
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
