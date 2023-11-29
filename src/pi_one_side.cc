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
    MPI_Win win;
    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int toss_per_rank = tosses / world_size;
    long long int remain = tosses % world_size;
    double *gather;
    double local_data;
    if (world_rank == 0)
    {
        local_data = monteCarlo(seed, toss_per_rank + remain);
        // MPI_Alloc_mem(sizeof(double), MPI_INFO_NULL, &gather);
        gather = (double*) malloc(sizeof(double));
        *gather = 0.0;
        MPI_Win_create(gather, sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    }
    else
    {
        local_data = monteCarlo(seed, toss_per_rank);

        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);//沒有這個會出現錯誤:A process or daemon was unable to complete a TCP connection to another process
        //會將所有人的local_data加到process 0的gather
        MPI_Accumulate(&local_data, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, win);
        MPI_Win_unlock(0, win);
        // local_data[world_rank] = monteCarlo(seed, toss_per_rank);
        // printf("local_data[%d] = %f\n", world_rank, local_data[world_rank]);
        // //MPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win)
        // MPI_Put(&local_data[world_rank], 1, MPI_DOUBLE, 0, world_rank, 1, MPI_DOUBLE, win);

    }
    // MPI_Win_create(&pi_result, sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    // MPI_Accumulate(&local_data, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, win);

    
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);
    if (world_rank == 0)
    {
        // TODO: handle PI result
        // for(int i = 0; i < world_size; i++){
        //     pi_result += local_data[i];
        // }
        *gather += local_data;
        pi_result = *gather / ((double)world_size);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}