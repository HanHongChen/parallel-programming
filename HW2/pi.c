#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
struct ThreadInfo{
    pthread_t threadId;
    long long int numOfTosses;
    int seed;
    // long long int numberInCircle;
    double result;
};
void *monteCarlo(void* arg){
    struct ThreadInfo *info = (struct ThreadInfo*) arg;
    unsigned int seed = info->seed;
    long long int numberInCircle = 0;
    long long int numOfTosses = info->numOfTosses;
    for(int toss = 0; toss < numOfTosses; toss++){
        double x = (float) rand_r(&seed) / RAND_MAX*2 - 1;//-1 + (float) (rand_r(&seed)) / ( (float) (RAND_MAX/(1-(-1))));
        double y = (float) rand_r(&seed) / RAND_MAX*2 - 1;//-1 + (float) (rand_r(&seed)) / ( (float) (RAND_MAX/(1-(-1))));
        double distanceSquared = x * x + y * y;

        if(distanceSquared <= 1) numberInCircle++;
    }
    // info->numberInCircle = numberInCircle;
    info->result = 4 * numberInCircle / ((double) info->numOfTosses);
}

void main(int argc, char* argv[]){
    int numOfPthread = atoi(argv[1]);
    long long int numOfTosses = atoll(argv[2]);
    struct ThreadInfo info[numOfPthread];
    long long int remain = numOfTosses % numOfPthread;
    long long int toss = numOfTosses / numOfPthread;

    for(int i = 0; i < numOfPthread; i++){
        info[i].numOfTosses = toss;
        if(i == numOfPthread - 1) info[i].numOfTosses += remain;
        info[i].seed = i;
    }

    for(int i = 0; i < numOfPthread; i++){
        pthread_create(&info[i].threadId, NULL, monteCarlo, (void*)&info[i]);
    }
    for(int i = 0; i < numOfPthread; i++){
        pthread_join(info[i].threadId, NULL);
    }
    double sum = 0;
    for(int i = 0; i < numOfPthread; i++){
        // printf("thread%d, seed = %d, numOfTosses = %lld, numberInCircle = %lld, result = %f\n", 
        //     i, info[i].seed, info[i].numOfTosses, info[i].numberInCircle, info[i].result);
        // sum += info[i].numberInCircle;
        sum += info[i].result;
    }
    // printf("PI = %f \n", 4*sum/ ((double)numOfTosses));
    printf("PI = %f\n", sum / numOfPthread);
}