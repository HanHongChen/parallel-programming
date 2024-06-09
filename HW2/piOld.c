
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
void *monteCarlo(void* arg){
	long long int numOfTosses = *(long long int*)arg;
    // printf("monteCarlo收到的Tosses數量 = %lld\n", numOfTosses);
    unsigned int s = (unsigned int) time(NULL);
    long long int numberInCircle = 0;
    for(int toss = 0; toss < numOfTosses; toss++){
        double x = -1 + (float) (rand_r(&s)) / ( (float) (RAND_MAX/(1-(-1))));
        double y = -1 + (float) (rand_r(&s)) / ( (float) (RAND_MAX/(1-(-1))));
        
        // double x = rand_r(&s) / RAND_MAX * 2 - 1;
        // double y = rand_r(&s) / RAND_MAX * 2 - 1;
        double distanceSquared = x * x + y * y;
        if(distanceSquared <= 1) numberInCircle++;
        // printf("toss = %d \n", toss);
        // printf("i = %d, x = %f, y = %f, distanceSquared = %f, numberInCircle = %lld\n", toss, x, y, distanceSquared, numberInCircle);
    }
    double *res = malloc(sizeof(double));
    // double res = 4 * numberInCircle / ((double) numOfTosses);
    *res = 4 * numberInCircle / ((double) numOfTosses);
    // printf("每個thread的結果 = %f\n", *res);
    // printf("位置 = %d\n", res);
    return (void *) res;
}
void main(int argc, char* argv[]){
    int numOfPthread = atoi(argv[1]);
    long long int numOfTosses = atoll(argv[2]);
    // printf("numOfPthread = %d\n", numOfPthread);
    // printf("numOfTosses = %lld\n", numOfTosses);
    pthread_t threads[numOfPthread];
    long long int remain = numOfTosses % numOfPthread;
    long long int tosses[numOfPthread];

    for(int i = 0; i < numOfPthread; i++){
        tosses[i] = numOfTosses / numOfPthread;
        if(i = numOfPthread - 1) tosses[i] += remain;
        printf("toss[%d] = %d\n" , i, tosses[i]);
    }
    
    
    for(int i = 0; i < numOfPthread; i++){
        // tosses[i] = (long long int)numOfTosses / numOfPthread;
        // 
        // printf("i = %d, tosses = %lld\n", i, tosses[i]);
        pthread_create(&threads[i], NULL, monteCarlo, (void*)&tosses[i]);
        
    }

    double sum = 0;
    for(int i = 0; i < numOfPthread; i++){
        void *result;
        pthread_join(threads[i], &result);
        // printf("收到 = %d\n", result);
        sum += *(double*)result;
        free(result);
    }
    printf("結果:%f", sum / numOfPthread);
    // printf("PI = %f\n", 4 * numberInCircle / ((double) numOfTosses));
    // printf("PI = %f\n", sum / (double)numOfPthread);
}