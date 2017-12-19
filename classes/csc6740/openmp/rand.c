#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <sodium.h>


double drand ( double low, double high );


int main() {
    //double t1;
    //t1 = rand();
    // int r;
    // r = rand();

    // printf("%i\n", r);
    // printf("%f\n", drand48());
    // printf("%f\n", drand48());
    // printf("%f\n", DBL_MAX);
    srand((unsigned int)time(NULL));

    double my_num = 0;
    for (int i = 0; i < 100000000; i++) {
        my_num = ((float)rand()/(float)(RAND_MAX)) * 1024;
    }

    printf("%f\n", my_num);
    // printf("%d\n", RAND_MAX);
    // printf("%f\n", ((float)drand48())*(float)RAND_MAX);


    // printf("%f\n", ((float)rand()/(float)(RAND_MAX)) * 1024);// * RAND_MAX);
}


