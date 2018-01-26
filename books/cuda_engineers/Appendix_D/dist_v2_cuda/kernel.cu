#include "kernel.h"
#include <stdio.h>
#include <time.h>
#define TPB 32
#define M 100       // number of times to do the data transfer

__device__
float distance(float x1, float x2) {
    return sqrt((x2-x1)*(x2-x1));
}

__global__
void distanceKernel(float *d_out, float *d_in, float ref) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const float x = d_in[i];
    d_out[i] = distance(x, ref);
}

void distanceArray(float *out, float *in, float ref, int len) {
    float *d_in = 0;
    float *d_out = 0;
    cudaMalloc(&d_in, len*sizeof(float));
    cudaMalloc(&d_out, len*sizeof(float));

    // record the clock cycle count before data transfer
    clock_t memcpyBegin = clock();

    // copy input data from host to device M times
    for (int i = 0; i < M; ++i) {
        cudaMemcpy(d_in, in, len*sizeof(float), cudaMemcpyHostToDevice);
    }

    // record the clock cycle count after memory transfer
    clock_t memcpyEnd = clock();

    clock_t kernelBegin = clock();
    distanceKernel<<<len/TPB, TPB>>>(d_out, d_in, ref);
    clock_t kernelEnd = clock();

    cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);

    // compute time in seconds between clock count readins
    double memcpyTime = ((double)(memcpyEnd - memcpyBegin))/CLOCKS_PER_SEC;
    double kernelTime = ((double)(kernelEnd - kernelBegin))/CLOCKS_PER_SEC;

    printf("Kernel time (ms): %f\n", kernelTime*1000);
    printf("Data transfer time (ms): %f\n", memcpyTime*1000);

    cudaFree(d_in);
    cudaFree(d_out);
}
