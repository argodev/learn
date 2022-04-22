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

    // create event variables for timing
    cudaEvent_t startMemcpy, stopMemcpy;
    cudaEvent_t startKernel, stopKernel;
    cudaEventCreate(&startMemcpy);
    cudaEventCreate(&stopMemcpy);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);

    float *d_in = 0;
    float *d_out = 0;
    cudaMalloc(&d_in, len*sizeof(float));
    cudaMalloc(&d_out, len*sizeof(float));

    // record the event that "starts the clock" on data transfer
    cudaEventRecord(startMemcpy);

    // copy input data from host to device M times
    for (int i = 0; i < M; ++i) {
        cudaMemcpy(d_in, in, len*sizeof(float), cudaMemcpyHostToDevice);
    }

    // record the event that "stops the clock" on data transfer
    cudaEventRecord(stopMemcpy);

    cudaEventRecord(startKernel);
    distanceKernel<<<len/TPB, TPB>>>(d_out, d_in, ref);
    cudaEventRecord(stopKernel);

    cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);

    // ensured timed events have stopped
    cudaEventSynchronize(stopMemcpy);
    cudaEventSynchronize(stopKernel);

    // Convert event records to time and output
    float memcpyTimeInMs = 0;
    cudaEventElapsedTime(&memcpyTimeInMs, startMemcpy, stopMemcpy);
    float kernelTimeInMs = 0;
    cudaEventElapsedTime(&kernelTimeInMs, startKernel, stopKernel);

    printf("Kernel time (ms): %f\n", kernelTimeInMs);
    printf("Data transfer time (ms): %f\n", memcpyTimeInMs);

    cudaFree(d_in);
    cudaFree(d_out);
}
