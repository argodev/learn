#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    printf("Number of devices: %d\n", numDevices);
    for (int i = 0; i < numDevices; ++i) {
        printf("-------------------------\n");
        cudaDeviceProp cdp;
        cudaGetDeviceProperties(&cdp, i);
        printf("Device Number: %d\n", i);
        printf("Device Name: %s\n", cdp.name);
        printf("Compute Capability: %d.%d\n", cdp.major, cdp.minor);
        printf("Maximum threads/block: %d\n", cdp.maxThreadsPerBlock);
        printf("Shared memory/block: %lu bytes\n", cdp.sharedMemPerBlock);
        printf("Total global memory: %lu bytes\n", cdp.totalGlobalMem);
    }
}