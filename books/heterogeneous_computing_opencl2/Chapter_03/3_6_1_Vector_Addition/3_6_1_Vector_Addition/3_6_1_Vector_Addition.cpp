// 3_6_1_Vector_Addition.cpp : Defines the entry point for the console application.
// from pages 66-69, Heterogeneous Computing with OpenCL 2.0
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <CL/cl.h>

using namespace std;


// OpenCL kernel to perform an element-wise addition
const char* programSource =
"__kernel                                                    \n"
"void vecadd(__global int *A,                                \n"
"            __global int *B,                                \n"
"            __global int *C)                                \n"
"{                                                           \n"
"                                                            \n"
"   // Get the work-item's unique ID                         \n"
"   int idx = get_global_id(0);                              \n"
"                                                            \n"
"   // Add the corresponding locations of                    \n"
"   'A' and 'B' and store the result in 'C'.                 \n"
"   C[idx] = A[idx] + B[idx];                                \n"
"}                                                           \n"
;


int main() { 
    // This code executes on the OpenCL host

    // elements in each array
    const int elements = 2048;

    // comput the size of the data
    size_t datasize = sizeof(int)*elements;

    // allocate space for input/output of host data
    int *A = (int*)malloc(datasize); // input array
    int *B = (int*)malloc(datasize); // input array
    int *C = (int*)malloc(datasize); // output array

    // initialize the input data
    int i;
    for (i = 0; i < elements; i++) {
        A[i] = i;
        B[i] = i;
    }
    cout << A << endl;
    cout << B << endl;

    // use this to check the output of each API call
    cl_int status;

    // get the first platform
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);

    // get the first device
    cl_device_id device;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    // create a context and associate it with the device
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);

    // create a command-queue and associate it with the device
    cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, device, 0, &status);

    // allocate two input buffers and one output buffer for the three vecorts in the 
    // vector addition process
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

    // write data from the input arrays to the buffers
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);

    // create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);

    // build (compile) the program for the device
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // create the vector addition kernel
    cl_kernel kernel = clCreateKernel(program, "vecadd", &status);

    // set the kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // define an index space of work-items for execution.
    // a work-group size is not rquired but can be used.
    size_t indexSpaceSize[1], workGroupSize[1];

    // there are 'elements' work-items
    indexSpaceSize[0] = elements;
    workGroupSize[0] = 256;

    // execute the kernel
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);

    // read the device output buffer to the host output array
    status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

    cout << C << endl;

    // free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);

    // free host resources
    free(A);
    free(B);
    free(C);

    return 0;
}
