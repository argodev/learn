// histogram.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <CL/cl.h>

#include "utils.h"
#include "bmp-utils.h"

static const int HIST_BINS = 256;

using namespace std;

int main()
{
	// Host Data
	int *hInputImage = NULL;
	int *hOutputHistogram = NULL;

	// allocate space for the input image and read the 
	// data from disk
	int imageRows;
	int imageCols;
	hInputImage = readBmp("../../../Images/cat.bmp", &imageRows, &imageCols);
	const int imageElements = imageRows * imageCols;
	const size_t imageSize = imageElements * sizeof(int);
	cout << "Finished Reading in the Image" << endl;

	// allocate space for the histogram on the host
	const int histogramSize = HIST_BINS * sizeof(int);
	hOutputHistogram = (int*)malloc(histogramSize);
	if (!hOutputHistogram) { exit(-1); }

	// use this to check the output of each API call
	cl_int status;

	// get the first platform
	cl_platform_id platform;
	status = clGetPlatformIDs(1, &platform, NULL);
	check(status);

	// get the first device
	cl_device_id device;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	check(status);

	// create a context and asociate it with the device
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	check(status);

	// create a command-queue and associate it with the device
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, device, 0, &status);
	check(status);

	// create a buffer object for the input image
	cl_mem bufInputImage;
	bufInputImage = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &status);
	check(status);

	// create a buffer object for the output histogram
	cl_mem bufOutputHistogram;
	bufOutputHistogram = clCreateBuffer(context, CL_MEM_WRITE_ONLY, histogramSize, NULL, &status);
	check(status);

	// write the input image to the device
	status = clEnqueueWriteBuffer(cmdQueue, bufInputImage, CL_TRUE, 0, imageSize, hInputImage, 0, NULL, NULL);
	check(status);

	// init the output histogram with zeros
	int zero = 0;
	status = clEnqueueFillBuffer(cmdQueue, bufOutputHistogram, &zero, sizeof(int), 0, histogramSize, 0, NULL, NULL);
	check(status);

	// create a program with source code
	char *programSource = readFile("histogram.cl");
	size_t programSourceLen = strlen(programSource);
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, &programSourceLen, &status);
	check(status);

	// build (compile) the program for the device
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		printCompilerError(program, device);
		exit(-1);
	}

	// create the kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, "histogram", &status);
	check(status);

	// set the kernel arguments
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufInputImage);
	status |= clSetKernelArg(kernel, 1, sizeof(int), &imageElements);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufOutputHistogram);
	check(status);

	// define the index space and work-group size
	size_t globalWorkSize[1];
	globalWorkSize[0] = 1024;

	size_t localWorkSize[1];
	localWorkSize[0] = 64;

	// enqueue the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	check(status);

	// read the output histogram to the host
	status = clEnqueueReadBuffer(cmdQueue, bufOutputHistogram, CL_TRUE, 0, histogramSize, hOutputHistogram, 0, NULL, NULL);
	check(status);

	// free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufInputImage);
	clReleaseMemObject(bufOutputHistogram);
	clReleaseContext(context);

	int i;
	int passed = 1;
	for (i = 0; i < HIST_BINS; i++) {
		cout << hOutputHistogram[i] << ",";
	}

	cout << endl;


	// free host resources
	free(hInputImage);
	free(hOutputHistogram);
	free(programSource);

    return 0;
}

