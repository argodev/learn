#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <CL/cl.h>

#include "utils.h"
#include "bmp-utils.h"

using namespace std;

int main(int argc, char **argv)
{
	/* host data */
	float *hInputImage = NULL;
	float *hOutputImage = NULL;

	/* angle for rotation (degrees) */
	const float theta = 45.0f;

	/* allocate space for the input image and read the data from disk */
	int imageRows;
	int imageCols;
	hInputImage = readBmpFloat("../../../Images/cat-face.bmp", &imageRows, &imageCols);
	const int imageElements = imageRows * imageCols;
	const size_t imageSize = imageElements * sizeof(float);

	/* allocate space for the output image */
	hOutputImage = (float*)malloc(imageSize);
	if (!hOutputImage) { exit(-1);  }

	/* use this to check the output of each API call */
	cl_int status;

	/* get the first platform */
	cl_platform_id platform;
	status = clGetPlatformIDs(1, &platform, NULL);
	check(status);

	/* Get the first device */
	cl_device_id device;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	check(status);

	/* create a context and associate it with the device */
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	check(status);

	/* create a command-queue and associate it with the device */
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueueWithProperties(context, device, 0, &status);
	check(status);

	/* The image descriptor describes how the data will be stored in memory.
	 * This descriptor initializes a 2D image with no pitch */
	cl_image_desc desc;
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = imageCols;
	desc.image_height = imageRows;
	desc.image_depth = 0;
	desc.image_array_size = 0;
	desc.image_row_pitch = 0;
	desc.image_slice_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
	desc.buffer = NULL;

	/* The image format describes the properties of each pixel */
	cl_image_format format;
	format.image_channel_order = CL_R; // single channel
	format.image_channel_data_type = CL_FLOAT;

	/* Create the input image and initialize it using a pointer to the image data on the host */
	cl_mem inputImage = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, NULL);

	/* create the output image */
	cl_mem outputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, NULL);

	/* copy the host image data to the device */
	size_t origin[3] = { 0, 0, 0 }; // Offset within the image to copy from
	size_t region[3] = { imageCols, imageRows, 1 }; // elements per dimension
	clEnqueueWriteImage(cmdQueue, inputImage, CL_TRUE, origin, region, 0, 0, hInputImage, 0, NULL, NULL);

	/* create a program with source code */
	char *programSource = readFile("image-rotation.cl");
	size_t programSourceLen = strlen(programSource);
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, &programSourceLen, &status);
	check(status);

	/* build/compile the program for the device */
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		printCompilerError(program, device);
		exit(-1);
	}

	/* create the kernel */
	cl_kernel kernel;
	kernel = clCreateKernel(program, "rotation", &status);
	check(status);

	/* set the kernel arguments */
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
	status |= clSetKernelArg(kernel, 2, sizeof(int), &imageCols);
	status |= clSetKernelArg(kernel, 3, sizeof(int), &imageRows);
	status |= clSetKernelArg(kernel, 4, sizeof(float), &theta);
	check(status);

	/* define the index space and work-group size */
	size_t globalWorkSize[2];
	globalWorkSize[0] = imageCols;
	cout << imageCols << endl;
	globalWorkSize[1] = imageRows;
	cout << imageRows << endl;

	size_t localWorkSize[2];
	localWorkSize[0] = 4; // 8;
	localWorkSize[1] = 4; // 8;

	/* enqueue the kernel for execution */
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	check(status);

	/* read the output image buffer to the host */
	status = clEnqueueReadImage(cmdQueue, outputImage, CL_TRUE, origin, region, 0, 0, hOutputImage, 0, NULL, NULL);

	/* write the output image to file */
	writeBmpFloat(hOutputImage, "../../../Images/rotated-cat.bmp", imageRows, imageCols, "../../../Images/cat-face.bmp");

	/* free OpenCL resources */
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(inputImage);
	clReleaseMemObject(outputImage);
	clReleaseContext(context);

	/* Free host resources */
	free(hInputImage);
	free(hOutputImage);
	free(programSource);

	return 0;
}

