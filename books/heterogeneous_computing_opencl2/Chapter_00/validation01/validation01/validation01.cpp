// validation01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<stdio.h>
#include<CL/cl.h>

int main()
{
    cl_int err;
    cl_uint numPlatforms;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (CL_SUCCESS == err)
        printf("\nDetected OpenCL platforms: %d", numPlatforms);
    else
        printf("\nError calling clGetPlatformIDs. Error code: %d", err);

    printf("\n");

    return 0;
}

