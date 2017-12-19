// 3_7_Vector_Addtion_With_Cpp.cpp : Defines the entry point for the console application.
// from pages 69-71, Heterogeneous Computing with OpenCL 2.0
//
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <CL/cl.hpp>



int main() {
	const int elements = 2048;
	size_t datasize = sizeof(int)*elements;

	int *A = new int[elements];
	int *B = new int[elements];
	int *C = new int[elements];

	for (int i = 0; i < elements; i++ ){
		A[i] = i;
		B[i] = i;
	}

	try {
		// query for platforms
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);

		// get a list of devices on this platform
		std::vector <cl::Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

		// create a context for the device
		cl::Context context(devices);

		// create a command-queue for teh first device
		cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

		// create the memeory buffers
		cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
		cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
		cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, datasize);

		// copy the input data to the input buffers using the command-queue for the first device
		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, datasize, A);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, datasize, B);

		// read the program source
		std::ifstream sourceFile("vector_add_kernel.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		// create the program from the source code
		cl::Program program = cl::Program(context, source);

		// build the program for the devices
		program.build(devices);

		// create the kernel
		cl::Kernel vecadd_kernel(program, "vecadd");

		// set the kernel arguments
		vecadd_kernel.setArg(0, bufferA);
		vecadd_kernel.setArg(1, bufferB);
		vecadd_kernel.setArg(2, bufferC);

		// exeucte the kernel
		cl::NDRange global(elements);
		cl::NDRange local(256);
		queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, global, local);

		// copy the data back to the host
		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, datasize, C);

	}
	catch (cl::Error error) {
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}

