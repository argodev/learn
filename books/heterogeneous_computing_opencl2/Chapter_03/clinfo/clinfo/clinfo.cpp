// clinfo.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <CL/cl.hpp>

using namespace std;

int main()
{
	// query for platforms
	std::vector <cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::vector<int>::size_type size = platforms.size();

	cout << "Number of Platforms: " << size << endl;

	for (unsigned i = 0; i < size; i++) {
		cl::Platform platform = platforms[i];
		platform.
	}





    return 0;
}

