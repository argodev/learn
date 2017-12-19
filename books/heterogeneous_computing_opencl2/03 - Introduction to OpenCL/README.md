# Chapter 3 - Introduction to OpenCL
This chapter, as you might surmize from the title, introduces OpenCL and gives 
the reader the background necessary to begin coding with it. I think a 
particularly important point was made at the bottom of the first page:

> The model set forth by OpenCL creates portable, vendor- and device-independent
programs that are capable of being accelerated on many different hardware 
platforms.

## Platform Model
- defines that an OpenCL envrionment has a host processor that serves and 
orchestration role
- there exist one or more devices (accelerators, etc.) with one or more 
processing elements.
- key methods include:
  - `clGetPlatformIDs()` which is used to discover the available OpenCL 
  platforms on a given system.
  - `clGetDeviceIDs()` which is used to query the devices available to that 
  platform.

> an interesting side project would be to write your own version of clinfo from 
the AMD SDK to see if you understood the API and the data available.

## Execution Model
- Contexts: Abstract environment that coordinates memory management for kernel
execution.
  - `clCreateContext()`
- Command-Queues: The communication mechanism by which a host requests an 
action be performed by a device. Can be in-orer or out-of-order (the latter 
allows the runtime to best keep things busy but may need external coordination 
to ensure results get reassembled as needed).
  - `clCreateCommandQueueWithProperties()` creates a queue
  - `clEnqueueReadBuffer()` asks the device to send something to the host
  - `clEnqueueNDRangeKernel()` asks the device to perform some work
- Events: used to specify dependencies between commands
  - `clGetEventInfo()`
  - `clWaithForEvents()`
- device-side enqueuing: ability of the device to create its own queues and 
subsequent work without communcating with the host. These are out-of-order 
queues and the parent kernel is dependent on the child kernels. Meaning, the 
parent kernel is not marked as completed until the child kernels it spawned are 
also complete.

## Kernel Programming Model
One of the key take-aways from this section is that OpenCL encourages you to 
consider the lowest-level (granularity) of parallelism within your program. 
Rather than thinking about the cost of threads and therefore chunking your 
data (e.g. strip-mining) you can simply express it as low as possible and let
the runtime/card/etc. handle making decisions as to the right balance.

One of the key things that this section points out (and is demonstrated in the 
code samples for this chapter) is that OpenCL kernels are compiled _at runtime_. 
This means, that you embed them simply as strings within your larger C/C++ 
program. While this seemed odd to me at first, it makes sense as it allows the 
runtime to optimize the compiled versions of the kernel for the devices and 
their properties that are actually available in the system as it is being run.
Note, that you can create/stash the binaries created from the source for faster 
execution for future runs on the same system.
- `clCreateProgramWithSource()`
- `clCreateKernel()`
- `clCreateProgramWithBinaries()`

## Memory Model
OpenCL uses an abstraction for memory that allows developers to know what to 
expect and device manufacturers to map their actual memory constructs to those 
abstractions. OpenCL uses three memory objects: buffers, images and pipes.

- Buffers: similar to arrays in C and also similar to the result of `malloc()` 
in the same.
- Images: an abstraction that allows the device maker to store memory in the 
manner best suited to that device. As such, it is not directly addressable and 
the developer must use generalized methods to read and write from these 
objects (e.g. `read_imagef()` and `read_imageui()`).
- Pipes: memory structure for passing streams of data (e.g. `packets`) between 
the host and device.

Memory is defined as being either host memory or device memory. Host memory is 
exactly what it sounds like... memory that exists within the host machine 
whereas device memory is usually on the card or accellerator and only available 
to it. Within the device, OpenCL defines 4 different memory regions that are 
logically distinct and the implementations of each are left to the device
manufacutrer.
- Global Memory: available to all work items executing within a kernel
- Constant Memory: similar to global but is used as constants (write once, 
read many w/o synchronization or locks)
- Local Memory: shared by work items within a work group.
- Private Memory: private to an individual work item. fastest option.








## Code Samples

### Vector Addition
The code for this listing is in the `3_6_1_Vector_Addition` directory.


### Vector Addition using C++
The code for this listing is in the `3_7_Vector_Addition_With_Cpp` directory.

Unlike the prior entry which could be entered essentially word-for-word from the
text, this code block required a few additional steps:

- When I created my project, I accepted the defaults which included both SDL 
checking as well as pre-compiled headers. These are both MSFT additions and, 
while often helpful, are not critical to understanding what is going on with 
the code in this section. Additionally, with them enabled, the program will 
not properly compile. To disable them:

  - __Pre-Compiled Headers__: Fixing this involves both removing 
  `#include "stdafx.h"` from the top of your *.cpp file as well as going into 
  the properties of your application and disabling it. The path for this 
  setting is C/C++ --> Precompiled Headers --> Precompiled Header: Not Using 
  Precompiled Headers. Setting this removes the `/Yu` compile switch.
  - __SDL__: This is disabled within the project properties. The path for this 
  setting is C/C++ --> General --> SDL checks: No. This sets the command line 
  switch to `/sdl-`.

Additionally, if you look at this code, it loads source code for the kernel from
a file (`vector_add_kernel.cl`) which is not provided. The source code is 
identical to the string text provided in the prior listing. I have included this 
file in the code directory referenced above. 

_However_, this still doesn't quite work as I would have expected. Extrapolating
your kernel code to an individual text file makes alot of sense to me, however, 
if you are going to take this approach, you need to distribute your `*.cl` files
with your code in some way (simple text files, embedded resources, etc.). I 
chose the simplist approach and simply wanted them copied out to my build 
directory to live with the executable. Unfortunately, this is not an option that
is exposed in the Visual Studio UI for C++ projects. To make this work, you have
to right-click on the project name in Visual Studio and choose `Unload`. Then, 
you can right-click again and choose `Edit`. This brings up an XML file. Locate 
your `*.cl` files which will look something like the following:

```xml
<ItemGroup>
    <None Include="vector_add_kernel.cl" />
</ItemGroup>
```  

You need to tell msbuild to copy them to the output directory. This is 
accomplished by modifying the XML to something like the following:

```xml
<ItemGroup>
    <Content Include="vector_add_kernel.cl">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>
</ItemGroup>
```

Save the file and close it. Then, right-click on the project file and choose 
`Reload`. For all subsequent builds, the latest version of your `*.cl` files 
will be copied to your output directory as desired.