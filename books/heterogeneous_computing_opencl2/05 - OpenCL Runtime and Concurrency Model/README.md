# Chapter 5 - OpenCL Runtime and Concurrency Model

This chapter picked up on a handful of topics that had been indirectly exposed in prior chapters but previously given very little explanation. The topics discussed include Commands and Queuing, multiple command queues, the kernel execution domain, native and built-in kernels, and device-side queuing.

## Commands and the Queuing Model
The key take away on this section is that you never issue a command directly. Much like UI programming wherein you have a UI thread against which you request commands to be run asynchronosly, commands are submitted to queues which are then monitored and executed in the proper context at the proper time.

## Multiple Command queues
Command queues are unique to a given device. Therefore, if you want to access multiple different devices from the same host, you need to create a unqiue command queue - in the appropriate context - per device. 

However, you can also use multiple command queues for the *same* device. This can be helpful in scenarios wherein you want to overlap independent commands or to avoid out-of-order queues. 

## The Kernel Execution Domain: Work Items, Work-Groups and NDRanges
As discussed previously, a `kernel` in OpenCL terms is essentially a function that can be executed on a device. 

NDRange == *n*-dimensional range (1,2, or 3-D). This is a grid of work items.

Work Items are essentially individual calls, or sets of parameters that are passed to individual instances of a given kernel.

Work Groups are a way in which OpenCL cuts up the processing capabilities of a given device. These may be 1, 2, or 3 dimensional units. Some communication between concurrently running  kernels within a workgroup may be possible, but inter-workgroup communication is not allowed.

This section is detailed and quite important to understanding the way in which tasks are dispateched to the given devices. There were two specific points that stuck out while reviewing this section.

The first is a bit obvious but bears repeating:
> "An application that involves global communication across its execution space is usually inefficient to parallelize with OpenCL"

And the second is more of an optimization comment:
> "...an OpenCL work-group's size should be an even multiple of that device's SIMD width" (available via runtime as `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` which can be acquired via the `clGetKernelWorkGroupInfo()` function.


## Native/Built-In kernels
This section was/is a major section in the book, however I remain unsure as to why. The net-net is that there exist built-in functionality (some might call them libraries) in the form of native and built-in kernels. Native kernels allow some standard C-code functions - compiled with a regular compiler - to be called directly from the OpenCL envionment. Built-In kernels are just that, and are device and SDK specific. These are libraries of functionalty beyond the OpenCL standard that a given vendor may provide. As such, documentation for these are provided by the individual vendors. 

## Device-Side Queuing
In general, commands are submitted to queues via host-side code and then executed in the proper context by the device. New in OpenCL 2.0 is the notion of device-side queues, as well as the ability to interact with them from device-side code. This is helpful to control levels of paralellism without having to step back to the host. You can have a given parallel task that then, conditionally, creates sub task, and handle all of this on the device without returning to the host for orchestration.
