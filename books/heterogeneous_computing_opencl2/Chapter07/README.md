# Chapter 7 - OpenCL Device-Side Memory Model

## Synchronization and Communication
Remembering that all work items within a work group may execute in full parallel, shared memory accesses can be troublesome. The specification allows for "Barriers" which are essentially functions that can be called from within a kernel. The runtime then requires all kernels in that group to reach the barrier prior to allowing any to proceed. This feels like one of those bits of functionality that you don't need until you really need it, and when you do, you won't be able to accomplish your task without it. In the present, however, it seems to not be all that common of a use case.

## Global Memory
The section on global memory was somewhat uninteresting after the previous chapter. It defines global memory as, well, memory that is globally readable to all kernels executing on the device. The text then re-discusses the concepts of buffers, images, and pipes that were explained in chapter 6. There is a little more discussion on the use of the image memory type, but otherwise it seems like repeat material.

## Constant Memory
This is a feature of the spec that allows for a carving out of some of global memory to hold smaller, constant-type data. This is supported logically across all implementations, but the physical implementation may differ greatly. The text references the AMD GPUs which utilize a specialized cache with much lower latency than L1 cache for this data. 

## Local Memory
Local memory is local to the work group unit and is visible to all kernels executing within the workgroup

## Private Memory
Private memory is just that... private to the currently executing kernel. The text does point out, however, that the size of private memory and implementation support varies widely by device and over-use can cause undesired results. 

## Memory Ordering
This last portion of the chapter deals with the levels of support within the runtime for synchronoization and locks when accessing shared memory. Specifically, it details a number of semantics for describing the level of (degree of) consistency required. Based on the logic of your application, various options may be appropriate. These include relaxed, acquire, release, acquire-release, and sequential. 
