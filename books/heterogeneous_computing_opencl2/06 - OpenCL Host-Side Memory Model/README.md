# Chapter 6 - OpenCL Host-Side Memory Model
This chapter (and the remainder of the chapters in the book) was significantly shorter and deals with a few specifics of memory management, specifically on the host-side (CH 7 deals with device-side stuff).


## Memory Objects
There exist three types of memory objects available within OpenCL on the hosts: Buffers, Images, and Pipes. Buffers aren't realy worth discussing as they are functionally the same as buffers within standard C. 

Images are memory structures specifically designed to represent, well, images. They are multi-dimensional and are designed to specifically leverage the graphics-processing hardware usually available on GPUs.

Pipes are very much like what you might assume they are... they are FIFO structures that pass blocks of data called `packets` around.


## Memory Management
The key takeaway from this section is that memory should not be read until after the given command has indicated it has succeeded properly. Meaning, there are commands to queue a write of data and a read of data, but simply queueing them doesn't mean they have occured. One must confirm that the queued actions actually finished prior to attempting to utilitize the referenced data. You can use the blocking parameter to force the method to not return until after it has fully completed if desired.


## Shared Virtual Memory
New in OpenCL 2.0 is Shared Virtual Memory (SVM) which allows global memory to be mapped into the host's memory region. The big key here is that you can pass pointer-based values into OpenCL Kernels. There are concerns here regarding race conditions and, if you expect bi-directional access (e.g. both host and device writing to the same memory), you must handle deconflicting access.
