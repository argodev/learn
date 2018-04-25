# Chapter 2: CUDA Essentials

This was a relatively short chapter that introduces the reader to the bare-minimum CUDA concepts. If, by this point, you have gone through the appendicies, you have probably inferred most of this knowlege from the sample code written.

The authors start by explaining (in appropriately broad strokes) the architecural differences between a CPU and a GPU. They introduce terms such as `host` and `device` in the CUDA context and show how the devices are __SIMT__ (Single Instruction Multiple Thread) oriented. They briefly introduce the concepts of cores, ALUs, FPUs, streaming multi-processors (SMs), threads, blocks, and warps. I expect the specifics of each of these will be emphasized (at least implicitly) as we progress through the book.

Next is a discussion of the `kernel` - the CUDA version of a minature program that is exposed as a function or method to the calling application. I particularly liked the way they described it's use: _"We launch a kernel to create a computational grid composed of blocks of threads (or threadblocks)"_. This helps put the terms in context.

CUDA API and C Language extensions are covered next and the `<<< >>>` symbols are introduced as the means of passing execution configuration to the device for a given kernel call. Also discussed are function decorators such as `__global__`, `__host__`, and `__device__`.

The chapter wraps up with a brief touch on dimension variables, memory accesses CUDA-specific types, and some additional functions such as memory copy operations (`cudaMalloc()`, `cudaMemcpy()`, `cudaFree()`) and synchronization methods (`__syncThreads()` and `cudaDeviceSynchronize()`). A few comments regarding Dynamic Parallelisim are also included for context.

There are no code samples/exercises in this chapter.

[<< Previous ](../Chapter_01/readme.md)
|
[ Next >>](../Chapter_03/readme.md)
