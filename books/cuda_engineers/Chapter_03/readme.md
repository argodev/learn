# Chapter 3: From Loops to Grids

Having been through the appendicies, a first glance at this chapter causes one to think that most of the material is redundant. We are going to end up with the parallelized versions of the `dist_v1` and `dist_v2` programs and run them on the GPU. You may recall that we did this already in [Appdenix D](../Appendix_D/readme.md). While this is mostly true, there are a few twists in the program code and, more importantly, there are some pointers in the text that are worth highlighting.

## Parallelizing `dist_v1`
As you work through the code for `dist_v1_cuda` in the text, the authors call out the topic of __Execution Configuration__: The configuration of threads/block, divsision of labor across the available SM (streaming multiprocessors, etc). This topic can be thorny and they hint that, while they are making some assumptions at this point, the topic will be dealt with in more detail in subsequent sections.

A quick compile and run of the application shows it successfully executes as expected. Consistent with most introduction-to-parallel-computing texts, they highlight the fact that the results of the printf statements appear to be out-of-order due to the individual cores returning as they finished.

## Parallelizing `dist_v2`
Whereas `dist_v1` showed how to take an existing application and run a portion of it on the gpu, that version made poor use of the device. By contrast, `dist_v2` does a better job in following the more traditional mores. Specifically, v2 copies a bunch of input data to the card, runs a number of caluclations there, and then returns the results. The interconnection bus between the CPU and GPU is often the chokepoint for these types of applications, and reducing the number of times your application needs to cross that boundary (and the amount of data that traverses it) is a good idea. The authors actually call this out as the _Standard Workflow_ and it bears repeating here:

- "copy your input data to the device _once_ and leave it there"
- "Launch a kernel that does a significant amount of work (so the benefits of massive parallelism are significant compared to the cost of memory transfers)"
- "Copy the results back to the host _only once_."

__Note__: There is a side-bar discussion in the text regarding the way one calculates the block size when initializing the kernel. The authors quite properly highlight the fact that, in integer arithmetic, 64/32 = 2 but so also 65/32 = 2. This matters in that it is easy to miss and can result in the "extra" part of the data never being processed by the kernel. A common approach to sovling this is forcing the kernel to over-allocate space (blocks) to pick up any remainder. One way to accomplish this is rather than using `N/TPB`, you could use `(N+TPB-1)/TPB` which ensures that you will always allocate slightly more space than needed.

## Introduction to `cudaMallocManaged()`

The chapter ends up with another visit to the `dist_v2` application to introduce the `cudaMallocManaged()` function. This is a means of allocating data once that is accessible both to the host and the device. It is important to clarify that, at this point, this is primarily syntactic sugar over the move operations we have seen previously. The key difference is that the data is moved back and forth with far less code-based ceremony. As with many "optimizations", you may choose to ignore this option and take explicit control of when data is being moved back and forth (due to performance considerations). It _does_ however, greatly simplify the reading of the code.

[<< Previous ](../Chapter_02/readme.md)
|
[ Next >>](../Chapter_04/readme.md)
