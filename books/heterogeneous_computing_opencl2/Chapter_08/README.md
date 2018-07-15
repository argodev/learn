# Chapter 8 - Dissecting OpenCL on a Heterogeneous System

This chapter begins to talk a bit more about the realities of using OpenCL in a heterogeneous environment. Specifically a CPU and GPU. The authors refresh the readers memories regarding some of the trade-offs. This seems to be one of those chapters that one could benefit from reading multiple times, especially when desiging a system/code that leverages multiple hardware architectures.

A few tips of interest:

When writing kernels that we want to achive high performance on a GPU, we need to:
  
- provide a lot of work for each kernel to dispatch
- match multiple launches together if the kernels are small

These both, of course, make significant sense when you think about it, particularly if you have done any sort of GPU programming in the past. In both cases, you are trying to overcome the overhead cost of getting info to the GPU and the general communication cost with the GPU. This communication is often the most expensive portion of the code so making the most of it is key.

Another, quite related tip is to handle memory between the devices much like you might if you were developing a hard drive. If your memory is in a straight line, and you need to access locations 3, 9, 15, and 17, you may be far better off to issue/return a single memory access from [3:17] rather than 4 reads/returns for 3, 9, 15, 17. This hints at the trade-space that must be considered. In this scenario, more data than needed is being returned, however the overhead from multiple calls across the bus is reduced to that of a single request/response. This latter optimization may well overwhelm the former (that of only returning exactly what is needed).

[<< Previous](../Chapter_07/README.md)
|
[Next >>](../Chapter_09/README.md)