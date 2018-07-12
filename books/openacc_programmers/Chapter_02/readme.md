# Chapter 2: Loop-Level Parallelism

This chapter focuses on the constructs available via OpenACC that support loop-level parallelism. The authors postulate that this is the most common form of parallelism, and OpenACC's constructs make it easty to fall into the pit of success. OpenACC takes a similar approach to OpenMP and OpenCL. However, when you approach many-core devices, there often exists multiple levels of parallelism and the parameters available to OpenACC directives support these levels.

The `kernels` construct (`#pragma acc kernels`) is the most brain-dead, simplistic approach to parallelizing your code. All you have to do is add the directive and the compiler will automatically analyze that section of code and, where possible, generate device-specific code and the necessary calls to that code. While this is an easy way to add parallelism, the compiler will only parallelize those blocks that it can garantee are safe. 

You can optionally add more hints for the compiler as to how to handle nested layers of parallelism:

- `gang`: the most corse-grained level of parallelism
- `worker`: defines how work is distributed within a gang
- `vector`: maps to the hardware's instruction-level parallelism

In each case, these are _optional_. The opinion of the authors is that you should not add these until you are positive that you need them (avoid pre-mature optimization). 

There were a few other constructs mentioned in the chapter that were interesting:

- `#pragma acc parallel loop collapse()`: Tells the compiler it can collapse `N` nested loops into 1
- `independent` is a clause you can use to tell the compiler that you guarantee a given block is data independent and that it should feel free to parallelize it
- `seq` is a clause that allows you to inform the compiler that a given section must be run sequentially
- `reduction()` is a handy clause you can use to provide hints that a given variable holds the reduced values from each iteration and the compiler can optimize appropriately. An example would be a loop that had contents similar to `total += data[i]` - you could decorate the loop with something similar to `#pragma acc parallel loop reduction(+:total)`. There are a number of different reduction operators available.



[<< Previous](../Chapter_01/readme.md)
|
[Next >>](../Chapter_03/readme.md)
