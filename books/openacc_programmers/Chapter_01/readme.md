# Chapter 1: OpenACC in a Nutshell

This chapter has already convinced me that I have made a good decision in my path of learning OpenCL, CUDA, and now OpenACC. The former are both lower-level, device-specific approaches (to one degree or another) whereas OpenACC is (notionally, at least) more about expressing your intent and letting the compiler handle the specific implementation.

OpenACC is an extension to both C/C++ and Fortran that utilizes compiler directives to express various parallelization options. One key aspect of this approach is that it allows for a unified codebase between those systems that have parallel accelerators and those that do not. At an initial glance, I am reminded of the CUDA-Specific Thrust library.

The author of this chapter posits (similar to others I have spoken with on this topic, including those leading the DARPA Page 3 initiative), that we are rapidly moving into an age wherein we need to let the machine generate optimized code based on our intentions/desires and that the inherent complexity of the systems we are desigining (extreme heterogeneity) is simply too much for any human to adequately exploit.

The goals of the chapter were to provide a brief introduction to the platform and to cover the basic syntax, parallel and kernel constructs, loop-level parallelisim, and data movement. These sections are summarized below:

## Syntax

The syntax portion of the chapter was rather short and simple which is appropriate due to the nature of the library. In general, the syntax is as follows (I'm focusing exclusively on the C/C++ variants of the code in this review):

```C
#pragma acc <directive> [clause[[,] clause] . . .]
```

There are three primary directive types which are focused on computation, data management, and synchronization. The computation directives indicate which sections of code are parallelizable and give the compiler additional information as to how to proceed. Data management directives are not always necessary (much is inferred from compiler directives) but give the programmer additional control (more-fine-grained) over the default assumptions that may be made regarding movment. The synchronization directives allow for task parallelisim (vs. data parallelism) and provide for synchronization barriers.

Of interest is the normal lack of a dependency on an OpenACC-specific header. In the normative case, you can decorate your code and compile it with no additional dependencies. You are, however, given the opportunity to have your code operate at a lower level by adding the `opencc.h` header file. Doing so, however, breaks your ability to compile on non-OpenACC aware compilers or to run on systems without the runtime.

## Mix of parallel and kernel constructs

There are two primary directives types used to mark your code. __Kernels__ are used to mark blocks of code for which the compiler is free to analyze for parallization opportunities and to generate device-specific code based on that analysis. The __Parallel__ directive is similar however it places the onus of paralleization on the developer.

## Loop-level parallelism

Similar to `parallel.for` and similar constructs in other languages, the `loop` directive informs the compiler that the contents of the loop are data-independent. It (the compiler) is then free to make whatever optimizations necessary for executing that loop in an efficient fashion.

The `routine` directive allows the author to indicate that any/all calls to a particular function should be executed on the device or run in a parallel fashion. There are additional options that allow the user to target a hand-rolled implementation (e.g. CUDA-specific) and to fall back to other approaches when such devices are unavaialble. While this starts to get very low-level and specific, the author stresses that this level of specificity is rare and only is needed in a few cases.

## Data Movement

The last category of directives deal with data movement. Initially these can be avoided and should be viewed as an optimization technique rather than a core development approach. Of particular interest to me were the `cache` directive and the partial data transfer options. The former is a simple indication to the compiler that the following data should be, where possible, moved to the fastest (closest) memory available. The latter is useful when referencing a large vector/array and you know that the code in question only needs a portion of it - you can indicate that only a slice of the data be moved.

Also, one other data clause caught my attention - the `present` clause. Unlike the `create` or `copy` clauses that move data in to the device or create it if necessary, the `present` clause indicates that a given function should only be run on a device if the data needed is _already_ present. This allows you to communicate that you know the given function is worth running on the device if the data is already there, but wouldn't be worth the cost of moving the data to the card/device just for the sake of this function. That is, in my estimation, a pretty powerful construct.


[<< Previous](../readme.md)
|
[Next >>](../Chapter_02/readme.md)
