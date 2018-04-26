# Chapter 8: Using CUDA Libraries

The goal of this chapter is to introduce the reader to a number of pre-built libraries that leverage the GPU (and CUDA) to perform a number of common activities. When I started studying GPUs back in 2010, most everything had to be hand-rolled and, the majority of this book has, up to this point, been focused on low-level CUDA programming. Since then however, many of the more common computational problems have been ported to CUDA and can be utilized from C/C++ without any need to write your own kernel.

## Thrust

Thrust is a C++ standard library-like platform that exposes a number of methods and "primatives" that can be used in your application with very little work. Working through the examples in this section provides a clear introduction to these tools and methods.

### Computing NORMs with inner_product()

This example is a simplistic introduction to Thurst showing the `inner_product()` method and the `device_vector()` data structures.

###  Computing Distances with transform()

Building on the prior example this application introduces `host_vector()`, sequences, and transforms. It also introduces the concept of __Functors__.

### Computing distances with lambda expressions

This example is the same as the prior but, rather than using using a functor, it makes use of lambda expressions. I found this particularly interesting as lambdas have gained significant traction in the broader programming community and providing workable examples of them in CUDA serves as another touch point for developers between languages they know and those they are learning.

### Estimating PI with generate(), transform() and reduce()

As you might guess from the title above, this example uses and approximation method to yield the value of PI. In prior examples we have seen `transform()` but this example builds on that and introduces `generate()` and `reduce()`.

### Estimating PI with fusion

This example does the same as the prior, but performs all of the parallel computation in a single fused statement. This results in fewer transitions between the host and the CPU often resulting in significantly improved total performance

### More Fusion: Centroid Calculation

This example is a simple modification of the centroid application shown in Chapter 6 but utilizing fusion and Thrust. The code is quite smaller than what was needed for the Chapter 6 version and the results are similar (see image below). I did not, however, profile the two to compare runtime speed.

![Washington State Output](wa_state_out.bmp)


## cuRAND



## NPP (NVIDIA Performance Primitives)


## cuSOLVER

## cuBLAS



## cuDNN

## ArrayFire




[<< Previous](../Chapter_07/readme.md)
|
[Next >>](../Chapter_09/readme.md)
