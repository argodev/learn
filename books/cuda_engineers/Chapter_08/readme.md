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

In prior examples we utilized the CPU for generating random numbers. In this example we use `cuRAND` to create pseudo-random numbers on the GPU. The code sample is a direct port of the _Estimaing Pi with fusion_ example above, but with the random number generator changed.


## NPP (NVIDIA Performance Primitives)
NPP is a collection of libraries that provide performance-optimized versions of common image and scientific operations. In this first example, we take the sharpening application from Chapter 5 and re-work it using NPP.

> NOTE: I ran into some issues getting this to build. Both the book sample and the NVIDIA documentation online suggested that I should reference part of this library by including `-lnppi_static` in my makefile. However, this didn't seem to work on my envrionment. As of version 9 of the CUDA SDK, they have split this into sub libraries and I needed to include a specific library (`-lnppif_static`) to get this sample to work.


![Butterfly](sharpening.png)

The text provides some code snippets for computing both the distance between two images as well as converting color scales. They have not assembled these into working samples and are, therefore, not included in this repository.


## cuSOLVER and cuBLAS
cuBLAS is the NVIDIA implementation of the standard _Basic Linear Algebra Subprograms (BLAS)_ library that is well known in the technical computing arena. cuSolver is the CUDA version of the _Linear Algebra Package (LAPACK)_. This example introduces the libraries in the form of solving a system of linear equations.

> NOTE: I again ran into some issues getting this to build properly. Both the book (and the web sample) had make files that were missing a dependency in the __LIBS__ property. The key was to add `-lcusparse_static` to the makefile and things proceeded fine from there.

## cuDNN

I was hoping the authors would dig in to cuDNN as I have a number of colleages that utilize this library, but alas, they simply mention it is important and point the reader to the NVIDIA site for more information.

## ArrayFire

Similar to their handling of cuDNN, the coverage of ArrayFire is simply to mention that it exists, that it may be more familiar to those coming from a Matlab background, and to point you to the documentation for more information.



[<< Previous](../Chapter_07/readme.md)
|
[Next >>](../Chapter_09/readme.md)
