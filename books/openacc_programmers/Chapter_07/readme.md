# Chapter 7: OpenACC and Performance Portability

This chapter was written by a couple of colleagues of mine from here at ORNL and looks at the problem of writing performance-portable OpenACC code. Some of the chapters up to this point describe how you can specify various levels of parallelization (granularity) and this chapter discusses the pros and cons of some of those choices. In general, they recommend that you specify as little as possible and allow the compiler to make decisions as to where to place what. 

They also spend some time discussing the possiblity that OpenACC could be used entirely in place of OpenMP for portable code. This is interesting as OpenMP is gaining support for device offloading.

They wrap up this short chapter with an experiment and the presented results. They take a benchmark test suite (__HAACmk microkernel__), modify it for OpenACC and run it on a large scale system.

Their results show that the Cray-specific compiler performed the best when running on Cray system with multi-core processors and NVIDIA GPUs. The PGI compiler also performed well and both were much better than the OpenMP variant. They then compiled and ran the code only for CPU multicore and still show a speedup over the OpenMP version, though not nearly as pronounced. They postulate that this is due to the OpenACC directives providing more information regarding possible vectorization to the compiler.



[<< Previous](../Chapter_06/readme.md)
|
[Next >>](../Chapter_08/readme.md)
