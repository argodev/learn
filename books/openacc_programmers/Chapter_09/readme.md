# Chapter 9: OpenACC and Interoperability

This is a rather short chapter that deals with an important topic - how to OpenACC allows you to interact between code it generates and manages and existing CUDA-based libraries. There are significant implications both for CUDA calling OpenACC, OpenACC calling CUDA, and data movement/references in both scenarios. 

The authors start by explaining the problem domain and providing the motivation for addressing the topic. They then dive in to a discussion of how to appropriately call native device code (e.g. CUDA libraries) from within OpenACC. 

The next section flips the equation and discusses (very briefly - about half a page) on calling OpenACC code from native code. They then wrap up with a few topics on data movement and other advanced topics. 

There wasn't alot of code in this chapter, however the authors did provide a sample that used cuFFT to do edge detection on an image and to illustrate the concepts shown in this chapter. I did compile and run the examples provided using the PGI/ACC/CUDA combination (`1.041 seconds`) but was unable to run the other options (serial and openmp) due to their dependencies on the Intel MKL.



[<< Previous](../Chapter_08/readme.md)
|
[Next >>](../Chapter_10/readme.md)
