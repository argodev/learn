# Chapter 1 - Introduction

> NOTE: There are no code examples/exercises for this chapter.

It is hard to accurately review the intro to a book prior to completeing the entire book (was it adequate, complete, properly foundational, etc.), but at this point, it seems to be well done. The following are some points that were particularly interesting to me as I worked through this portion of the text:

- In a heterogeneous computing platform, the goal is to map each task to the best available processor for the given computation.
- "No single device is best for running all classes of workloads". This is obvious in context of this book, but I was glad to see it called out. Even the all-mighty GPU is not a panecea for all problems. It becomes a tool among others.
- They provide a decent overview of the types of parallelism and some decent figures explaining how it should work (e.g. 1.1, 1.2, 1.4 and 1.5)
- Due to my work with text analytics, I appreciated the text-centric examples
- I thought that the section dealing with the difference between concurrency and parallelism was interesting. They stress the difference between the machine doing two things at the same *general* time (concurrency) and the machine doing two things at the *same exact instant in time* (parallelism).
- Prior to reading this chapter, I was unaware that OpenCL had any applicability to FPGAs - I thought it only targeted GPUs.

They then proceed with a discussion of memory sharing approaches and then summarize each of the subsequent chapters.

I should note that I also pulled down the univeristy class materials for this book and have been working through them along with the text. It was interesting when reviewing chapter 1 that there were a few things I pulled out of the slides that either weren't in the text or I completely missed. One of those was the notion of "Loop Strip Mining" which, sounded much fancier than it actually is. A bit of digging online resulted in the following definition:

> Loop strip mining is a loop-transformation technique that partitions the iterations of a loop so that multiple iterations can be:
> - executed at the same time (vector/SIMD units),
> - split between different processing units (multicore CPUs, GPUs), or both.
>
> (https://www.cvg.ethz.ch/teaching/2011spring/gpgpu/Lec1-intro.pdf)

Essentially, if I have four processors, and have a loop of 1..N, Loop strip mining has processor #1 work on 1..N/4, #2 work on 1+N/4..(N/4)*2, etc.

There were a couple of key take-aways at the end of the slides for chapter 1 that bear listing here (only slightly paraphrased):

- often, for loops can map directly to OpenCL work items, however significant performance improvements may depend on a thorough understanding of the hardware
- codes that require global synchronization are generally not well suited to the GPUs

[<< Previous](../Chapter_00/README.md)
|
[Next >>](../Chapter_02/README.md)