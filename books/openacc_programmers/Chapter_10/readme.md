# Chapter 10: Advanced OpenACC

This chapter focuses on two main topics. The first is task scheduling via asynchronous work queues. The discussion on work queues is highly interesting and provides a path for using constructs we are familar with in other master/worker scenarios on the GPU. Of particular interest is the discussion on __asynchronous waiting__ which, while it caught me off guard at first, makes quite a bit of sense and the fact that it is provided for in the framework is quite helpful. This led into a discussion of software pipelining and an example (source code is in the book's repo) showing how they interleaved data transfer and on-board computing to minimize the "downtime" of any given processor.

> NOTE: While I tried, I was unable to successfully run their software pipelining example. It appears to be an issue with the version of OpenCV I have installed on my machine... I may try it on another machine later.

## Multidevice Programming

The second half of the chapter focused on various ways to leverage multiple GPUs. I do not presently have access to a system wherein I can run multi-device code, however I still found this section interesting. Options include targeting multiple GPUs directly from within the OpenACC code, handling it via different OpenMP Processes (or other processes), or even using direct MPI calls that are GPU-aware. Each of these come with its own set of restrictions and challenges. It is nice to know that a wide variety of options exist should they be needed/desired.

[<< Previous](../Chapter_09/readme.md)
|
[Next >>](../Chapter_11/readme.md)
