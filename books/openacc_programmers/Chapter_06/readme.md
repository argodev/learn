# Chapter 6: Best Programming Practicies

This chapter does a great job summarizing some basic (mildly obvious) programming best practices and then ends with a real-world example that walks the reader through the process (this is great). The author acknowledges that providing an example in a book such as this is hard to get beyond _toy_ scenarios, but he does his best to provide a representative sample. The chapter leads out with his objectives for readers to learn:

- The necessity of baseline profiling
- The strategy of incremental acceleration and verification
- Techniques to maximize on-device computation
- Techniques to minimize data transfer and improvde data locality

He then walks through this process and explains that most of these "best practices" are not unique to OpenACC and really apply to all heterogeneous computing optimizations. In his overview section he gives a brief description of what he means by maximizing on-device computation and optmizing data locality. He then digs much deeper into both of these topics in sections 6.2 and 6.3

Most of the material in this section is somewhat obvious but he points out a couple of things that seemed noteworthy to me:

1. The use of `atomic`: He suggests that there are many times wherein loops update a shared varible and might not be considered data parallel. In these cases, he recommends testing with `#pragma acc atomic` before the updates (much like you would do a lock) and test its affects - you may see perf improvements while still ensuring accuracy.
1. Wrapping various directives with the `if` clause and peforming runtime tuning. He describes a use case wherein you may be unsure of the appropriateness of sending code to the device given the current hardware usage. In these scenarios, you can run a quick (relatively) runtime test to determine the best place to run your code and then set a boolean flag that can guard any of the directives and control where the code is run. 
1. As before, this author mentions the `present` clause which can intelligently indicate that code should be run on the device only if the data is already present. 
1. He describes data clauses and unstructured data lifetimes which can be helpful in bounding when data is moved to/from the device.
1. Finally, there is a discussion of array shaping - the notion that you only move the portion of the array data needed to the device and back again (sub-selections).

## Example Optimizations
The chapter ends with an example wherein the author walks you through the entire process previously discussed. He utilizes a theromdynamic table lookup code and briefly explains what it does and how it works. He has you compile it and then walk throug the following steps:

### Profiling

Just as described many places in the book, you have to be good at profiling and therefore he describes how you go about doing it with the example application. In this scenario, the `bisection()` method is the hotspot.

### Acceleration with OpenACC

The next step in the process is to add acceleration directives. He shows adding data movement directives as well as loop parallelization commands. He then re-runs the tool having compiled it both for CPU, Multi-Core CPU, and CPU+GPU. At this point, the CPU+GPU code is significantly faster than the single-core CPU code, but only slightly faster than the multi-core CPU code.

### Optimized Data Locality

Once the logic has been parallelized, the focuses on where data lives, where it needs to be, and how we can optimize the code accordingly. He adds a few data directives as well as a data enter/exit region. The code is recompiled and, at this point, the CPU+GPU code clearly outshines either of the other options.

### Data-Dependent Optimizations

Finally, he walks through testing the code with different data sizes and shows that there are clear benefits to running on the device once the data set grows beyond a certain size, but performance is degraded below that point. He uses this as a discussion point for using an `if` clause to only parallelize when the input data is beyond a certain scale.


[<< Previous](../Chapter_05/readme.md)
|
[Next >>](../Chapter_07/readme.md)
