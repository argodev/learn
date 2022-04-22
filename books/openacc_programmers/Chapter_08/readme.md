# Chapter 8: Additional Approaches to Parallel Programming

In some ways, this chapter would have been helpful earlier on in the book. The main objective is to compare/contrast OpenACC with other parallel methodologies. The authors begin by describing the categories of parallelism they are interested in (list below) followed by a brief introduction to each of the systems they are evaluating (also listed below). This is followed by a rather helpful cross-walk wherein they address each parallelism construct and show how the different tools/systems address/support that construct. 

They end the chapter with a real-world example based on a benchmark tool. While the authors provide code for this example, they have dependencies on the Intel compiler which I didn't have availble to me and therefore I didn't run the tests directly.

| Approach |
|:---------|
| Parallel Loops |
| Parallel Reduction |
| Tightly Nested Loops |
| Non-Tightly Nested Loops |
| Task Parallelism |
| Data Allocations |
| Data Transfers |


| Programming Models |
|:-------------------|
| OpenACC |
| OpenMP |
| CUDA |
| OpenCL |
| C++ AMP |
| Kokkos |
| RAJA |
| TBB |
| C++17 |
| FOrtran 2008 |

The authors for this chapter to not specify/indicate a "winner" - and in fact, they seem to avoid that desgination. The key is that given your target platform/environment, the results and "best" option may differ. Understanding the strengths/weaknesses of various compilers, programming models, etc. is the key here. The first result chart on page 168 is interesting as it highlights the reasonableness of OpenACC (significantly improved performance) while also making it clear that the lower-level (closer to the hardware) options such as CUDA often edge it out. This drives the discussion to developer productivity which is somewhat of a different discussion entirely.

[<< Previous](../Chapter_07/readme.md)
|
[Next >>](../Chapter_09/readme.md)
