# Chapter 3: Programming Tools for OpenACC

This chapter deals with various tools you will use when developing code for OpenACC. The chapter focuses on three main categories: Compilers, Performance Analysis tools, and Deubbing utilities. These are expanded below.

## Compilers
The authors lists four compilers that are OpenACC aware: `PGI`, `GCC`, `OpenUH`, and `Cray`. Each of these have pros and cons and support the OpenACC standard to varying degrees. While I would normally lean towards the `GCC` variant, both the book and some local experts I know encouraged me to use the `PGI` community edition. According to those who know better, the OpenACC support is particularly good in this compiler and switches such as `-Minfo=accel` causes the compiler to output a significant amount of detail regarding what is generated giving you as the developer insight into how the _"magic"_ is applied.

## Performance Analysis

The majority of the chapter focused on performance analysis an profilers. This is not surprising as the loop of compile/profile/optimize steps is a common pattern. They discuss at length notions of measurement preparation, running, and analysis. Further, they talk about constant sampling of performance data compared to event-based data collection and how the runtime and various tools support various aspects of these approaches.

At an introductory level, simply setting the following:

```bash
export PGI_ACC_TIME=1
```

will enable applications compiled with the PGI compiler to output performance tracing data which can be helpful in attempting to understand how the application is behaving.


## Debugging

The story for debugging OpenACC code is not quite as mature as you might hope, though there is a clear path that the author's lay out on page 52:

1. Compile without OpenACC support to confirm that the issue only appears with OpenACC enabled
1. Use additional information from the compiler to understand how the problem was decomposed (e.g. `-Minfo=accel`)
1. Review OpenACC spec
1. Isolate the bug-inducing section by moving it back to the host
1. Use commercial tools such as __TotalView__ and __Allinea DDT__.


[<< Previous](../Chapter_02/readme.md)
|
[Next >>](../Chapter_04/readme.md)
