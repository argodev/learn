# Chapter 4: Using OpenACC for Your First Program

The aim of this chapter is to walk the user through creating their first OpenACC program. I must say, however, that I appreciate the approach taken by the author. Rather than simply giving you a bit of code to compile/run, there is a clear intentionality to teach the reader. The author begins by explaining the problem and having the user implement the application  in a serial fashion (much the way they recommend throughout the book) and then proceed from there. My notes on this section follow my walk-through of the exercises.

## Setup
 
I am utilizing two different machines for this lab. The first (Lap01) is my main laptop that I carry each day and has the following specifications:

- Ubuntu 18.04 LTS
- 32 GB RAM
- Intel Core i7-7820HQ CPU @ 2.90 GHz x 8
- Quadro M1200/PCIe/SSE2

The second machine is a bigger laptop (Lap02) and is provided for comparison purposes as it has a better GPU. Its specifications are as follows:

- Ubuntu 18.04 LTS
- 32 GB RAM
- Intel Core i7-820HQ CPU @ 2.90 GHz x 8
- Quadro P3000/PCIe/SSE2

## Serial Code

This step was rather straight-forward. I simply implemented the code as described and compiled it on both machines using the command below:

```bash
pgcc laplace.c -o serial
```
 
```bash
$ pgcc -acc -Minfo=acc laplace.c -o laplace
main:
     26, Generating implicit copyin(Temperature_previous[:][:])
         Generating implicit copyout(Temperature[1:1000][1:1000])
     27, Loop is parallelizable
     28, Loop is parallelizable
         Accelerator kernel generated
         Generating Tesla code
         27, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
         28, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
     38, Generating implicit copyin(Temperature[1:1000][1:1000])
         Generating implicit copy(Temperature_previous[1:1000][1:1000])
     39, Loop is parallelizable
     40, Loop is parallelizable
         Accelerator kernel generated
         Generating Tesla code
         39, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
         40, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
         41, Generating implicit reduction(max:worst_dt)
```


pgcc -o laplace -mp laplace.c
export OMP_NUM_THREADS=8



[<< Previous](../Chapter_03/readme.md)
|
[Next >>](../Chapter_05/readme.md)
