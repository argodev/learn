# Chapter 4: Using OpenACC for Your First Program



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
