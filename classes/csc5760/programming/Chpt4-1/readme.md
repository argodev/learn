---
title: CSC5760 Programming Assignment 1
author: Rob Gillen
header-includes:
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[RO,RE]{Programming Assignment 1}
    - \fancyhead[LO,LE]{Rob Gillen, T00215814}
    - \usepackage{tikz}
    - \usetikzlibrary{calc,shapes.multipart,chains,arrows}
---

# CSC5760 Programming Assignment 1

## Building the Code
Compiling the applictions in this assignment is as simple as running `make` at the root of the directory. With no arguments, all of the applications will be compiled into the `/bin` directory.

Alternatively, you can run make with the various targets to compile the individual programs

```bash
$ make no4.13.2
$ make no4.13.14
$ make no4.14.4
```

## Running the Code
In general, running each of the executeables pasing the `--help` argument should cause them to display their usage information, after which, running the application is as simple as providing the appropriate values. 

\pagebreak

### 4.13.2
This is a modified version of the matrix/vector multiplication routine discussed in the text (`pth_mat_vect.c`). The modifications include a randomly-generated matrix and vector (rather than requiring the user to enter them each time), and the input matrix and output vector are both distributed into the threads as local variables rather than global variables.

```bash
$ bin/no4.13.2 -m 20 -n 10 -t 4
```

The assumption is that the purpose of this exercise is to illustrate the impact of using a thread-specific variables vs. global. As such, for the "isolated" version, we generate the portions of the input matrix in each individual thread, calculate the results in the thread-specific variables, and then copy the output to the shared heap. This solves the "scheduling" problem described in the book's question. 

Ran the code 5x using both shared (global) variables and local variables. The parameters used were m=4000, n=4000, t=4 and the results are listed below.

|Run|Shared Compute|Shared Wall|Private Compute|Private Wall|
|--:|-------------:|----------:|--------------:|-----------:|
|1 |1.219969e-01|3.168392e-02|9.232092e-02|2.283108e+00|
|2 |1.182439e-01|3.142715e-02|9.216976e-02|2.229935e+00|
|3 |1.189280e-01|3.217888e-02|9.718895e-02|2.164398e+00|
|4 |1.187811e-01|3.226018e-02|9.339714e-02|2.279607e+00|
|5|1.184990e-01|3.226399e-02|1.032341e-01|2.217488e+00|
|Avg|1.192898E-01|3.196282E-02|9.566217E-02|2.234907E+00|

In the above, `Compute` refers to the sum of the time the threads each spent caluculating the answers for `y`. It is generally expected that this time is higher than the wall clock time if more than 1 thread is being utilized. As expected, `Wall` refers to the total time elapsed from the point wherein the threads were created until the join operation completed. This includes compute, data movement, and - in teh case of the `private` steps, the data generation for their portion of `A`.

Based on the data above, using thread-managed variables is more efficient when only the local, thread-specific computation is considered. In the private/isolated case the variables (`my_A` and `my_Y`) are allocated on the heap which I believe is a shared across the process (stack couldn't hold input values of interesting size).


\pagebreak

### 4.13.14

In this problem, we re-work the matrix/vector code to see the impact of using a 1D vector as the structure for the array vs. a 2D matrix (in both cases, they are logically storing the 2D matrix). It is possible to implement the 2D array simply by declaring it on the stack (e.g. `double my_array[m][n]`) however, the stack isn't of sufficient size to create arrays/matricies of interesting size and therefore this option was skipped.

Instead, both storage structures (A - 1D, B - 2D) were allocated on the heap via normal `malloc()` comands.

```bash
$ bin/no4.13.14 -m 20 -n 10 -t 4
```

The instructions were to run it from low to high and time the operations. The results are as follows (each value is the aveage over 5 runs):

|Size|1D Wall|2D Wall|
|:--:|------:|------:|
|10x10|1.419544E-04|6.794930E-05|
|100x100|7.126330E-04|1.153469E-04|
|1000x1000|2.021551E-03|2.503204E-03|
|10000x10000|1.942849E-01|1.985028E-01|

For the smaller-sized matricies, the two-dimensional (2D) structure appears to perform slighly better whereas when the matrix size increases, the 1D structure performs better.


\pagebreak

### 4.14.4

The purpose of this program is to calculate the average time needed to create and terminate a thread on my system, and to determine if it is affected by the number of threads. Running the program is simple:

```bash
$ bin/no4.14.4 -i 100 -t 100
```

Where:

- `i` is the number of iterations/tests to run
- `t` is the number of threads to create/destroy during each iteration

|Iterations|Threads|Average Time/Thread|
|---------:|------:|------------------:|
|100|4|1.316428e-05|
|100|10|1.037812e-05|
|100|100|1.356919e-05|
|100|500|1.387991e-05|

To answer the question, __No__, it does not appear that the number of threads created noticeably affects the average amount of time needed to create/delete the threads.

