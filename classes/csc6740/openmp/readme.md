# OpenMP Matrix Multiplication

This file explains how to build and run the application for testing the OpenMP Matrix Multiplication assignment.

## Building the application
Building the application is straight-forward assuming you are on a Linux-based system with the development tools installed. You can compile via the supplied makefile:

```bash
$ make
```

or you can build explicitly:
```bash
$ gcc -Wall -fopenmp -o prog1 prog1.c
```

## Running the application
As with most command line applications, simply running `./prog1 --help` will provide details as to the expected parameters. Examples of running the program are as follows:

```bash
$ prog1 -s 5000 -t 4
```

In the command above, `-s` specifies the size of the matrices to be generated (in this case 5000x5000), and `-t` provides the number of threads to use for the multiplication step.

For the purposes of this assignment, each command was run with the `time` utility to record the wall time of the run:

```bash
$ time prog1 -s 5000 -t 4
```
