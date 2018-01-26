# Appendix C: Need-to-Know C Programming

This chapter is designed to help the reader get started with C if they are not yet comfortable. It starts with an explanation of compiled vs. interpreted code, typed languages, and passing by value vs. reference. This is followed by a quick survey of some of the syntax (semicolons, braces, etc.).

Data types, delcarations and assignments are covered next. The authors then briefly describe function declarations before moving into the first example to create, complile, run and debug an application.

## Buliding Apps: Create, Compile, Run, Debug

This is a simple test/verification step to ensure all is going well.  We ran the following commands:

```bash
$ cd declare_and_assign
$ make
nvcc -g -G -Xcompiler -Wall main.cpp -o main
main.cpp: In function ‘int main()’:
main.cpp:3:9: warning: variable ‘i’ set but not used [-Wunused-but-set-variable]
     int i;
         ^
main.cpp:4:11: warning: variable ‘x’ set but not used [-Wunused-but-set-variable]
     float x;
           ^
$ cuda-gdb main
NVIDIA (R) CUDA Debugger
9.0 release
Portions Copyright (C) 2007-2017 NVIDIA Corporation
GNU gdb (GDB) 7.12
Copyright (C) 2016 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from main...done.
(cuda-gdb) start
Temporary breakpoint 1 at 0x5dba: file main.cpp, line 5.
Starting program: /home/ru7/workspace/learn/books/cuda_engineers/Appendix_C/declare_and_assign/main
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Temporary breakpoint 1, main () at main.cpp:5
5	    i = 2;
(cuda-gdb) next
6	    x = 1.3f;
(cuda-gdb) info locals
i = 2
x = 0
(cuda-gdb) next
8	    return 0;
(cuda-gdb) info locals
i = 2
x = 1.29999995
(cuda-gdb) continue
Continuing.
[Inferior 1 (process 10854) exited normally]
(cuda-gdb) quit
```

Once this has completed successfully, the reader is introduced to the following concepts:

- Arrays
- Memory Allocation
- Pointers
- Control Statements such as `for` and `if`


## Sample C Programs
Finally, the chapter presents two sample programs that both solve the same problem (calculating distances) in slightly different ways.


