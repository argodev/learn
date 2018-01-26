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

### dist_v1
```bash
$ cd dist_v1
$ make
nvcc -g -G -Xcompiler -Wall main.cpp -o main
main.cpp: In function ‘int main()’:
main.cpp:18:11: warning: variable ‘out’ set but not used [-Wunused-but-set-variable]
     float out[N] = {0.0f};
           ^~~
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
(cuda-gdb) break main.cpp:33
Breakpoint 1 at 0x5f8e: file main.cpp, line 33.
(cuda-gdb) run
Starting program: /home/ru7/workspace/learn/books/cuda_engineers/Appendix_C/dist_v1/main
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main () at main.cpp:33
33	    return 0;
(cuda-gdb) info locals
out = {0.5, 0.484126985, 0.46825397, 0.452380955, 0.43650794, 0.420634925, 0.40476191, 0.388888896, 0.373015881, 0.357142866, 0.341269851, 0.325396836, 0.309523821, 0.293650806, 0.277777791, 0.261904776,
  0.246031731, 0.230158716, 0.214285702, 0.198412687, 0.182539672, 0.166666657, 0.150793642, 0.134920627, 0.119047612, 0.103174597, 0.0873015821, 0.0714285672, 0.0555555522, 0.0396825373, 0.0238095224,
  0.00793650746, 0.00793653727, 0.0238095522, 0.0396825671, 0.055555582, 0.071428597, 0.0873016119, 0.103174627, 0.119047642, 0.134920657, 0.150793672, 0.166666687, 0.182539701, 0.198412716, 0.214285731,
  0.230158746, 0.246031761, 0.261904776, 0.277777791, 0.293650806, 0.309523821, 0.325396836, 0.341269851, 0.357142866, 0.373015881, 0.388888896, 0.40476191, 0.420634925, 0.43650794, 0.452380955, 0.46825397,
  0.484126985, 0.5}
ref = 0.5
(cuda-gdb) print out
$1 = {0.5, 0.484126985, 0.46825397, 0.452380955, 0.43650794, 0.420634925, 0.40476191, 0.388888896, 0.373015881, 0.357142866, 0.341269851, 0.325396836, 0.309523821, 0.293650806, 0.277777791, 0.261904776,
  0.246031731, 0.230158716, 0.214285702, 0.198412687, 0.182539672, 0.166666657, 0.150793642, 0.134920627, 0.119047612, 0.103174597, 0.0873015821, 0.0714285672, 0.0555555522, 0.0396825373, 0.0238095224,
  0.00793650746, 0.00793653727, 0.0238095522, 0.0396825671, 0.055555582, 0.071428597, 0.0873016119, 0.103174627, 0.119047642, 0.134920657, 0.150793672, 0.166666687, 0.182539701, 0.198412716, 0.214285731,
  0.230158746, 0.246031761, 0.261904776, 0.277777791, 0.293650806, 0.309523821, 0.325396836, 0.341269851, 0.357142866, 0.373015881, 0.388888896, 0.40476191, 0.420634925, 0.43650794, 0.452380955, 0.46825397,
  0.484126985, 0.5}
(cuda-gdb) continue
Continuing.
[Inferior 1 (process 11528) exited normally]
(cuda-gdb) quit
```

### dist_v2

