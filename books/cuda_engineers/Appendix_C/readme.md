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

```bash
$ cd dist_v2
$ make
/usr/local/cuda-9.0/bin/nvcc -g -G -Xcompiler -Wall -c main.cpp -o main.o
/usr/local/cuda-9.0/bin/nvcc -g -G -Xcompiler -Wall -c aux_functions.cpp -o aux_functions.o
/usr/local/cuda-9.0/bin/nvcc main.o aux_functions.o -o main

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
(cuda-gdb) break main.cpp:21
Breakpoint 1 at 0x5f45: file main.cpp, line 21.
(cuda-gdb) run
Starting program: /home/ru7/workspace/learn/books/cuda_engineers/Appendix_C/dist_v2/main
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main () at main.cpp:21
21	    return 0;
(cuda-gdb) print out
$1 = {0.5, 0.484126985, 0.46825397, 0.452380955, 0.43650794, 0.420634925, 0.40476191, 0.388888896, 0.373015881, 0.357142866, 0.341269851, 0.325396836, 0.309523821, 0.293650806, 0.277777791, 0.261904776,
  0.246031731, 0.230158716, 0.214285702, 0.198412687, 0.182539672, 0.166666657, 0.150793642, 0.134920627, 0.119047612, 0.103174597, 0.0873015821, 0.0714285672, 0.0555555522, 0.0396825373, 0.0238095224,
  0.00793650746, 0.00793653727, 0.0238095522, 0.0396825671, 0.055555582, 0.071428597, 0.0873016119, 0.103174627, 0.119047642, 0.134920657, 0.150793672, 0.166666687, 0.182539701, 0.198412716, 0.214285731,
  0.230158746, 0.246031761, 0.261904776, 0.277777791, 0.293650806, 0.309523821, 0.325396836, 0.341269851, 0.357142866, 0.373015881, 0.388888896, 0.40476191, 0.420634925, 0.43650794, 0.452380955, 0.46825397,
  0.484126985, 0.5}
(cuda-gdb) info locals
in = {0, 0.0158730168, 0.0317460336, 0.0476190485, 0.0634920672, 0.0793650821, 0.095238097, 0.111111112, 0.126984134, 0.142857149, 0.158730164, 0.174603179, 0.190476194, 0.206349209, 0.222222224, 0.238095239,
  0.253968269, 0.269841284, 0.285714298, 0.301587313, 0.317460328, 0.333333343, 0.349206358, 0.365079373, 0.380952388, 0.396825403, 0.412698418, 0.428571433, 0.444444448, 0.460317463, 0.476190478, 0.492063493,
  0.507936537, 0.523809552, 0.539682567, 0.555555582, 0.571428597, 0.587301612, 0.603174627, 0.619047642, 0.634920657, 0.650793672, 0.666666687, 0.682539701, 0.698412716, 0.714285731, 0.730158746, 0.746031761,
  0.761904776, 0.777777791, 0.793650806, 0.809523821, 0.825396836, 0.841269851, 0.857142866, 0.873015881, 0.888888896, 0.90476191, 0.920634925, 0.93650794, 0.952380955, 0.96825397, 0.984126985, 1}
out = {0.5, 0.484126985, 0.46825397, 0.452380955, 0.43650794, 0.420634925, 0.40476191, 0.388888896, 0.373015881, 0.357142866, 0.341269851, 0.325396836, 0.309523821, 0.293650806, 0.277777791, 0.261904776,
  0.246031731, 0.230158716, 0.214285702, 0.198412687, 0.182539672, 0.166666657, 0.150793642, 0.134920627, 0.119047612, 0.103174597, 0.0873015821, 0.0714285672, 0.0555555522, 0.0396825373, 0.0238095224,
  0.00793650746, 0.00793653727, 0.0238095522, 0.0396825671, 0.055555582, 0.071428597, 0.0873016119, 0.103174627, 0.119047642, 0.134920657, 0.150793672, 0.166666687, 0.182539701, 0.198412716, 0.214285731,
  0.230158746, 0.246031761, 0.261904776, 0.277777791, 0.293650806, 0.309523821, 0.325396836, 0.341269851, 0.357142866, 0.373015881, 0.388888896, 0.40476191, 0.420634925, 0.43650794, 0.452380955, 0.46825397,
  0.484126985, 0.5}
ref = 0.5
(cuda-gdb) continue
Continuing.
[Inferior 1 (process 15643) exited normally]
(cuda-gdb) quit

```

As you can see, most everything in v2 is the same as v1 but the code is structured a bit differently. The chapter than wraps up with a discusison of larger-memory allocation (heap vs. stack) and shows a modified version of v2.  I updated the source code (commented out prior version) and retested. The commands look as follows:

```bash
$ make
/usr/local/cuda-9.0/bin/nvcc -g -G -Xcompiler -Wall -c main.cpp -o main.o
/usr/local/cuda-9.0/bin/nvcc main.o aux_functions.o -o main

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
(cuda-gdb) break main.cpp:27
Breakpoint 1 at 0x5ec2: file main.cpp, line 27.
(cuda-gdb) run
Starting program: /home/ru7/workspace/learn/books/cuda_engineers/Appendix_C/dist_v2/main
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main () at main.cpp:27
27	    free(in);
(cuda-gdb) print out[5000]@100
$1 = {0.499749988, 0.499749959, 0.499749899, 0.499749839, 0.49974981, 0.49974975, 0.49974969, 0.49974966, 0.499749601, 0.499749541, 0.499749511, 0.499749452, 0.499749392, 0.499749362, 0.499749303, 0.499749243,
  0.499749213, 0.499749154, 0.499749094, 0.499749064, 0.499749005, 0.499748945, 0.499748886, 0.499748856, 0.499748796, 0.499748737, 0.499748707, 0.499748647, 0.499748588, 0.499748558, 0.499748498, 0.499748439,
  0.499748409, 0.499748349, 0.49974829, 0.49974826, 0.4997482, 0.499748141, 0.499748111, 0.499748051, 0.499747992, 0.499747962, 0.499747902, 0.499747843, 0.499747813, 0.499747753, 0.499747694, 0.499747664,
  0.499747604, 0.499747545, 0.499747515, 0.499747455, 0.499747396, 0.499747336, 0.499747306, 0.499747247, 0.499747187, 0.499747157, 0.499747097, 0.499747038, 0.499747008, 0.499746948, 0.499746889, 0.499746859,
  0.499746799, 0.49974674, 0.49974671, 0.49974665, 0.499746591, 0.499746561, 0.499746501, 0.499746442, 0.499746412, 0.499746352, 0.499746293, 0.499746263, 0.499746203, 0.499746144, 0.499746114, 0.499746054,
  0.499745995, 0.499745935, 0.499745905, 0.499745846, 0.499745786, 0.499745756, 0.499745697, 0.499745637, 0.499745607, 0.499745548, 0.499745488, 0.499745458, 0.499745399, 0.499745339, 0.499745309, 0.49974525,
  0.49974519, 0.49974516, 0.499745101, 0.499745041}
(cuda-gdb) continue
Continuing.
[Inferior 1 (process 16035) exited normally]
(cuda-gdb) quit
```

Note, however, that the print command in the debugger is modified. Rather than simply saying `print out` (asking it to print the contents of the `out` variable), we issued a command like: `print out[5000]@100` which tells the debugger to print to the console 100 values of out starting at the 5,000th index.

[<< Previous](../Appendix_B/readme.md)
|
[Next >>](../Appendix_D/readme.md)
