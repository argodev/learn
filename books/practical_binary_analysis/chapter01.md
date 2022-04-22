# Chapter 1: Anatomy of a Binary

Four stages of "compilation":

* Preprocessor
* Complier
* Assembler
* Linker

Stopping after the preprocessor stage:

```
$ gcc -E -P compliation_example.c
```

Quite interesting to see how much gets dumped for such a simple program

Generate assembly code... stop just after the complier/prior to the linker.  
Also, force the use of Intel syntax (vs. AT&T).

```
$ gcc -S -masm=intel compilation_example.c
$ cat compilation_example.s
```

This works, but I have precious little understanding of what I'm looking at. 

Now, we are going to do the assembly phase:

```
$ gcc -c compilation_example.c
$ file compliation_example.o
```

Now, let's read some of the symbols in one of the binaries

```
$ readelf --syms a.out
```

Strip the debug symbols:

```
$ strip --strip-all a.out
```

an attempt at this point to read the syms (via `readelf`) fails... only yields a 
few lines 
