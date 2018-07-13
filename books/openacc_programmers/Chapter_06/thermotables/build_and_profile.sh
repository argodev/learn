#!/bin/sh
pgc++ --c++11 -pg thermo_cpu.c -o out_cpu
./out_cpu 100000 1000 
gprof out_cpu gmon.out > profile.txt 