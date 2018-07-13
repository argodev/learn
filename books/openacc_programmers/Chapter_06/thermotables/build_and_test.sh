#!/bin/sh
echo "building CPU"
pgc++ --c++11 thermo_openacc.c -o out_cpu

echo "building CPU+GPU"
pgc++ --c++11 -acc thermo_openacc.c -o out_gpu

echo "building CPU multicore"
pgc++ --c++11 -acc -ta=multicore thermo_openacc.c -o out_cpu_multicore

echo "building CPU+GPU, optimized data locality"
pgc++ --c++11 -acc thermo_openacc_optimized.c -o out_gpu_optimized

export ACC_NUM_CORES=8
nbIter=1000
for nbData in 10 100 1000 10000 100000 1000000 10000000
do
  echo "----------------------------------------------"
  echo ""
  echo "tests with nbData, nbIter = " $nbData $nbIter
  echo ""
  echo "running CPU, 1 Core"
  time ./out_cpu $nbData $nbIter

  echo ""
  echo "running CPU, multicore"
  time ./out_cpu_multicore $nbData $nbIter

  echo ""
  echo "running CPU+GPU"
  time ./out_gpu $nbData $nbIter

  echo ""
  echo "running CPU+GPU, optimized data locality"
  time ./out_gpu_optimized $nbData $nbIter

  echo ""
done