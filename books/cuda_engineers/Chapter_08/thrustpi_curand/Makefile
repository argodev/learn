NVCC = /usr/local/cuda-9.1/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall --std=c++11 --expt-extended-lambda
LIBS = -lcurand

main: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)
