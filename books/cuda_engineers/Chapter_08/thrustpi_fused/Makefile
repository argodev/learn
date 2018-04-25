NVCC = /usr/local/cuda-9.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall --std=c++11 --expt-extended-lambda

main: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
