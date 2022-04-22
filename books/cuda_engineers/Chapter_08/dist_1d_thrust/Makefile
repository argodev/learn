NVCC = /usr/local/cuda-9.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations" -rdc=true

main: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
