NVCC = /usr/local/cuda-9.1/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
LIBS = -lnppc_static -lnppif_static -lculibos

main: main.cpp
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)
