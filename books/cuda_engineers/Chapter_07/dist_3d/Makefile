NVCC = /usr/local/cuda-9.1/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: main

main: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o main
