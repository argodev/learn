NVCC = /usr/local/cuda-9.1/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
INC = -I /usr/local/cuda-9.1/samples/common/inc/

all: main

main: main.o kernel.o
	$(NVCC) $^ -o $@

main.o: main.cpp kernel.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.o main out.bmp
