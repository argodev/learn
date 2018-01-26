NVCC = /usr/local/cuda-9.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
INC = -I /usr/local/cuda-9.0/include

all: main

main: main.cpp
	$(NVCC) $(NVCC_FLAGS) $(INC) $< -o $@
