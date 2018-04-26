NVCC = /usr/local/cuda-9.1/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
LIBS += -lcublas_static -lcusolver_static -lcusparse_static -lculibos

main: main.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)
