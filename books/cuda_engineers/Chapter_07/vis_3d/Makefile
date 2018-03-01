# LDFLAGS += -L/usr/local/cuda-9.0/common/lib/linux/x86_64
LDFLAGS += -lglut -lGL -lGLU -lGLEW

NVCC = /usr/local/cuda-9.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations" -rdc=true
INC = -I /usr/local/cuda-9.0/samples/common/inc/

all: main

main: main.o kernel.o device_funcs.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h interactions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

kernel.o: kernel.cu kernel.h device_funcs.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

device_funcs.o: device_funcs.cu device_funcs.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.o main
