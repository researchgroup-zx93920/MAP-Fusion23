ARCH := $(shell ./get_SM.sh)
CUDAFLAGS ?= -g -Xcompiler -fopenmp -lineinfo -O3 -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
LDFLAGS := -lcuda -lgomp

all: main.o functions_cuda.o culap.o f_culap.o f_cutils.o
	g++ -fopenmp -g o/main.o o/functions_cuda.o o/culap.o o/f_culap.o o/f_cutils.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o test

main.o: main.cpp
	g++ -fopenmp -g -std=c++14 -I/usr/local/cuda/include -c main.cpp -o o/main.o

functions_cuda.o: functions_cuda.cu
	nvcc -arch=sm_80 -g -Xcompiler -O3 -lineinfo -std=c++14 -c functions_cuda.cu -o o/functions_cuda.o

culap.o: culap.cu
	nvcc -arch=sm_80 -g -Xcompiler -O3 -lineinfo -std=c++14 -c culap.cu -o o/culap.o

f_culap.o: f_culap.cu
	nvcc -arch=sm_80 -g -Xcompiler -O3 -lineinfo -std=c++14 -c f_culap.cu -o o/f_culap.o

f_cutils.o: f_cutils.cu
	nvcc -arch=sm_80 -g -Xcompiler -O3 -lineinfo -std=c++14 -c f_cutils.cu -o o/f_cutils.o

clean:
	rm -f test o/*.o