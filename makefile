all: main.o functions_cuda.o culap.o f_culap.o f_cutils.o
	mpic++ -fopenmp -g o/main.o o/functions_cuda.o o/culap.o o/f_culap.o o/f_cutils.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o test

main.o: main.cpp
	mpic++ -fopenmp -g -std=c++14 -I/usr/local/cuda/include -c main.cpp -o o/main.o

functions_cuda.o: functions_cuda.cu
	nvcc -arch=sm_80 -g -lineinfo -std=c++14 -c functions_cuda.cu -o o/functions_cuda.o

culap.o: culap.cu
	nvcc -arch=sm_80 -g -lineinfo -std=c++14 -c culap.cu -o o/culap.o

f_culap.o: f_culap.cu
	nvcc -arch=sm_80 -g -lineinfo -std=c++14 -c f_culap.cu -o o/f_culap.o

f_cutils.o: f_cutils.cu
	nvcc -arch=sm_80 -g -lineinfo -std=c++14 -c f_cutils.cu -o o/f_cutils.o

clean:
	rm -f test o/*.o