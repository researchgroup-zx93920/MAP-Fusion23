/*
 * f_cutils.cpp
 *
 *  Created on: Jul 11, 2015
 *      Author: date2
 */

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <thrust/scan.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "d_vars.h"
#include <sstream>
#include <random>
#include <time.h>
#include <omp.h>
#include <thread>

#ifndef F_CUTILS_H_
#define F_CUTILS_H_

#define cudaSafeCall_new(ans, message)        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
    cudaDeviceSynchronize();
    if (code != cudaSuccess)
    {
        // fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        // if (abort)
        exit(1);
    }
}

void cudaSafeCall(cudaError_t error, const char *message);
void printMemoryUsage(double memory);

void readFile(const char *filename);
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize);
void calculateCubicDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize, int zsize);
void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath, int N);
void exclusiveSumScan(int *arr, int size);
void exclusiveSumScan(long *arr, int size);
void exclusiveSumScan(double *array, int size);
int reduceSUM(int *arr, int size);
long reduceSUM(long *arr, int size);
double reduceSUM(double *arr, int size, int devid);
double reduceSUM(double *array, int size);

bool reduceOR(bool *arr, int size);
double reduceMIN(double *arr, int size);

void printDebugArray(int *d_array, int size, const char *name);
void printDebugMatrix(double *d_matrix, int rowsize, int colsize, const char *name);
void printDebugMatrix(int *d_matrix, int rowsize, int colsize, const char *name);
void printHostArray(int *h_array, int size, const char *name);
void printHostArray(double *h_array, int size, const char *name);
void printDebugArray(long *d_array, int size, const char *name);
void printDebugArray(double *d_array, int size, const char *name);
void printHostArray(long *h_array, int size, const char *name);
void printHostMatrix(double *h_matrix, int rowsize, int colsize, const char *name);
void printHostMatrix(int *h_matrix, int rowsize, int colsize, const char *name);
void printDebugArray(double *d_array, int size, const char *name, unsigned int devid);
void printToFile(int *row_assignments, double LB, double UB, double time, int N, int K, int procsize, int numdev);
void readFile(double *cost_matrix, const char *filename);
void readFiley(double *cost_matrix, const char *filename);
void createProbGenData(int *cycle, unsigned long seed);
void gen_costs(double *cost_matrix, double *y_costs, int *cycle, unsigned long seed, int SP_x, int SP_y, std::size_t N, std::size_t K);
void gen_costs_mod(double *cost_matrix, double *y_costs, int *cycle, unsigned long seed, int SP_x, int SP_y, std::size_t N, std::size_t K);

void printToFile2(double LB, double UB, int N, int K, int procsize, int numdev, double time);
// double getUB_all_batches(double *h_x_costs, double *h_y_costs, int *h_row_assignments, int N);

void split(int *array, int val, int size);
void split(long *array, long val, int size);

__global__ void kernel_memSet(int *array, int val, int size);
__global__ void kernel_memSet(double *array, double val, int size);
__global__ void kernel_memSet(long *array, long val, int size);

#endif /* F_CUTILS_H_ */
