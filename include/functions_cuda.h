#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
//#include <thrust/add.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "structures.h"
#include "d_structs.h"
// #include "helper_utils.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "d_vars.h"
#include "f_cutils.h"

#ifndef FUNCTIONS_CUDA_H_
#define FUNCTIONS_CUDA_H_

// void solveYLSAP(Matrix *d_y_costs, Matrix *d_x_costs, int N , int K, unsigned int devid, int *DSPC_x, int *DSPC_y, int offset_y, int offset_x);
void multiplier_update(Matrix *d_y_costs_dev, int N, int K, unsigned int devid, int *DSPC_y, int offset_y, int devcount, int procid, int procsize);

void solveYLSAP(Matrix *d_y_costs_dev, Matrix *d_x_costs_dev, int N, int K, unsigned int devid, int *DSPC_x, int *DSPC_y, int offset_y, int offset_x);

void transferCosts(Matrix *d_y_costs, Matrix *d_x_costs, Vertices *d_vertices_dev, int N, int K, unsigned int devid, int *DSPC_x, int *DSPC_y, int offset_y, int offset_x);

void computeUB(Matrix *d_x_costs_dev, Matrix *d_y_costs_dev, Vertices *d_vertices_dev, int N, int K, Objective *d_UB_dev, unsigned int devid, int *DSPC_x);

__global__ void kernel_transferCosts_cuda(double *d_y_costs, double *d_x_costs, double *d_row_duals, double *d_col_duals, unsigned int devid, std::size_t N, std::size_t K, int DSPC_y, int DSPC_x, int offset_y, int offset_x);

__global__ void kernel_multiplier_update_cuda(double *d_y_costs, std::size_t N, std::size_t K, unsigned int devid, int DSPC_y, int offset_y, int devcount, int procid, int procsize);

__global__ void kernel_computeUB(double *d_y_costs, double *d_x_costs, int *d_row_assignments, int N, int K, int SP, double *UB);

// __global__ void kernel_solveYLSAP_cuda_min(double *d_y_costs, double *d_x_costs, unsigned int devid, int N , int K,  int DSPC_y, int DSPC_x, int offset_y, int offset_x, int ylapid);
__global__ void kernel_solveYLSAP_cuda_min(double *d_y_costs, double *d_x_costs, unsigned int devid, std::size_t N, std::size_t K, int DSPC_y, int DSPC_x, int offset_y, int offset_x, std::size_t ylapid);

__global__ void kernel_solveYLSAP_cuda_dual(double *d_y_costs, double *d_x_costs, unsigned int devid, std::size_t N, std::size_t K, int DSPC_y, int DSPC_x, int offset_y, int offset_x, std::size_t ylapid);

#endif
