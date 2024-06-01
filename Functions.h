#pragma once
#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <omp.h>
// #include "structures.h"
// #include "variables.h"
// #include "helper_utils.h"
#include "include/Timer.h"
#include "functions_cuda.cuh"
#include <set>
#include <ctime>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <climits>
#include <cmath>
#include "LAP/Hung_lap.cuh"
#include "culap.cuh"
#include "d_structs.h"
#include "d_vars.h"
#include "f_cutils.cuh"
#include <cstddef>

enum Countername
{
	OTHER = 0,
	LAP,
	NUM_COUNTERS
};
double LAP_total_time = 0.0;
class Functions
{

	std::size_t N, K;
	std::size_t SP_y, SP_x;	 // Number of subproblems on a host.
	std::size_t SP_offset;	 // y offset on host
	std::size_t SP_X_offset; // x offset on host
	std::size_t iterno;

	int numprocs;
	std::size_t numdev;
	int procid;
	int *sp_y_ptr, *sp_x_ptr; // offsets
	std::size_t devid;

	YSubProbDim *d_sp_y_dim;
	SubProbDim *d_sp_x_dim;
	Matrix h_y_costs, *d_y_costs_dev, *d_y_old_costs_dev, *d_y_new_costs_dev;
	Matrix h_x_costs, *d_x_costs_dev, *d_x_old_costs_dev, *d_x_new_costs_dev;

	uint *ungatedYindices;
	size_t *ungatedYscan;
	double *_x_costs, *_y_costs;
	int *DSPC_y, *DSPC_x;
	std::size_t y_size, x_size;

	std::size_t offset_x1, offset_y1;
	long N2;
	long *M; // total number of zero cost edges on a single host.

	std::size_t initial_assignment_count;
	int *stepcounts;
	double *steptimes;

	int *n_SP;
	int *n_ptr;

	int prevstep;
	bool flag;
	bool isFirstIter;

	double obj_val;
	double *SP_obj_val;

	std::size_t max_iter, iteration, iter;
	Vertices h_vertices, *d_vertices_dev, Best;
	CompactEdges *d_edges_csr_dev;
	VertexData *d_row_data_dev, *d_col_data_dev;
	double start_transfer, end_transfer, start_mult_update, end_mult_update, start_solveXLAP, end_solveXLAP, start_solveYLSAP, end_solveYLSAP;
	double gap, Best_UB, Best_gap, Best_LB, Total_objective_value, UB1;
	double *UB, *objec, *global_objec_dev, change, *start_time_dev, *stop_time_dev, *total_time_dev;
	int *h_row_assignments_main;
	Objective *d_x_opt_obj_dev, *d_UB_dev;
	int *row_assignments;

public:
	Functions(std::size_t _size, std::size_t _K, int _numdev, int _subprob_y, int _subprob_x, int _subproboffset,
						uint *ungatedYindices, size_t *ungatedYscan,
						int _subproboffset_X, int _iterno, int ***dispc, int _max_iter);

	void solve_DA_transfercosts(double *_x_costs, double *_y_costs, double &_LB,
															double *_proc_SP_obj_val, int *_row_assignments, bool _isFirstIter,
															const char *logfileName, double &_UB);

	~Functions();

private:
	void getMemInfo(void);
	void initialize_device(unsigned int devid);
	void finalize_device(unsigned int devid);
	double getUB(double *h_x_costs, double *h_y_costs, int *row_assignment, int SP_x);
};

Functions::Functions(std::size_t _size, std::size_t _K, int _numdev, int _subprob_y,
										 int _subprob_x, int _subproboffset,
										 uint *_ungatedYindices, size_t *_ungatedYscan,
										 int _subproboffset_X, int _iterno, int ***dispc, int _max_iter)
{

	N = _size;
	SP_y = _subprob_y;
	SP_x = _subprob_x;
	K = _K;
	SP_offset = _subproboffset;
	SP_X_offset = _subproboffset_X;
	max_iter = _max_iter;

	ungatedYindices = _ungatedYindices;
	ungatedYscan = _ungatedYscan;

	numdev = _numdev;
	iterno = _iterno;

	sp_y_ptr = new int[numdev + 1];
	sp_x_ptr = new int[numdev + 1];
	for (int i = 0; i < numdev; i++)
	{

		sp_y_ptr[i] = dev_y_iter_sub_prob_count[procid][i][iterno];
		sp_x_ptr[i] = dev_iter_sub_prob_count[procid][i][iterno];
	}
	exclusiveSumScan(sp_x_ptr, numdev);
	exclusiveSumScan(sp_y_ptr, numdev);
	for (int i = 0; i <= numdev; i++)
	{
		sp_y_ptr[i] += SP_offset;
		sp_x_ptr[i] += SP_X_offset;
	}
	d_sp_y_dim = new YSubProbDim[numdev];
	d_sp_x_dim = new SubProbDim[numdev];
	d_x_costs_dev = new Matrix[numdev];
	d_y_costs_dev = new Matrix[numdev];

	DSPC_y = new int[numdev];
	DSPC_x = new int[numdev];
	y_size = N * N * N;
	x_size = N * N;

	N2 = N * N * SP_x;
	M = new long[numdev];
	std::fill(M, M + numdev, 0);

	n_SP = new int[numdev];
	for (int i = 0; i < numdev; i++)
	{
		n_SP[i] = dispc[procid][i][_iterno];
	}

	n_ptr = new int[numdev + 1];
	std::copy(n_SP, n_SP + numdev, n_ptr);
	exclusiveSumScan(n_ptr, numdev);

	prevstep = 0;

	obj_val = 0;
	SP_obj_val = new double[SP_x];

	flag = false;
	isFirstIter = false;

	initial_assignment_count = 0;

	d_vertices_dev = new Vertices[numdev];
	d_edges_csr_dev = new CompactEdges[numdev];
	d_row_data_dev = new VertexData[numdev];
	d_col_data_dev = new VertexData[numdev];
	d_x_opt_obj_dev = new Objective[numdev];
	d_UB_dev = new Objective[numdev];

	std::size_t N1 = N;
	std::size_t SP_x1 = SP_x;

	h_vertices.row_assignments = new int[N1 * SP_x1];
	h_vertices.col_assignments = new int[N1 * SP_x1];

	std::fill(h_vertices.row_assignments, h_vertices.row_assignments + N1 * SP_x1, 0);
	std::fill(h_vertices.col_assignments, h_vertices.col_assignments + N1 * SP_x1, 0);

	objec = new double[numdev];
	UB = new double[numdev];
	global_objec_dev = new double[numdev];

	row_assignments = new int[N * SP_x];
	Best.row_assignments = new int[N * SP_x];
}

Functions::~Functions()
{
}

void Functions::getMemInfo(void)
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
					 prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
					 prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
					 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if (cudaSuccess != cuda_status)
	{
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
				 used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

void Functions::initialize_device(unsigned int devid)
{

	cudaSetDevice(devid);

	DSPC_y[devid] = dev_y_iter_sub_prob_count[procid][devid][iterno];

	DSPC_x[devid] = dev_iter_sub_prob_count[procid][devid][iterno];

	int offset_x = sp_x_ptr[devid];
	int offset_y = sp_y_ptr[devid];

	std::size_t N1 = N;
	std::size_t y_size1 = N1 * N1 * N1;
	std::size_t offset_y1 = offset_y;

	CUDA_RUNTIME(cudaMalloc((void **)&d_y_costs_dev[devid].elements, (std::size_t(DSPC_y[devid])) * y_size1 * sizeof(double)));
	CUDA_RUNTIME(cudaMalloc((void **)&d_x_costs_dev[devid].elements, (std::size_t(DSPC_x[devid])) * x_size * sizeof(double)));
	CUDA_RUNTIME(cudaMalloc((void **)&d_x_opt_obj_dev[devid].obj, (std::size_t(SP_x)) * sizeof(double)));
	CUDA_RUNTIME(cudaMemcpy(d_y_costs_dev[devid].elements, &h_y_costs.elements[y_size1 * offset_y1], (std::size_t(DSPC_y[devid])) * y_size1 * sizeof(double), cudaMemcpyHostToDevice));

	std::size_t x_size1 = N1 * N1;
	std::size_t offset_x1 = offset_x;
	CUDA_RUNTIME(cudaMemcpy(d_x_costs_dev[devid].elements, &h_x_costs.elements[offset_x1 * x_size1], (std::size_t(DSPC_x[devid])) * x_size1 * sizeof(double), cudaMemcpyHostToDevice));

	// long size = n_SP[devid] * N;

	// long row_offset = n_ptr[devid] * N;

	std::size_t size = DSPC_x[devid] * N;
	CUDA_RUNTIME(cudaMalloc((void **)(&d_vertices_dev[devid].row_assignments), size * sizeof(int)));
	CUDA_RUNTIME(cudaMalloc((void **)(&d_vertices_dev[devid].row_duals), size * sizeof(double)));
	CUDA_RUNTIME(cudaMalloc((void **)(&d_vertices_dev[devid].col_duals), size * sizeof(double)));
	CUDA_RUNTIME(cudaMemset(d_vertices_dev[devid].row_assignments, -1, size * sizeof(int)));
	CUDA_RUNTIME(cudaMemset(d_vertices_dev[devid].row_duals, 0, size * sizeof(double)));
	CUDA_RUNTIME(cudaMemset(d_vertices_dev[devid].col_duals, 0, size * sizeof(double)));
}

void Functions::finalize_device(unsigned int devid)
{

	cudaSetDevice(devid);
	CUDA_RUNTIME(cudaFree(d_y_costs_dev[devid].elements));
	CUDA_RUNTIME(cudaFree(d_x_opt_obj_dev[devid].obj));
	CUDA_RUNTIME(cudaFree(d_x_costs_dev[devid].elements));
	CUDA_RUNTIME(cudaFree(d_vertices_dev[devid].row_assignments));
	CUDA_RUNTIME(cudaFree(d_vertices_dev[devid].row_duals));
	CUDA_RUNTIME(cudaFree(d_vertices_dev[devid].col_duals));
}

void Functions::solve_DA_transfercosts(double *_x_costs, double *_y_costs, double &_LB, double *_proc_SP_obj_val,
																			 int *_row_assignments, bool _isFirstIter,
																			 const char *logfileName, double &_UB)
{

	Timer iter_start;
	h_y_costs.elements = _y_costs;
	h_x_costs.elements = _x_costs;
	h_row_assignments_main = _row_assignments;
	isFirstIter = _isFirstIter;
	Best_gap = INF;
	Best_UB = 0;
	Best_LB = 0;
	omp_set_num_threads(numdev);

	Timer start;

///////////////////////////////////////////////////////////////////
#pragma omp parallel
	{
		unsigned int devid = omp_get_thread_num();
		if (n_SP[devid] > 0)
		{
			// Hcheckpoint();
			initialize_device(devid);
			// Hcheckpoint();
			Total_objective_value = 0;
			TLAP<double> *tlap = new TLAP<double>(DSPC_x[devid], N, devid);
			for (int iter = 0; iter < max_iter; iter++)
			{
				objec[devid] = 0;
				UB[devid] = 0;
				change = 0;

				DSPC_y[devid] = dev_y_iter_sub_prob_count[procid][devid][iterno];

				DSPC_x[devid] = dev_iter_sub_prob_count[procid][devid][iterno];

				int offset_x1 = sp_x_ptr[devid];

				int offset_y1 = sp_y_ptr[devid];

				Timer time;
				// Hcheckpoint();
				transferCosts(d_y_costs_dev, d_x_costs_dev, d_vertices_dev,
											ungatedYindices, ungatedYscan,
											N, K, devid, DSPC_x, DSPC_y, offset_y1, offset_x1);
				end_transfer = time.elapsed_and_reset();
				// Hcheckpoint();
				// Timer start_mult;
				multiplier_update(d_y_costs_dev, N, K, devid, DSPC_y, offset_y1, numdev, procid, numprocs);
				end_mult_update = time.elapsed_and_reset();
				// Hcheckpoint();
				solveYLSAP(d_y_costs_dev, d_x_costs_dev, N, K, devid, DSPC_x, DSPC_y, offset_y1, offset_x1);
				end_solveYLSAP = time.elapsed_and_reset();

				// bool done = false;
				prevstep = -1;
				// Hcheckpoint();
				//  bool is_dynamic = false;
				Timer LAP_time_start;
				// CuLAP solvelap(N, DSPC_x[devid], devid, iter > 0); // can change to is_dynamic
				// solvelap.solve(d_x_costs_dev[devid].elements, d_vertices_dev[devid].row_assignments, d_vertices_dev[devid].row_duals, d_vertices_dev[devid].col_duals, d_x_opt_obj_dev[devid].obj);
				tlap->solve(d_x_costs_dev[devid].elements, d_vertices_dev[devid].row_assignments, d_vertices_dev[devid].row_duals, d_vertices_dev[devid].col_duals, d_x_opt_obj_dev[devid].obj);
				// printDebugMatrix(d_x_costs_dev[devid].elements, N, N, "cost matrix");
				// printDebugArray(d_vertices_dev[devid].row_assignments, N, "row assignments");
				// printDebugArrayDouble(d_vertices_dev[devid].row_duals, N, "row duals");
				// printDebugArrayDouble(d_vertices_dev[devid].col_duals, N, "col duals");
				LAP_total_time += LAP_time_start.elapsed();
				// Hcheckpoint();
				objec[devid] = reduceSUM(d_x_opt_obj_dev[devid].obj, SP_x, devid);

				global_objec_dev[devid] += objec[devid];

				/////////////////////////////////////Computing UB////////////////////////////////////
			}
			delete tlap;
			// Hcheckpoint();
			std::size_t offset_x2 = n_ptr[devid];
			std::size_t N1 = N;

			cudaSafeCall_new(cudaMemcpy(&h_vertices.row_assignments[offset_x2 * N], d_vertices_dev[devid].row_assignments, N1 * (std::size_t(DSPC_x[devid])) * sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_row_assignment");

			finalize_device(devid);
		}
#pragma omp barrier
	}

	double iter_time = start.elapsed();
	if (procid == 0)
		std::cout << "time " << iter_time << std::endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Total_objective_value = reduceSUM(global_objec_dev, numdev);

	// int offset_x = SP_X_offset;
	std::size_t N1 = N;

	// std::size_t y_size1 = N1 * N1 * N1;
	// std::size_t offset_y = SP_offset;
	std::size_t SP_x1 = SP_x;

	std::copy(h_vertices.row_assignments, h_vertices.row_assignments + N1 * SP_x1, _row_assignments);
	// for (std::size_t i = 0; i < N1 * SP_x1; i++)
	// {
	//   _row_assignments[i] = h_vertices.row_assignments[i];
	// }

	// UB1 = getUB(&h_x_costs.elements[offset_x * x_size], &h_y_costs.elements[offset_y * y_size1], h_vertices.row_assignments, SP_x);
	// _UB = UB1;
	_LB = Total_objective_value;
	// std::cout << "proc " << procid << "  LB  " << Total_objective_value << "   UB    " << UB1 << std::endl;
}

double Functions::getUB(double *h_x_costs, double *h_y_costs, int *row_assignments, int SP_x)
{
	double total_cost = 0;
	std::size_t N1 = N;
	// std::size_t K1 = K;
	std::size_t p = 0;
	std::size_t i = 0;
	std::size_t i1 = 0;
	std::size_t SP_x1 = SP_x;
	for (p = 0; p < SP_x1 - 1; p++)
	{
		for (i = 0; i < N1; i++)
		{

			std::size_t j = row_assignments[N1 * (p) + i] % N1;
			std::size_t k = row_assignments[N1 * ((p) + 1) + j] % N1;
			//
			//
			total_cost += h_x_costs[(p)*N1 * N1 + N1 * i + j];
			total_cost += h_y_costs[N1 * N1 * N1 * (p) + N1 * N1 * k + N1 * i + j];
		}
	}

	for (i1 = 0; i1 < N1; i1++)
	{
		std::size_t j = row_assignments[N1 * (SP_x1 - 1) + i1] % N1;

		total_cost += h_x_costs[(SP_x1 - 1) * N1 * N1 + N1 * i1 + j];
	}

	return total_cost;
}

#endif /* FUNCTIONS_H_ */
