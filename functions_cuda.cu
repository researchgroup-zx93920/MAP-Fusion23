#include "functions_cuda.h"

void transferCosts(Matrix *d_y_costs_dev, Matrix *d_x_costs_dev, Vertices *d_vertices_dev, int N, int K, unsigned int devid, int *DSPC_x, int *DSPC_y, int offset_y, int offset_x)
{

	cudaSafeCall(cudaSetDevice(devid), "Error in cudaSetDevice function_cuda::initializeYCosts");

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;
	int y_size = N * N * N;
	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, y_size, DSPC_y[devid]);
	kernel_transferCosts_cuda<<<blocks_per_grid, threads_per_block>>>(d_y_costs_dev[devid].elements, d_x_costs_dev[devid].elements, d_vertices_dev[devid].row_duals, d_vertices_dev[devid].col_duals, devid, N, K, DSPC_y[devid], DSPC_x[devid], offset_y, offset_x);
	cudaDeviceSynchronize(); // was required to make the code enter the kernel
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
	cudaSafeCall_new(cudaGetLastError(), "Error in kernel_initializeYCosts Functions initializeYCosts");
}

void multiplier_update(Matrix *d_y_costs_dev, int N, int K, unsigned int devid, int *DSPC_y, int offset_y, int devcount, int procid, int procsize)
{

	cudaSafeCall(cudaSetDevice(devid), "Error in cudaSetDevice function_cuda::initializeYCosts");
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;
	int y_size = N * N;

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, y_size, DSPC_y[devid]);
	kernel_multiplier_update_cuda<<<blocks_per_grid, threads_per_block>>>(d_y_costs_dev[devid].elements, N, K, devid, DSPC_y[devid], offset_y, devcount, procid, procsize);
	cudaDeviceSynchronize(); // was required to make the code enter the kernel
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
	cudaSafeCall_new(cudaGetLastError(), "Error in kernel_initializeYCosts Functions initializeYCosts");
}

void solveYLSAP(Matrix *d_y_costs_dev, Matrix *d_x_costs_dev, int N, int K, unsigned int devid, int *DSPC_x, int *DSPC_y, int offset_y, int offset_x)
{

	cudaSafeCall(cudaSetDevice(devid), "Error in cudaSetDevice function_cuda::initializeYCosts");
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;
	// int y_size = N * N * N;
	// int x_size = N * N;
	// printf("%d\n", DSPC_y[devid]);

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, N);
	for (std::size_t i = 0; i < DSPC_y[devid]; i++)
	{
		kernel_solveYLSAP_cuda_min<<<blocks_per_grid, threads_per_block>>>(d_y_costs_dev[devid].elements, d_x_costs_dev[devid].elements, devid, N, K, DSPC_y[devid], DSPC_x[devid], offset_y, offset_x, i);

		cudaSafeCall_new(cudaGetLastError(), "Error in kernel_initializeYCosts Functions initializeYCosts");
	}

	for (int i = 0; i < DSPC_y[devid]; i++)
	{
		kernel_solveYLSAP_cuda_dual<<<blocks_per_grid, threads_per_block>>>(d_y_costs_dev[devid].elements, d_x_costs_dev[devid].elements, devid, N, K, DSPC_y[devid], DSPC_x[devid], offset_y, offset_x, i);
		cudaSafeCall_new(cudaGetLastError(), "Error in kernel_initializeYCosts Functions initializeYCosts");
	}

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
	cudaSafeCall_new(cudaGetLastError(), "Error in kernel_initializeYCosts Functions initializeYCosts");
}

// void solveYLSAP(Matrix *d_y_costs_dev, Matrix *d_x_costs_dev, int N , int K, unsigned int devid, int *DSPC_x, int *DSPC_y, int offset_y, int offset_x){
//
// 		cudaSafeCall(cudaSetDevice(devid), "Error in cudaSetDevice function_cuda::initializeYCosts");
// 		dim3 blocks_per_grid;
// 		dim3 threads_per_block;
// 		int total_blocks = 0;
// 		int y_size = N * N * N;
// 		int x_size = N * N;
//
// 		calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, x_size, DSPC_y[devid]);
//
// 		cudaDeviceSynchronize(); //was required to make the code enter the kernel
// 		cudaError_t error = cudaGetLastError();
// 		if (error != cudaSuccess) {
// 		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
// 		}
// 		cudaSafeCall(cudaGetLastError(), "Error in kernel_initializeYCosts Functions initializeYCosts");
// }

void computeUB(Matrix *d_x_costs_dev, Matrix *d_y_costs_dev, Vertices *d_vertices_dev, int N, int K, Objective *d_UB_dev, unsigned int devid, int *DSPC_x)
{

	cudaSafeCall(cudaSetDevice(devid), "Error in cudaSetDevice function_cuda::initializeYCosts");

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;
	int x_size = N;
	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, x_size, DSPC_x[devid]);
	kernel_computeUB<<<blocks_per_grid, threads_per_block>>>(d_y_costs_dev[devid].elements, d_x_costs_dev[devid].elements, d_vertices_dev[devid].row_assignments, N, K, DSPC_x[devid], d_UB_dev[devid].obj);
	// cudaDeviceSynchronize(); //was required to make the code enter the kernel
	//  printDebugArray(d_y_costs_dev[devid].elements, DSPC_y[devid] * N * N * N, 	"0", devid);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
	cudaSafeCall(cudaGetLastError(), "Error in kernel_initializeYCosts Functions initializeYCosts");
}

__global__ void kernel_computeUB(double *d_y_costs, double *d_x_costs, int *d_row_assignments, int N, int K, int SP, double *d_UB)
{

	int p = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (p < SP - 1 && i < N)
	{
		int j = d_row_assignments[N * p + i] % N;
		int k = d_row_assignments[N * (p + 1) + j] % N;

		atomicAdd(&d_UB[p], d_x_costs[p * N * N + N * i + j] + d_y_costs[N * N * N * p + N * N * i + N * j + k]);
		// d_UB[p] += d_x_costs[ p * N * N + N * i + j];
		// 	d_UB[p] += d_y_costs[N * N * N * p + N * N * i + N * j +k] ;
		// printf("%f\n",  d_x_costs[ p * N * N + N * i + j]);
	}
}

__global__ void kernel_transferCosts_cuda(double *d_y_costs, double *d_x_costs, double *d_row_duals, double *d_col_duals, unsigned int devid, std::size_t N, std::size_t K, int DSPC_y, int DSPC_x, int offset_y, int offset_x)
{

	std::size_t ylapid = blockIdx.y * blockDim.y + threadIdx.y;
	std::size_t ijk = blockIdx.x * blockDim.x + threadIdx.x;
	if (ylapid < DSPC_y)
	{

		std::size_t i = ijk / (N * N);
		std::size_t j = ijk % (N * N) / N;
		std::size_t k = ijk % N;

		if (i < N && j < N && k < N)
		{
			d_y_costs[(ylapid * N * N * N) + N * N * k + N * i + j] += d_x_costs[(ylapid) * (N * N) + N * i + j] - d_row_duals[(ylapid) * (N) + i] - d_col_duals[(ylapid) * (N) + j];
		}
	}
}

__global__ void kernel_multiplier_update_cuda(double *d_y_costs, std::size_t N, std::size_t K, unsigned int devid, int DSPC_y, int offset_y, int devcount, int procid, int procsize)
{

	std::size_t ylapid = blockIdx.y * blockDim.y + threadIdx.y;
	std::size_t ijk = blockIdx.x * blockDim.x + threadIdx.x;
	ylapid = ylapid * 2;
	if (devid != devcount - 1)
	{
		//	if(DSPC_y %2 ==0){

		if (ylapid < DSPC_y)
		{

			std::size_t j = ijk / N;
			std::size_t k = ijk % N;
			std::size_t y_size = N * N * N;

			if (ylapid < K - 2 && j < N && k < N)
			{

				double sum = 0;

				double min_cost1 = INF;
				double min_cost2 = INF;
				std::size_t k1 = 0;
				std::size_t k2 = 0;

				/////////////////////////////////////////////////////////////////////////

				for (std::size_t i_ = 0; i_ < N; i_++)
				{
					std::size_t y_id1 = (ylapid + 1) * y_size + N * N * i_ + N * j + k;
					if (min_cost1 >= d_y_costs[y_id1])
					{
						min_cost1 = d_y_costs[y_id1];
						k1 = i_;
					}
				}

				/////////////////////////////////////////////////////////////////////////

				for (std::size_t k_ = 0; k_ < N; k_++)
				{
					std::size_t y_id2 = (ylapid)*y_size + N * N * k + N * k_ + j;
					if (min_cost2 >= d_y_costs[y_id2])
					{
						min_cost2 = d_y_costs[y_id2];
						k2 = k_;
					}
				}
				/////////////////////////////////////////////////////////////////////////

				sum = (min_cost1) + (min_cost2);

				std::size_t yid1 = (ylapid + 1) * y_size + N * N * k1 + N * j + k;
				std::size_t yid2 = (ylapid)*y_size + N * N * k + N * k2 + j;

				d_y_costs[yid1] = sum * 0.44;

				d_y_costs[yid2] = sum * 0.56;

				//  d_y_costs[yid1] = sum * 0.5;
				// d_y_costs[yid2] = sum * 0.5;
			}
		}
	}

	if (devid == devcount - 1)
	{
		//	if(DSPC_y%2!=0){
		if (ylapid < DSPC_y - 1)
		{

			std::size_t j = ijk / N;
			std::size_t k = ijk % N;
			std::size_t y_size = N * N * N;

			if (ylapid < K - 2 && j < N && k < N)
			{

				double sum = 0;

				double min_cost1 = INF;
				double min_cost2 = INF;
				std::size_t k1 = 0;
				std::size_t k2 = 0;

				/////////////////////////////////////////////////////////////////////////

				for (std::size_t i_ = 0; i_ < N; i_++)
				{
					std::size_t y_id1 = (ylapid + 1) * y_size + N * N * i_ + N * j + k;
					if (min_cost1 >= d_y_costs[y_id1])
					{
						min_cost1 = d_y_costs[y_id1];
						k1 = i_;
					}
				}

				/////////////////////////////////////////////////////////////////////////

				for (std::size_t k_ = 0; k_ < N; k_++)
				{
					std::size_t y_id2 = (ylapid)*y_size + N * N * k + N * k_ + j;
					if (min_cost2 >= d_y_costs[y_id2])
					{
						min_cost2 = d_y_costs[y_id2];
						k2 = k_;
					}
				}
				/////////////////////////////////////////////////////////////////////////

				sum = (min_cost1) + (min_cost2);

				std::size_t yid1 = (ylapid + 1) * y_size + N * N * k1 + N * j + k;
				std::size_t yid2 = (ylapid)*y_size + N * N * k + N * k2 + j;

				d_y_costs[yid1] = sum * 0.44;

				d_y_costs[yid2] = sum * 0.56;

				//  d_y_costs[yid1] = sum * 0.5;
				// d_y_costs[yid2] = sum * 0.5;
			}
		}
	}
}

__global__ void kernel_solveYLSAP_cuda_min(double *d_y_costs, double *d_x_costs, unsigned int devid, std::size_t N, std::size_t K, int DSPC_y, int DSPC_x, int offset_y, int offset_x, std::size_t ylapid)
{

	// int ylapid = blockIdx.y * blockDim.y + threadIdx.y;

	std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N && j < N)
	{

		double min = INF;
		for (std::size_t k = 0; k < N; k++)
		{
			double cost = d_y_costs[ylapid * N * N * N + k * N * N + i * N + j];
			if (cost < min)
				min = cost;
		}

		d_x_costs[ylapid * N * N + i * N + j] = min;
	}
}
__global__ void kernel_solveYLSAP_cuda_dual(double *d_y_costs, double *d_x_costs, unsigned int devid, std::size_t N, std::size_t K, int DSPC_y, int DSPC_x, int offset_y, int offset_x, std::size_t ylapid)
{

	// int ylapid = blockIdx.y * blockDim.y + threadIdx.y;

	std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N && j < N)
	{

		double min = d_x_costs[ylapid * N * N + i * N + j];
		for (std::size_t k = 0; k < N; k++)
		{
			d_y_costs[ylapid * N * N * N + k * N * N + i * N + j] -= min;
		}
	}
}
