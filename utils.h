#pragma once

#include <omp.h>
#include <thread>
#include <iostream>
#include <math.h>
#include <random>
#include "f_cutils.cuh"

void gen_costs_mod(double *cost_matrix, double *y_costs, const int *cycle, unsigned long seed, int SP_x, int SP_y, std::size_t N, std::size_t K)
{
  double val = 0;

  // std::random_device rd;

  double value = 0;
  //	int SP = K-2;
  float sigma10 = 0.3;
  float sigma20 = 0.2;
  std::cout << "X subproblems: " << SP_x << std::endl;
  std::size_t p = 0;
  std::size_t i = 0;
  std::size_t j = 0;
  // uint nthreads = 10;
  uint nthreads = std::min((uint)SP_y, (uint)std::thread::hardware_concurrency() - 3);
  std::cout << "Nthreads available: " << nthreads << std::endl;
  uint rows_per_thread = ceil(((SP_x - 1) * 1.0) / nthreads);
#pragma omp parallel for num_threads(nthreads)
  for (uint tid = 0; tid < nthreads; tid++)
  {
    uint first_row = tid * rows_per_thread;
    uint last_row = std::min(first_row + rows_per_thread, (uint)SP_x - 1);
    std::mt19937 gen(seed + tid);
    gen.discard(1);
    for (p = first_row; p < last_row; p++)
    {
      for (i = 0; i < N; i++)
      {
        for (j = 0; j < N; j++)
        {

          value = 0;
          val = 0;

          value = std::abs(cycle[p * N + i] - cycle[(p + 1) * N + j]) - 1;
          std::normal_distribution<double> d(value, sigma10);
          val = d(gen);
          std::size_t index = (p * N * N) + (i * N) + j;
          cost_matrix[index] = val;
          //		std::cout<<cost_matrix[index]<<std::endl;
        }
      }
    }
  }
  // #pragma omp barrier
  std::cout << "x costs generated" << std::endl;
  // exit(0);
  val = 0;
  value = 0;
  // long count = 0;
  std::size_t p2 = 0;
  std::size_t i2 = 0;
  std::size_t j2 = 0;
  std::size_t k2 = 0;
  rows_per_thread = ceil(((SP_y - 1) * 1.0) / nthreads);
  std::cout << rows_per_thread << std::endl;
  checkpoint();
#pragma omp parallel for num_threads(nthreads)
  for (uint tid = 0; tid < nthreads; tid++)
  {
    uint first_row = tid * rows_per_thread;
    uint last_row = std::min(first_row + rows_per_thread, (uint)SP_y - 1);
    std::mt19937 gen(seed + tid);
    gen.discard(1);
    for (p2 = first_row; p2 < last_row; p2++)
    {
      for (i2 = 0; i2 < N; i2++)
      {
        for (j2 = 0; j2 < N; j2++)
        {
          for (k2 = 0; k2 < N; k2++)
          {

            value = 0;
            val = 0;
            value = cost_matrix[p2 * N * N + (i2 * N) + j2] + cost_matrix[(p2 + 1) * N * N + (j2 * N) + k2];
            std::normal_distribution<double> d(value, sigma20);
            // val = d(gen);
            std::size_t id = N * N * N * (p2) + (k2 * N * N) + (i2 * N) + j2;
            y_costs[id] = d(gen);
            //	std::cout<<y_costs[id]<<std::endl;
          }
        }
      }
    }
  }

  std::cout << "y costs generated" << std::endl;
  std::size_t i3 = 0;
  std::size_t j3 = 0;
  std::size_t k3 = 0;

  for (i3 = 0; i3 < N; i3++)
  {
    for (j3 = 0; j3 < N; j3++)
    {
      std::size_t index = ((SP_x - 1) * N * N) + (i3 * N) + j3;
      cost_matrix[index] = 0;
      for (k3 = 0; k3 < N; k3++)
      {
        std::size_t id = N * N * N * (SP_y - 1) + (k3 * N * N) + (i3 * N) + j3;
        y_costs[id] = 0;
      }
    }
  }
  std::cout << "gen last costs" << std::endl;
}
