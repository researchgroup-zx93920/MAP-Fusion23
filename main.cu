#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <set>
#include <algorithm>
#include <climits>
#include <omp.h>
#include <cmath>
#include "d_structs.h"
#include "d_vars.h"
#include "f_cutils.cuh"
#include "Functions.h"
#include <thrust/set_operations.h>
#include <random>
#include "timer.h"
#include <cstddef>
#include "utils.h"

int ranks = 0, procsize = 1;

int devcount = 0;
std::size_t n = 0;
std::size_t K = 0;
int n_factor = 25;
int seedId = 0;

int sub_prob_count = 0;
int *proc_sub_prob_count = 0;
int *proc_iterations = 0;
int **proc_iter_sub_prob_count = 0;
int **dev_sub_prob_count = 0;
int **dev_iterations = 0;
int ***dev_iter_sub_prob_count = 0;
int ***n_dev_iter_ptr = 0;
int **n_proc_iter_ptr = 0;
int *n_global_ptr = 0;
SubProbMap *spMap = 0;
int scorer;
int problem_number;
int *prob_gen_cycle = 0;

int y_sub_prob_count = 0;
int *proc_y_sub_prob_count = 0;
int *proc_y_iterations = 0;
int **proc_y_iter_sub_prob_count = 0;
int **dev_y_sub_prob_count = 0;
int **dev_y_iterations = 0;
int ***dev_y_iter_sub_prob_count = 0;
int ***n_y_dev_iter_ptr = 0;
int **n_y_proc_iter_ptr = 0;
int *n_y_global_ptr = 0;
int *y_sp_dev_split = 0;
int *y_sp_proc_split = 0;
int *total_proc_iterations = 0;
int **total_proc_iter_count = 0;
int **total_y_proc_iter_count = 0;
// int ***global_dev_iter_sub_prob_count =0;
// int ***global_dev_y_iter_sub_prob_count =0;
double *UB = 0;
double *LB = 0;
double proc_obj_val = 0;
double proc_UB = 0;
double global_UB = 0;
double global_obj_val = 0;

YSubProbMap *y_spMap = 0;
unsigned long seeds[] = {242169117975701111, 242168292950692026, 242168237993338490, 241739468444391687, 242140973494040460};

double Epsilon = 0.0; // 0.00000000000000000001;
int MAX_SUBPROB, MAX_SUBPROB_Y_CPU, MAX_SUBPROB_Y_GPU;
int MAX_SUBPROB_OVER, MAX_SUBPROB_OVER_Y;

// int iter =1;
void initialize();
void finalize();
float getMemInfo();
double getUB_all_batches(double *h_x_costs, double *h_y_costs, int *h_row_assignments, int N);

int main(int argc, char **argv)
{
  // std::cout << "procsize: " << procsize << " ranks: " << ranks << std::endl;

  n = atoi(argv[1]);
  K = atoi(argv[2]);
  devcount = atoi(argv[3]);
  int iterations = atoi(argv[4]);

  problem_number = atoi(argv[5]); // Problem instance of size (n, k)
  scorer = atoi(argv[6]);         // Linear or spline
  seedId = atoi(argv[7]);
  bool get_costs_from_file = atoi(argv[8]);
  const char *filename_triplet = argv[9]; // Provided as 0 for large tests
  const char *filename = argv[10];

  // if (get_costs_from_file)
  // {
  //   readFiley(y_costs, filename_triplet);
  //   readFile(cost_matrix, filename);
  // }
  // else
  // {

  char logfileName[2500];
  sprintf(logfileName, "./Results/Log_K%zu_n%zu_s%d_p%d_proc%d_dev%d.txt", K, n, scorer, problem_number, procsize, devcount);

  initialize();
  std::cout << "fin" << std::endl;
  // int subproblems_proc = proc_sub_prob_count[ranks];
  // int subproblems_y_proc = proc_y_sub_prob_count[ranks];

  std::cout << "subproblems_proc" << std::endl;
  double global_obj_val = 0;
  // double global_UB = 0;

  std::size_t pspc = proc_sub_prob_count[ranks];
  std::size_t pspc_y = proc_y_sub_prob_count[ranks];
  std::cout << "n*pspc " << n * pspc << std::endl;
  int *row_assignments = new int[n * pspc];
  std::fill(row_assignments, row_assignments + n * pspc, 0);
  checkpoint();
  //    int *greedy_row_assignments = new int[n * pspc];
  double *proc_SP_obj_val = new double[pspc];
  //   int *global_row_assignments = new int[n * sub_prob_count];
  double *global_SP_obj_val = new double[sub_prob_count];
  int *disps = new int[procsize];
  int *recvs = new int[procsize];
  checkpoint();
  double *cost_matrix = new double[n * n * (pspc)];
  checkpoint();
  // double *y_costs;
  double *y_costs = new double[n * n * n * (pspc_y)];
  std::size_t offset_x = 0;
  std::size_t offset_y = 0;

  Timer read_time;
  // double start_read = t.elapsed();

  std::cout << "cpp fill matrix memory " << std::endl;

  std::fill(cost_matrix, cost_matrix + n * n * pspc, 0);
  std::cout << "cpp fill cost matrix memory done   " << std::endl;

  // std::fill(y_costs, y_costs + n * n * n * pspc_y, 0);

  if (get_costs_from_file)
  {
    std::fill(y_costs, y_costs + n * n * n * pspc_y, 0);

    readFiley(y_costs, filename_triplet); // Find these functions in f_cutils.cu
    readFile(cost_matrix, filename);
  }
  else
  {

    prob_gen_cycle = new int[K * n];
    unsigned long seed = 0;

    seed = seeds[seedId];
    if (ranks == 0)
      createProbGenData(prob_gen_cycle, seed);

    std::cout << "cycle gen" << std::endl;

    // std::cout << "cpp fill y_cost matrix memory done   " << std::endl;

    seedId = ranks;
    seed = seeds[seedId];
    gen_costs_mod(cost_matrix, y_costs, prob_gen_cycle, seed, pspc, pspc_y, n, K); // Find these functions in f_cutils.cu

    std::cout << "all costs generated" << std::endl;
  }
  double total_read_time = read_time.elapsed();

  std::ofstream logfile(logfileName);
  if (ranks == 0)
  {
    logfile << "\nGen_time (s),\tTransfer_time(s),\tLAP_time,\tObjective_value,\tUpper_Bound,\tGap\n";
    logfile << total_read_time << ", ";
    // logfile.close();
    // std::ofstream logfile(filename,)
  }
  //
  //
  //

  Timer tot_time;
  /////////////////////////////////////////////////////////////DUAL ASCENT - ENTERING DEVICE //////////////////////////////////////////////////////

  Timer transfer_time;
  offset_x = 0;
  offset_y = 0;

  double UB_val = 0;
  for (int j = 0; j < proc_y_iterations[ranks]; j++)
  {
    double obj_val = 0;
    if (proc_y_iter_sub_prob_count[ranks][j] > 0)
    {
      Functions rlt(n, K, devcount, proc_y_iter_sub_prob_count[ranks][j], proc_iter_sub_prob_count[ranks][j], n_y_proc_iter_ptr[ranks][j], n_proc_iter_ptr[ranks][j], j, dev_iter_sub_prob_count, iterations);
      rlt.solve_DA_transfercosts(cost_matrix, y_costs, obj_val, &proc_SP_obj_val[offset_x], &row_assignments[offset_x * n], true, logfileName, UB_val);

      proc_obj_val += obj_val;
      proc_UB += UB_val;
      if (proc_iter_sub_prob_count[ranks][j] <= proc_sub_prob_count[ranks])
      {
        offset_x += proc_iter_sub_prob_count[ranks][j];
      }
      offset_y += proc_y_iter_sub_prob_count[ranks][j];
    }
  }
  for (int p = 0; p < procsize; p++)
  {
    recvs[p] = proc_sub_prob_count[p] * n;
    disps[p] = n_global_ptr[p] * n;
  }

  double ub_all_batches = 0;
  ub_all_batches = getUB_all_batches(cost_matrix, y_costs, row_assignments, n);

  std::cout << "ub_all_batches"
            << "  " << ub_all_batches << std::endl;

  global_obj_val = proc_obj_val;

  double total_time_transfer = transfer_time.elapsed();

  if (ranks == 0)
  {
    double gap = std::abs((ub_all_batches - global_obj_val) / ub_all_batches) * 100;
    // std::ofstream logfile(logfileName);
    logfile << total_time_transfer << ", " << LAP_total_time << ", " << global_obj_val << ", " << ub_all_batches << ", " << gap << std::endl;
    logfile.close();
  }

  /*
             delete[] cost_matrix;
           delete[] row_assignments;
           delete[] global_row_assignments;
            delete[] y_costs;
             delete[] disps;
             delete[] recvs;
             finalize();
  */

  double total_time = tot_time.elapsed();
  std::cout << "Total time: " << total_time << ";  LAP time " << LAP_total_time << std::endl;

  printToFile2(global_obj_val, ub_all_batches, n, K, procsize, devcount, total_time);

  return 0;
}

float getMemInfo()
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
  printf("GPU memory usage: used = %f GB, free = %f GB, total = %f GB\n",
         used_db / 1024.0 / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0 / 1024.0);
  return free_db / 1024.0 / 1024.0 / 1024.0;
}

void initialize()
{

  // All the memory requirement values are in Giga Bytes.
  float available_memory_on_GPU = 0.95 * getMemInfo();
  // float available_memory_on_GPU = 0.01 * getMemInfo();

  float mem_required_for_y = (n * n * n * (K - 1)) * 8 / 1024.0 / 1024.0 / 1024.0;
  float mem_required_for_x = (n * n * K) * 8 / 1024.0 / 1024.0 / 1024.0;
  float mem_required_per_subproblem_y = (n * n * n) * 8 / 1024.0 / 1024.0 / 1024.0;
  float mem_required_per_subproblem_x = (n * n) * 8 / 1024.0 / 1024.0 / 1024.0;

  MAX_SUBPROB_Y_GPU = int(available_memory_on_GPU / (mem_required_per_subproblem_y + mem_required_per_subproblem_x));
  MAX_SUBPROB_Y_CPU = 500;
  // MAX_SUBPROB_Y_GPU = 76;

  MAX_SUBPROB_OVER_Y = int((K - 1) / devcount) % MAX_SUBPROB_Y_GPU - 1;

  // MAX_SUBPROB_Y_GPU = max_subprob_list_new[n/201];
  // MAX_SUBPROB_OVER_Y = max_subprob_over_list_new[n/201];

  // int max_subprob_list[] = {40000, 10000, 4444, 2500, 1600, 1111, 816, 625, 493, 400, 330, 277, 236, 204, 177, 156, 138, 123, 111, 100, 90, 82, 75, 69, 64, 59, 54, 51, 47, 44, 41, 39, 36, 34, 32, 30, 29, 27, 26, 25};
  // int max_subprob_over_list[] = {8001, 2001, 890, 501, 321, 223, 164, 126, 100, 81, 67, 57, 49, 41, 37, 32, 29, 26, 22, 21, 19, 18, 16, 15, 13, 13, 12, 11, 11, 10, 9, 8, 9, 8, 8, 8, 7, 7, 6, 6};
  // MAX_SUBPROB = max_subprob_list[(n - 1) / n_factor];
  // MAX_SUBPROB_OVER = max_subprob_over_list[(n - 1) / n_factor];
  // int max_subprob_list_new[] = {200, 76, 27};

  // int max_subprob_over_list_new[] = {41, 16, 7};

  // Computing max_subprob_over_list_new

  // MAX_SUBPROB_Y_GPU = max_subprob_list_new[n/200];

  // MAX_SUBPROB_OVER_Y = max_subprob_over_list_new[n/200];

  printf("GPU memory usage: Y = %f GB, X = %f GB,  Each Y = %f GB, Each X = %f GB,   Num of previous subproblems = %d, Num of over subproblems = %d\n",
         mem_required_for_y, mem_required_for_x, mem_required_per_subproblem_y, mem_required_per_subproblem_x, MAX_SUBPROB_Y_GPU, MAX_SUBPROB_OVER_Y);
  // MAX_SUBPROB_OVER_Y = 16;

  y_sub_prob_count = K - 1;
  spMap = new SubProbMap[1];
  y_spMap = new YSubProbMap[1];
  sub_prob_count = K;
  spMap[0].size = sub_prob_count;
  spMap[0].dim1 = new int[sub_prob_count];
  spMap[0].dim2 = new int[sub_prob_count];
  spMap[0].procId = new int[sub_prob_count];
  spMap[0].devId = new int[sub_prob_count];

  y_spMap[0].size = y_sub_prob_count;
  y_spMap[0].dim1 = new int[y_sub_prob_count];
  y_spMap[0].procId = new int[y_sub_prob_count];
  y_spMap[0].devId = new int[y_sub_prob_count];

  proc_sub_prob_count = new int[procsize];
  proc_iterations = new int[procsize];
  proc_iter_sub_prob_count = new int *[procsize];
  dev_iterations = new int *[procsize];
  dev_sub_prob_count = new int *[procsize];
  dev_iter_sub_prob_count = new int **[procsize];
  n_dev_iter_ptr = new int **[procsize];
  n_proc_iter_ptr = new int *[procsize];
  n_global_ptr = new int[procsize + 1];
  int overflow = sub_prob_count % procsize;
  y_sp_proc_split = new int[procsize + 1];
  y_sp_dev_split = new int[devcount + 1];
  // int MIN_GPU = procsize * devcount;

  proc_y_sub_prob_count = new int[procsize];
  proc_y_iterations = new int[procsize];
  proc_y_iter_sub_prob_count = new int *[procsize];
  dev_y_iterations = new int *[procsize];
  dev_y_sub_prob_count = new int *[procsize];
  dev_y_iter_sub_prob_count = new int **[procsize];
  n_y_dev_iter_ptr = new int **[procsize];
  n_y_proc_iter_ptr = new int *[procsize];
  n_y_global_ptr = new int[procsize + 1];
  int overflow_y = y_sub_prob_count % procsize;
  int dimension1 = 0, dimension2 = 0, spCount = 0;
  int dimension1_y = 0, y_spCount = 0;
  total_proc_iterations = new int[procsize];
  total_y_proc_iter_count = new int *[procsize];
  total_proc_iter_count = new int *[procsize];

  for (int i = 0; i < procsize; i++)
  {
    proc_sub_prob_count[i] = overflow >= (procsize - i) ? (sub_prob_count / procsize) + 1 : (sub_prob_count / procsize);
    dev_sub_prob_count[i] = new int[devcount];
  }

  for (int i = 0; i < procsize; i++)
  {
    proc_y_sub_prob_count[i] = overflow_y >= (procsize - i) ? (y_sub_prob_count / procsize) + 1 : (y_sub_prob_count / procsize);
    dev_y_sub_prob_count[i] = new int[devcount];
  }

  ////////////////////////////////////////////////////////////////////////////////PROC LEVEL///////////////////////////////////////////////////////////
  int extra_proc_overflow_y = 0, extra_proc_overflow_x = 0;
  bool false_flag = false;
  for (int i = 0; i < procsize - 1; i++)
  {
    if (proc_sub_prob_count[i] > proc_y_sub_prob_count[i])
    {
      extra_proc_overflow_x += proc_sub_prob_count[i] - proc_y_sub_prob_count[i];
      proc_sub_prob_count[i] = proc_y_sub_prob_count[i];
    }
    if (proc_y_sub_prob_count[i] > proc_sub_prob_count[i])
    {
      extra_proc_overflow_y += proc_y_sub_prob_count[i] - proc_sub_prob_count[i];
      proc_y_sub_prob_count[i] = proc_sub_prob_count[i];
    }
  }
  proc_sub_prob_count[procsize - 1] += extra_proc_overflow_x;
  proc_y_sub_prob_count[procsize - 1] += extra_proc_overflow_y;

  for (int j = 0; j < procsize - 1; j++)
  { // 1
    if (proc_sub_prob_count[j] % 2 != 0)
    { // 2
      if (proc_sub_prob_count[j + 1] % 2 != 0)
      { // 3
        if (proc_sub_prob_count[j + 1] >= proc_sub_prob_count[j])
        { // 4
          proc_sub_prob_count[j + 1] -= 1;
          proc_sub_prob_count[j] += 1;
          proc_y_sub_prob_count[j + 1] -= 1;
          proc_y_sub_prob_count[j] += 1;

        } // 4
        else
        { // 4
          false_flag = true;
        } // 4
      }   // 3
      else if (proc_sub_prob_count[j + 1] % 2 == 0)
      { // 3
        proc_sub_prob_count[j] -= 1;
        proc_sub_prob_count[procsize - 1] += 1;
        proc_y_sub_prob_count[j] -= 1;
        proc_y_sub_prob_count[procsize - 1] += 1;

      } // 3
      else
      { // 3
        false_flag = true;
      } // 3
    }   // 2
    else
    { // 2
      false_flag = true;
    } // 2

  } // 1
  if (false_flag)
    printf("False flag is True!!");

  // for (int i = 0; i < procsize; i++)
  // {
  //   std::cout << "proc  " << i << " x " << proc_sub_prob_count[i] << std::endl;
  //   std::cout << "proc  " << i << " y " << proc_y_sub_prob_count[i] << std::endl;
  // }

  ////////////////////////////////////////////

  for (int i = 0; i < procsize; i++)
  {
    int dev_y_overflow = proc_y_sub_prob_count[i] % devcount;
    int dev_overflow = proc_sub_prob_count[i] % devcount;

    for (int d = 0; d < devcount; d++)
    {
      dev_sub_prob_count[i][d] = dev_overflow >= (devcount - d) ? (proc_sub_prob_count[i] / devcount) + 1 : (proc_sub_prob_count[i] / devcount);
      dev_y_sub_prob_count[i][d] = dev_y_overflow >= (devcount - d) ? (proc_y_sub_prob_count[i] / devcount) + 1 : (proc_y_sub_prob_count[i] / devcount);
    }
  }

  for (int i = 0; i < procsize; i++)
  {
    int extra_dev_overflow_y = 0;
    int extra_dev_overflow_x = 0;
    for (int j = 0; j < devcount - 1; j++)
    {
      if (dev_sub_prob_count[i][j] > dev_y_sub_prob_count[i][j])
      {
        extra_dev_overflow_x += dev_sub_prob_count[i][j] - dev_y_sub_prob_count[i][j];
        dev_sub_prob_count[i][j] = dev_y_sub_prob_count[i][j];
      }
      if (dev_y_sub_prob_count[i][j] > dev_sub_prob_count[i][j])
      {
        extra_dev_overflow_y += dev_y_sub_prob_count[i][j] - dev_sub_prob_count[i][j];
        dev_y_sub_prob_count[i][j] = dev_sub_prob_count[i][j];
      }
    }
    dev_sub_prob_count[i][devcount - 1] += extra_dev_overflow_x;
    dev_y_sub_prob_count[i][devcount - 1] += extra_dev_overflow_y;

    for (int j = 0; j < devcount - 1; j++)
    {
      if (dev_sub_prob_count[i][j] % 2 != 0)
      {
        if (dev_sub_prob_count[i][j + 1] % 2 != 0)
        {
          if (dev_sub_prob_count[i][j + 1] >= dev_sub_prob_count[i][j])
          {
            dev_sub_prob_count[i][j + 1] -= 1;
            dev_sub_prob_count[i][j] += 1;
            dev_y_sub_prob_count[i][j + 1] -= 1;
            dev_y_sub_prob_count[i][j] += 1;
          }
          else
          {
            false_flag = true;
          }
        }
        else if (dev_sub_prob_count[i][j + 1] % 2 == 0)
        {
          dev_sub_prob_count[i][j] -= 1;
          dev_sub_prob_count[i][devcount - 1] += 1;
          dev_y_sub_prob_count[i][j] -= 1;
          dev_y_sub_prob_count[i][devcount - 1] += 1;
        }
        else
        {
          false_flag = true;
        }
      }
      else
      {
        false_flag = true;
      }
    }
  }
  int iter = 0;
  int iter_y = 0;
  int *iter_dev_y = new int[devcount];
  int *iter_dev_x = new int[devcount];

  for (int i = 0; i < procsize; i++)
  {
    dev_iterations[i] = new int[devcount];
    dev_iter_sub_prob_count[i] = new int *[devcount];
    n_dev_iter_ptr[i] = new int *[devcount];

    for (int d = 0; d < devcount; d++)
    {
      iter = 0;
      iter = dev_sub_prob_count[i][d] % MAX_SUBPROB_Y_GPU < MAX_SUBPROB_OVER_Y ? (dev_sub_prob_count[i][d] / MAX_SUBPROB_Y_GPU) : (dev_sub_prob_count[i][d] / MAX_SUBPROB_Y_GPU) + 1;
      iter = (iter == 0 && dev_sub_prob_count[i][d] > 0) ? 1 : iter;
      iter_dev_x[d] = iter;

      int iter_overflow = (iter == 0) ? 0 : dev_sub_prob_count[i][d] % iter;
      dev_iterations[i][d] = iter;

      dev_iter_sub_prob_count[i][d] = new int[iter];
      n_dev_iter_ptr[i][d] = new int[iter + 1];

      for (int j = 0; j < iter; j++)
      {
        dev_iter_sub_prob_count[i][d][j] = iter_overflow >= (iter - j) ? (dev_sub_prob_count[i][d] / iter) + 1 : (dev_sub_prob_count[i][d] / iter);
      }
    }
  }

  for (int i = 0; i < procsize; i++)
  {
    dev_y_iterations[i] = new int[devcount];
    dev_y_iter_sub_prob_count[i] = new int *[devcount];
    n_y_dev_iter_ptr[i] = new int *[devcount];

    for (int d = 0; d < devcount; d++)
    {
      iter_y = 0;
      iter_y = dev_y_sub_prob_count[i][d] % MAX_SUBPROB_Y_GPU < MAX_SUBPROB_OVER_Y ? (dev_y_sub_prob_count[i][d] / MAX_SUBPROB_Y_GPU) : (dev_y_sub_prob_count[i][d] / MAX_SUBPROB_Y_GPU) + 1;

      iter_y = (iter_y == 0 && dev_y_sub_prob_count[i][d] > 0) ? 1 : iter_y;
      iter_dev_y[d] = iter_y;

      int iter_y_overflow = (iter_y == 0) ? 0 : dev_y_sub_prob_count[i][d] % iter_y;

      dev_y_iterations[i][d] = iter_y;
      dev_y_iter_sub_prob_count[i][d] = new int[iter_y];

      n_y_dev_iter_ptr[i][d] = new int[iter_y + 1];

      for (int j = 0; j < iter_y; j++)
      {

        dev_y_iter_sub_prob_count[i][d][j] = iter_y_overflow >= (iter_y - j) ? (dev_y_sub_prob_count[i][d] / iter_y) + 1 : (dev_y_sub_prob_count[i][d] / iter_y);
      }
    }
  }

  for (int i = 0; i < procsize; i++)
  {
    int extra_dev_iter_overflow_y = 0;
    int extra_dev_iter_overflow_x = 0;
    for (int j = 0; j < devcount; j++)
    {
      // for (int k = 0; k < iter_y - 1; k++)
      for (int k = 0; k < iter_dev_y[j] - 1; k++)

      {
        if (dev_iter_sub_prob_count[i][j][k] > dev_y_iter_sub_prob_count[i][j][k])
        {
          extra_dev_iter_overflow_x += dev_iter_sub_prob_count[i][j][k] - dev_y_iter_sub_prob_count[i][j][k];
          dev_iter_sub_prob_count[i][j][k] = dev_y_iter_sub_prob_count[i][j][k];
        }
        if (dev_y_iter_sub_prob_count[i][j][k] > dev_iter_sub_prob_count[i][j][k])
        {
          extra_dev_iter_overflow_y += dev_y_iter_sub_prob_count[i][j][k] - dev_iter_sub_prob_count[i][j][k];
          dev_y_iter_sub_prob_count[i][j][k] = dev_iter_sub_prob_count[i][j][k];
        }
      }
      // dev_iter_sub_prob_count[i][j][iter_y - 1] += extra_dev_iter_overflow_x;
      // dev_y_iter_sub_prob_count[i][j][iter_y - 1] += extra_dev_iter_overflow_y;

      dev_iter_sub_prob_count[i][j][iter_dev_y[j] - 1] += extra_dev_iter_overflow_x;
      dev_y_iter_sub_prob_count[i][j][iter_dev_y[j] - 1] += extra_dev_iter_overflow_y;

      for (int k = 0; k < iter_y - 1; k++)
      {
        if (dev_iter_sub_prob_count[i][j][k] % 2 != 0)
        {
          if (dev_iter_sub_prob_count[i][j][k + 1] % 2 != 0)
          {
            if (dev_iter_sub_prob_count[i][j][k + 1] >= dev_iter_sub_prob_count[i][j][k])
            {
              dev_iter_sub_prob_count[i][j][k + 1] -= 1;
              dev_iter_sub_prob_count[i][j][k] += 1;
              dev_y_iter_sub_prob_count[i][j][k + 1] -= 1;
              dev_y_iter_sub_prob_count[i][j][k] += 1;
            }
            else
            {
              false_flag = true;
            }
          }
          else if (dev_iter_sub_prob_count[i][j][k + 1] % 2 == 0)
          {
            dev_iter_sub_prob_count[i][j][k] -= 1;
            dev_iter_sub_prob_count[i][j][iter_y - 1] += 1;
            dev_y_iter_sub_prob_count[i][j][k] -= 1;
            // dev_y_iter_sub_prob_count[i][j][iter_y - 1] += 1;
            dev_y_iter_sub_prob_count[i][j][iter_dev_y[j] - 1] += 1;
          }
          else
          {
            false_flag = true;
          }
        }
        else
        {
          false_flag = true;
        }
      }
    }
  }

  for (int i = 0; i < procsize; i++)
  {
    for (int d = 0; d < devcount; d++)
    {

      // std::copy(dev_iter_sub_prob_count[i][d], dev_iter_sub_prob_count[i][d] + iter, n_dev_iter_ptr[i][d]);
      // exclusiveSumScan(n_dev_iter_ptr[i][d], iter);
      // std::copy(dev_y_iter_sub_prob_count[i][d], dev_y_iter_sub_prob_count[i][d] + iter_y, n_y_dev_iter_ptr[i][d]);
      // exclusiveSumScan(n_y_dev_iter_ptr[i][d], iter_y);

      std::copy(dev_iter_sub_prob_count[i][d], dev_iter_sub_prob_count[i][d] + iter_dev_x[d], n_dev_iter_ptr[i][d]);
      exclusiveSumScan(n_dev_iter_ptr[i][d], iter_dev_x[d]);
      std::copy(dev_y_iter_sub_prob_count[i][d], dev_y_iter_sub_prob_count[i][d] + iter_dev_y[d], n_y_dev_iter_ptr[i][d]);
      exclusiveSumScan(n_y_dev_iter_ptr[i][d], iter_dev_y[d]);
    }
    proc_iterations[i] = dev_iterations[i][0];
    proc_iter_sub_prob_count[i] = new int[proc_iterations[i]];
    n_proc_iter_ptr[i] = new int[proc_iterations[i] + 1];
    for (int k = 0; k < proc_iterations[i]; k++)
    {
      int sum = 0;
      for (int d = 0; d < devcount; d++)
      {
        if (k < dev_iterations[i][d])
        {
          sum += dev_iter_sub_prob_count[i][d][k];

          int j = 0;
          while (j < dev_iter_sub_prob_count[i][d][k])
          {
            j++;
            if (dimension2 == (K - 1))
            {
              dimension1++;
              dimension2 = dimension1 + 1;
            }
            else
            {
              dimension2++;
            }
            spMap[0].dim1[spCount] = dimension1;
            spMap[0].dim2[spCount] = dimension2;
            spMap[0].procId[spCount] = i;
            spMap[0].devId[spCount] = d;
            spCount++;
          }
        }
      }

      proc_iter_sub_prob_count[i][k] = sum;
    }
    std::copy(proc_iter_sub_prob_count[i], proc_iter_sub_prob_count[i] + proc_iterations[i], n_proc_iter_ptr[i]);
    exclusiveSumScan(n_proc_iter_ptr[i], proc_iterations[i]);

    proc_y_iterations[i] = dev_y_iterations[i][0];
    proc_y_iter_sub_prob_count[i] = new int[proc_y_iterations[i]];
    n_y_proc_iter_ptr[i] = new int[proc_y_iterations[i] + 1];
    for (int k = 0; k < proc_y_iterations[i]; k++)
    {

      int sum = 0;
      for (int d = 0; d < devcount; d++)
      {
        if (k < dev_y_iterations[i][d])
        {
          sum += dev_y_iter_sub_prob_count[i][d][k];
          int j = 0;
          while (j < dev_y_iter_sub_prob_count[i][d][k])
          {
            j++;
            if (dimension1_y < (K - 2))
            {
              dimension1_y++;
            }
            y_spMap[0].dim1[y_spCount] = dimension1_y;
            y_spMap[0].procId[y_spCount] = i;
            y_spMap[0].devId[y_spCount] = d;
            y_spCount++;
          }
        }
      }
      proc_y_iter_sub_prob_count[i][k] = sum;
    }
    std::copy(proc_y_iter_sub_prob_count[i], proc_y_iter_sub_prob_count[i] + proc_y_iterations[i], n_y_proc_iter_ptr[i]);
    exclusiveSumScan(n_y_proc_iter_ptr[i], proc_y_iterations[i]);
  }

  std::copy(proc_sub_prob_count, proc_sub_prob_count + procsize, n_global_ptr);
  exclusiveSumScan(n_global_ptr, procsize);

  std::copy(proc_y_sub_prob_count, proc_y_sub_prob_count + procsize, n_y_global_ptr);
  exclusiveSumScan(n_y_global_ptr, procsize);

  for (int i = 0; i < procsize; i++)
  {
    for (int j = 0; j < proc_y_iterations[i] + 1; j++)
    {
      n_y_proc_iter_ptr[i][j] += n_y_global_ptr[i];
    }
  }

  for (int i = 0; i < procsize; i++)
  {
    for (int j = 0; j < proc_iterations[i] + 1; j++)
    {
      n_proc_iter_ptr[i][j] += n_global_ptr[i];
    }
  }

  /////////////////////PRINT INFORMATION ON DIVISION OF SUBPROBLEMS//////////////////////

  std::cout << "Maximum number of y subproblems that can fit in a single GPU device: " << MAX_SUBPROB_Y_GPU << std::endl;
  std::cout << "Memory required for one y subproblem: " << mem_required_per_subproblem_y << "  GB" << std::endl;

  std::cout << "Maximum number of x subproblems that can fit in a single GPU device: " << MAX_SUBPROB_Y_GPU << std::endl;
  std::cout << "NUmber of y subproblems that overflow: " << MAX_SUBPROB_OVER_Y << std::endl;

  for (int p = 0; p < procsize; p++)
  {
    for (int d = 0; d < devcount; d++)
    {
      std::cout << "Number of subprobems on device " << d << "\t" << dev_y_sub_prob_count[p][d] << std::endl;
      for (int i = 0; i < iter_dev_y[d]; i++)
      {
        std::cout << "Number of subprobems on device " << d << "iteration\t" << i << "\t" << dev_y_iter_sub_prob_count[p][d][i] << std::endl;
      }
    }
  }

  for (int p = 0; p < procsize; p++)
  {
    for (int d = 0; d < devcount; d++)
    {
      std::cout << "Number of subprobems on device " << d << "\t" << dev_sub_prob_count[p][d] << std::endl;
      for (int i = 0; i < iter_dev_x[d]; i++)
      {
        std::cout << "Number of subprobems on device " << d << "iteration\t" << i << "\t" << dev_iter_sub_prob_count[p][d][i] << std::endl;
      }
    }
  }

  for (int i = 0; i < procsize; i++)
  {
    for (int j = 0; j < proc_y_iterations[i]; j++)
    {
      std::cout << "Proc iter sub prob count x"
                << "iteration\t" << j << "\t" << proc_iter_sub_prob_count[i][j] << std::endl;
      std::cout << "Proc iter sub prob count t"
                << "iteration\t" << j << "\t" << proc_y_iter_sub_prob_count[i][j] << std::endl;
      std::cout << "Proc iter ptr "
                << "iteration\t" << j << "\t" << n_proc_iter_ptr[i][j] << std::endl;
      std::cout << "Proc y iter ptr "
                << "iteration\t" << j << "\t" << n_y_proc_iter_ptr[i][j] << std::endl;
    }
  }
}

void finalize()
{
  delete[] spMap[0].dim1;
  delete[] spMap[0].dim2;
  delete[] spMap[0].procId;
  delete[] spMap[0].devId;
  delete[] spMap;

  delete[] proc_sub_prob_count;
  delete[] proc_iterations;

  for (int i = 0; i < procsize; i++)
  {
    delete[] proc_iter_sub_prob_count[i];
    delete[] dev_sub_prob_count[i];
    delete[] dev_iterations[i];

    for (int j = 0; j < devcount; j++)
    {
      delete[] dev_iter_sub_prob_count[i][j];
      delete[] n_dev_iter_ptr[i][j];
    }

    delete[] dev_iter_sub_prob_count[i];
    delete[] n_dev_iter_ptr[i];
  }
  delete[] proc_iter_sub_prob_count;
  delete[] dev_sub_prob_count;
  delete[] dev_iterations;
  delete[] dev_iter_sub_prob_count;
  delete[] n_dev_iter_ptr;
  delete[] n_proc_iter_ptr;
  delete[] n_global_ptr;

  delete[] proc_y_iter_sub_prob_count;
  delete[] dev_y_sub_prob_count;
  delete[] dev_y_iterations;
  delete[] dev_y_iter_sub_prob_count;
  delete[] n_y_dev_iter_ptr;
  delete[] n_y_proc_iter_ptr;
  delete[] n_y_global_ptr;
}

double getUB_all_batches(double *h_x_costs, double *h_y_costs, int *h_row_assignments, int N)
{

  // USE SIZE_T TO AVOID INTEGER OVERFLOW ISSUES WITH Y COSTS
  std::size_t N1 = N;
  std::size_t K1 = K;
  std::size_t p = 0;
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t k = 0;
  double total_cost = 0;
  for (p = 0; p < K1 - 1; p++)
  {
    for (i = 0; i < N1; i++)
    {

      j = h_row_assignments[N1 * p + i];
      k = h_row_assignments[N1 * (p + 1) + j];

      total_cost += h_x_costs[p * N1 * N1 + N1 * i + j];
      total_cost += h_y_costs[(N1 * N1 * N1 * p) + (N1 * N1 * k) + (N1 * i) + j];
    }
  }
  for (i = 0; i < N1; i++)
  {
    j = h_row_assignments[N1 * (K1 - 1) + i];
    total_cost += h_x_costs[(K1 - 1) * N1 * N1 + N1 * i + j];
  }
  return total_cost;
}