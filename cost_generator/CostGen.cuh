#pragma once
#include <random>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <thread>
#include <math.h>
// #include "./alglib/interpolation.h"
#include "defs.cuh"

class CostGen
{
private:
  std::default_random_engine generator1, generator2;
  int num_targets, num_frames, num_polynom_coeffs;
  double *target_velocities, *translation_offsets, *rotation_angles, *trajectory_coeffs;
  std::vector<std::vector<Position>> tracking_data;
  uint seed1, seed2;
  DistanceType dtype;

  void generate_tracking_data();
  void generatePolynomialTrajectories();
  void generateTrackingGT();
  void rotate();
  void translate();
  void populateVectors();
  void scrambleDetection(Position &pos);
  void addNoise();
  double getY(double x, double *trajectory_coeffs);
  void generate_scores();
  double euclideanDistance(Position pos1, Position pos2)
  {
    return std::sqrt(std::pow(pos1.x - pos2.x, 2) + std::pow(pos1.y - pos2.y, 2));
  }
  double rectilinearDistance(Position pos1, Position pos2)
  {
    return std::abs(pos1.x - pos2.x) + std::abs(pos1.y - pos2.y);
  }
  double splineDistance(Position pos1, Position pos2)
  {
    return euclideanDistance(pos1, pos2);
  }

public:
  Position *tracking_gt;
  double *X_costs, *Y_costs;
  // constructor
  CostGen(int _num_targets, int _num_frames, int _polynom_deg, DistanceType dtype_, uint _seed1, uint _seed2)
  {
    num_targets = _num_targets;
    num_frames = _num_frames;
    num_polynom_coeffs = _polynom_deg + 1;
    dtype = dtype_;
    seed1 = _seed1;
    seed2 = _seed2;
    tracking_gt = new Position[num_targets * num_frames];
    trajectory_coeffs = new double[num_targets * num_polynom_coeffs];
    target_velocities = new double[num_targets];
    translation_offsets = new double[num_targets];
    rotation_angles = new double[num_targets];
    generator1 = std::default_random_engine(seed1);
    generator2 = std::default_random_engine(seed2);
  }
  // destructor
  ~CostGen()
  {
    delete[] tracking_gt;
    delete[] trajectory_coeffs;
    delete[] target_velocities;
    delete[] translation_offsets;
    delete[] rotation_angles;
  }
  void generate_scores(double *xcost_matrix, double *ycost_matrix, int SP_x, int SP_y);
  void generate_tracks()
  {
    std::uniform_real_distribution<double> distribution(V_LB, V_UB);
    for (int i = 0; i < num_targets; i++)
      target_velocities[i] = distribution(generator1) * VELOCITY_MULTIPLIER;

    double sigma = POLYNOM_SIG;

    for (int p = 0; p < num_polynom_coeffs; p++) // get polynomial trajectory coefficients
    {
      std::uniform_real_distribution<double> distribution(-sigma, sigma);

      for (int i = 0; i < num_targets; i++)
      {
        double coeff = distribution(generator1);
        trajectory_coeffs[i * num_polynom_coeffs + p] = coeff;
      }
      sigma /= 4;
    }
    // uint particle = 2;
    generateTrackingGT(); // get coordinates dx, dy etc

    // for (int k = 0; k < K; k++)
    //   std::cout << tracking_gt[particle * K + k].x << "\t" << tracking_gt[particle * K + k].y << "\n";
    // std::cout << std::endl
    //           << std::endl;
    rotate();

    // for (int k = 0; k < K; k++)
    //   std::cout << tracking_gt[particle * K + k].x << "\t" << tracking_gt[particle * K + k].y << "\n";
    // std::cout << std::endl
    //           << std::endl;

    translate();

    // for (int k = 0; k < K; k++)
    //   std::cout << tracking_gt[particle * K + k].x << "\t" << tracking_gt[particle * K + k].y << "\n";
    // std::cout << std::endl
    //           << std::endl;

    populateVectors(); // copy positions to Position vector
    if (num_targets < 50)
      addNoise(); // Add standard normally distributed noise

    // for (int k = 0; k < K; k++)
    //   std::cout << tracking_gt[particle * K + k].x << "\t" << tracking_gt[particle * K + k].y << "\n";
    // std::cout << std::endl
    //           << std::endl;
  }
  double getDistance(Position pos1, Position pos2);
  void filterUngated(double *ycosts, std::vector<uint> *ungated, int SP_x, int SP_y);
};

double CostGen::getDistance(Position pos1, Position pos2)
{
  double distance = 0.0;
  switch (dtype)
  {
  case EUCLID:
    distance = euclideanDistance(pos1, pos2);
    break;
  case RECTILINEAR:
    distance = rectilinearDistance(pos1, pos2);
    break;
  case SPLINE:
    distance = splineDistance(pos1, pos2);
    break;
  default:
    distance = euclideanDistance(pos1, pos2);
    break;
  }
  return distance;
}

double CostGen::getY(double x, double *trajectory_coeffs)
{

  double y = 0.0;
  for (int p = 0; p < num_polynom_coeffs; p++)
    y += trajectory_coeffs[p] * std::pow(x, (double)p);

  return y;
}

void CostGen::generateTrackingGT()
{

  double stepsize = (double)1 / PTS_PER_UNIT;

  // std::cout << "Trajectory coeff: " << std::endl;
  // for (int i = 0; i < num_polynom_coeffs; i++)
  // {
  //   std::cout << trajectory_coeffs[1 * num_polynom_coeffs + i] << "\t";
  // }
  std::cout << std::endl;
  uint nthreads = std::min((uint)num_targets, (uint)std::thread::hardware_concurrency() - 3);
#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < num_targets; i++)
  {

    double velocity = target_velocities[i];

    Position *target_ptr = &tracking_gt[i * num_frames];
    Position *target_ptr_end = &tracking_gt[(i + 1) * num_frames];

    double x = X_LB;
    double y = getY(x, &trajectory_coeffs[i * num_polynom_coeffs]);

    while (target_ptr != target_ptr_end)
    {
      target_ptr->x = x;
      target_ptr->y = y;
      target_ptr->id = i;

      double x_rev = x;
      double y_rev = y;

      bool flag = true;
      bool flag_rev = true;

      double dist = 0.0;
      double dist_rev = 0.0;

      while (dist < velocity)
      {

        double cur_x = x;
        double cur_y = y;

        double cur_x_rev = x_rev;
        double cur_y_rev = y_rev;

        x += stepsize;
        y = getY(x, &trajectory_coeffs[i * num_polynom_coeffs]);

        dist += std::sqrt(std::pow(cur_x - x, 2) + std::pow(cur_y - y, 2));

        if (flag && (dist > velocity * stepsize))
        {
          target_ptr->dx = x;
          target_ptr->dy = y;
          flag = false;
        }

        x_rev -= stepsize;
        y_rev = getY(x_rev, &trajectory_coeffs[i * num_polynom_coeffs]);
        dist_rev += std::sqrt(std::pow(cur_x_rev - x_rev, 2) + std::pow(cur_y_rev - y_rev, 2));
        if (flag_rev && (dist_rev > velocity * stepsize))
        {
          target_ptr->d_x = x_rev;
          target_ptr->d_y = y_rev;
          flag_rev = false;
        }
      }

      target_ptr++;
    }
  }
}

void CostGen::rotate()
{
  std::uniform_real_distribution<double> distribution(ROTATE_LB, ROTATE_UB);
  uint nthreads = std::min((uint)num_targets, (uint)std::thread::hardware_concurrency() - 3);
#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < num_targets; i++)
  {

    double theta = distribution(generator1);

    Position *pos = &tracking_gt[i * num_frames];
    Position *pos_end = &tracking_gt[(i + 1) * num_frames];

    while (pos != pos_end)
    {

      double x = pos->x;
      double y = pos->y;
      double dx = pos->dx;
      double dy = pos->dy;
      double d_x = pos->d_x;
      double d_y = pos->d_y;

      double sine = sin(theta * PI / 180.0);
      double cosine = cos(theta * PI / 180.0);

      pos->x = x * cosine - y * sine;
      pos->y = x * sine + y * cosine;

      pos->dx = dx * cosine - dy * sine;
      pos->dy = dx * sine + dy * cosine;

      pos->d_x = d_x * cosine - d_y * sine;
      pos->d_y = d_x * sine + d_y * cosine;

      rotation_angles[i] = theta;

      pos++;
    }
  }
}

void CostGen::translate()
{
  std::uniform_real_distribution<double> distribution(TRANSLATE_LB, TRANSLATE_UB);
  uint nthreads = std::min((uint)num_targets, (uint)std::thread::hardware_concurrency() - 3);
#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < num_targets; i++)
  {

    double delta = distribution(generator1);

    Position *pos = &tracking_gt[i * num_frames];
    Position *pos_end = &tracking_gt[(i + 1) * num_frames];

    while (pos != pos_end)
    {

      double x = pos->x;
      double y = pos->y;
      double dx = pos->dx;
      double dy = pos->dy;
      double d_x = pos->d_x;
      double d_y = pos->d_y;

      pos->x = x + delta;
      pos->y = y + delta;

      pos->dx = dx + delta;
      pos->dy = dy + delta;

      pos->d_x = d_x + delta;
      pos->d_y = d_y + delta;

      translation_offsets[i] = delta;

      pos++;
    }
  }
}

void CostGen::populateVectors()
{

  std::uniform_int_distribution<int> distribution(0, std::numeric_limits<int>::max());

  // transpose
  for (int t = 0; t < num_frames; t++)
  {
    std::vector<Position> pos;
    for (int i = 0; i < num_targets; i++)
    {
      int id = i * num_frames + t;
      pos.push_back(tracking_gt[id]);
    }

    unsigned seed = distribution(generator1);
    std::shuffle(pos.begin(), pos.end(), std::default_random_engine(seed));

    tracking_data.push_back(pos);
  }
}

void CostGen::scrambleDetection(Position &pos)
{

  auto generator = std::default_random_engine(55345 + (uint)ceil(pos.x + pos.y));
  std::uniform_real_distribution<double> distribution1(0, 1);

  // Translate
  double std_dev_X = max(1.0, min((std::abs(pos.x)) / 50, 10.0)); // Higher deviation further from origin
  double std_dev_Y = max(1.0, min((std::abs(pos.y)) / 50, 10.0)); // Higher deviation further from origin

  std::normal_distribution<double> distributionX(TRANSLATE_LB, std_dev_X);
  std::normal_distribution<double> distributionY(TRANSLATE_LB, std_dev_Y);
  double deltaX = distributionX(generator);
  double deltaY = distributionX(generator);
  // Log(debug, "Delta %f\n", delta);
  int sign = distribution1(generator) < 0.5 ? 1 : -1;
  pos.x += sign * deltaX;
  sign = distribution1(generator) < 0.5 ? 1 : -1;
  pos.y += sign * deltaY;
  sign = distribution1(generator) < 0.5 ? 1 : -1;
  pos.dx += sign * deltaX;
  sign = distribution1(generator) < 0.5 ? 1 : -1;
  pos.dy += sign * deltaY;
  sign = distribution1(generator) < 0.5 ? 1 : -1;
  pos.d_x += sign * deltaX;
  sign = distribution1(generator) < 0.5 ? 1 : -1;
  pos.d_y += sign * deltaY;
}

void CostGen::addNoise()
{
  std::uniform_real_distribution<double> probability_gen(0, 1);
  uint nthreads = std::min((uint)num_targets, (uint)std::thread::hardware_concurrency() - 3);
#pragma omp parallel for num_threads(nthreads)
  for (uint i = 0; i < num_targets; i++)
  {
    for (uint k = 0; k < num_frames; k++)
    {
      double prob = probability_gen(generator1);
      if (prob < NOISE_PROBABILITY)
      {
        // Position pos = tracking_gt[i * num_frames + k];
        scrambleDetection(tracking_gt[i * num_frames + k]);
      }
    }
  }
}

void CostGen::generate_scores(double *xcost_matrix, double *ycost_matrix, int SP_x, int SP_y)
{
  X_costs = xcost_matrix;
  Y_costs = ycost_matrix;
  // uint nthreads = std::min((uint)SP_y, (uint)std::thread::hardware_concurrency() - 3);
  uint nthreads = 1;
  uint ungated = 0, gated = 0;
  std::cout << "Nthreads available: " << nthreads << std::endl;
  uint rows_per_thread = ceil(((SP_x - 1) * 1.0) / nthreads);
  size_t N = num_targets, K = num_frames;

#pragma omp parallel for num_threads(nthreads)
  for (uint tid = 0; tid < nthreads; tid++)
  {

    uint first_row = tid * rows_per_thread;
    uint last_row = std::min(first_row + rows_per_thread, (uint)SP_x - 1);
    // std::mt19937 gen(seed1 + tid *);
    // gen.discard(1);
    for (uint p = first_row; p < last_row; p++)
    {
      for (size_t i = 0; i < N; i++)
      {
        for (size_t j = 0; j < N; j++)
        {
          Position pos1 = tracking_gt[i * K + p];
          Position pos2 = tracking_gt[j * (K) + p + 1];
          double value = getDistance(pos1, pos2);
          // double value = rectilinearDistance(pos1, pos2);

          std::size_t index = (p * N * N) + (i * N) + j;
          X_costs[index] = value;
          //		std::cout<<cost_matrix[index]<<std::endl;
        }
      }
    }
  }
  std::cout << "x costs generated" << std::endl;

  rows_per_thread = ceil(((SP_y - 1) * 1.0) / nthreads);
#pragma omp parallel for num_threads(nthreads)
  for (uint tid = 0; tid < nthreads; tid++)
  {
    uint first_row = tid * rows_per_thread;
    uint last_row = std::min(first_row + rows_per_thread, (uint)SP_y - 1);
    for (uint p2 = first_row; p2 < last_row; p2++)
    {
      for (uint i2 = 0; i2 < N; i2++)
      {
        for (uint j2 = 0; j2 < N; j2++)
        {
          for (uint k2 = 0; k2 < N; k2++)
          {

            Position pos1 = tracking_gt[i2 * K + p2];
            Position pos2 = tracking_gt[j2 * K + p2 + 1];
            Position pos3 = tracking_gt[k2 * K + p2 + 2];
            // double value = euclideanDistance(pos1, pos2) + euclideanDistance(pos2, pos3);
            double value = 0.0;
            // if (true)
            // if (getDistance(pos1, pos2) < V_UB * 10 && getDistance(pos2, pos3) < V_UB * 10)
            // {
            value = std::abs(getDistance(pos1, pos2) - getDistance(pos2, pos3));
            ++ungated;
            // }
            // else
            // {
            //   value = 1E8;
            //   ++gated;
            // }
            std::size_t id = N * N * N * (p2) + (k2 * N * N) + (i2 * N) + j2;
            Y_costs[id] = value;
            //	std::cout<<y_costs[id]<<std::endl;
          }
        }
      }
    }
  }
  std::cout << "y costs generated" << std::endl;
  std::cout << "amount of y costs filtered: " << gated * 100.0 / (ungated + gated) << " %" << std::endl;

  for (size_t i3 = 0; i3 < N; i3++)
  {
    for (size_t j3 = 0; j3 < N; j3++)
    {
      std::size_t index = ((SP_x - 1) * N * N) + (i3 * N) + j3;
      X_costs[index] = 0;
      for (size_t k3 = 0; k3 < N; k3++)
      {
        std::size_t id = N * N * N * (SP_y - 1) + (k3 * N * N) + (i3 * N) + j3;
        Y_costs[id] = 0;
      }
    }
  }
  std::cout << "last costs generated" << std::endl;
}

void CostGen::filterUngated(double *ycosts, std::vector<uint> *ungated, int SP_x, int SP_y)
{
  size_t N = num_targets;
  // uint nthreads = std::min((uint)SP_y, (uint)std::thread::hardware_concurrency() - 3);
  uint nthreads = 1;
  uint rows_per_thread = ceil(((SP_y - 1) * 1.0) / nthreads);
  std::cout << "Nthreads available: " << nthreads << std::endl;
#pragma omp parallel for num_threads(nthreads)
  for (uint tid = 0; tid < nthreads; tid++)
  {
    uint first_row = tid * rows_per_thread;
    uint last_row = std::min(first_row + rows_per_thread, (uint)SP_y - 1);
    for (size_t p = first_row; p < last_row; p++)
    {
      for (size_t k = 0; k < N; k++)
      {
        double *lcosts = &ycosts[p * N * N * N + k * N * N];
        for (uint i = 0; i < N; i++)
        {
          for (uint j = 0; j < N; j++)
          {
            if (lcosts[i * N + j] != 1E8)
            {
              uint number = (((uint)i << 16) | (uint)j);
              ungated[p * N + k].push_back(number);
              // if (p == 1 && k == 1)
              //   std::cout << number << std::endl;
            }
          }
        }
        // std::cout << p << ". " << k << ": " << ungated[p * N + k].size() << std::endl;
      }
    }
  }
}
