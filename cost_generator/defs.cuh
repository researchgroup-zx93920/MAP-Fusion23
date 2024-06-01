#pragma once

struct Position
{
  double x; // x position [based on timeframe]
  double y; // y position [based on polynomial]
  double dx;
  double dy;
  double d_x;
  double d_y;
  int id;
};

#define V_LB 1.0
#define V_UB 10.0
#define VELOCITY_MULTIPLIER 1.0
#define POLYNOM_SIG 4

#define PTS_PER_UNIT 10
#define X_LB -10.0
#define ROTATE_LB -180.0
#define ROTATE_UB 180.0
#define TRANSLATE_LB 0.0
#define TRANSLATE_UB 10.0

#define NOISE_PROBABILITY 1.0

#define PI 3.14159265

enum DistanceType
{
  EUCLID,
  RECTILINEAR,
  SPLINE
};

#define GATING
// #define OLD_COSTS