# Multi-dimensional-Assignment
A GPU accelerated algorithm for a multi dimensional assignment problem  (NP-Hard) using dual ascent on a linearized formulation. 

The Multi- dimensional Assignment Problem (MAP) is NP-Hard even for 3 dimensions. A heuristic is developed to find the closest possible solution. A creative linearization is applied to the formulation and the dual ascent technique is applied for the multiplier update scheme. As the variables explode with growing problem size, the algorithm is accelerated using CUDA and MPI, where data parallelism is induced at the high granular task level. The code is written in C++ and uses CUDA for thread level parallelism on the GPUs. An application to MAP is the Multi-Target Tracking problem (MTT)
