# !/bin/bash
# module load mpi
# make clean all
# COST_PATH=/home/samiran2/Downloads/new/Multi-dimensional-Assignment-master/

# mpirun -np 1 test 5 5 1 100 0 0 0 1 ${COST_PATH}/Costs/costs_linear_triplet_K5_N5-0.txt ${COST_PATH}/Costs/outputcosts_Samhita_K5_N5_S0_problem0.txt
mpirun -np 1 ./test $1 $1 1 100 0 0 0 0 0 0