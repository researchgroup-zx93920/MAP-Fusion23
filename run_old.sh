# !/bin/bash

#ulimit -s umlimited
#For smaller tests: 5, 10, 20, 30, 50

mpirun -np 1 test 5 5 1 100 0 0 0 1 /home/mpiuser/mpi_workdir/Samhita/Costs/costs_linear_triplet_K5_N5-0.txt /home/mpiuser/mpi_workdir/Samhita/Costs/outputcosts_Samhita_K5_N5_S0_problem0.txt 
mpirun -np 1 test 5 5 1 100 1 0 0 1 /home/mpiuser/mpi_workdir/Samhita/Costs/costs_linear_triplet_K5_N5-1.txt /home/mpiuser/mpi_workdir/Samhita/Costs/outputcosts_Samhita_K5_N5_S0_problem1.txt
mpirun -np 1 test 5 5 1 100 2 0 0 1 /home/mpiuser/mpi_workdir/Samhita/Costs/costs_linear_triplet_K5_N5-2.txt /home/mpiuser/mpi_workdir/Samhita/Costs/outputcosts_Samhita_K5_N5_S0_problem2.txt
mpirun -np 1 test 5 5 1 100 3 0 0 1 /home/mpiuser/mpi_workdir/Samhita/Costs/costs_linear_triplet_K5_N5-3.txt /home/mpiuser/mpi_workdir/Samhita/Costs/outputcosts_Samhita_K5_N5_S0_problem3.txt
mpirun -np 1 test 5 5 1 100 4 0 0 1 /home/mpiuser/mpi_workdir/Samhita/Costs/costs_linear_triplet_K5_N5-4.txt /home/mpiuser/mpi_workdir/Samhita/Costs/outputcosts_Samhita_K5_N5_S0_problem4.txt


#For larger tests: 100, 200, 300, 400
mpirun -np 1 test 200 200 1 100 0 0 0 0 0 0



# /usr/local/bin/mpiexec -np 1 -hostfile /home/mpiuser/mpi_workdir/hostfile -x LD_LIBRARY_PATH=/usr/local/cuda/lib64 /home/mpiuser/mpi_workdir/Samhita/HMN_thrust_combined/tests/test 10 5 1 100 /home/mpiuser/mpi_workdir/Samhita/Costs/costs_linear_triplet_K5_N10-0.txt /home/mpiuser/mpi_workdir/Samhita/Costs/outputcosts_Samhita_K5_N10_S0_problem0.txt 0 0
