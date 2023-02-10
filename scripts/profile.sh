# !/bin/bash
cd ..
# make clean all
nsys profile --stats=true --show-output=true --trace=cuda,cublas ./test 30 30 1 100 0 0 0 0 0 0

# rm *.nsys-rep *.sqlite # Results/*.txt