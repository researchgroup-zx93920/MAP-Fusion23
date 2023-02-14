# !/bin/bash
cd ..
# make clean all
nsys profile --stats=true --show-output=true \
  ./build/exe/main.cu.exe $1 $1 1 100 0 0 0 1 \
  ./costs/Ycosts_N$1_K$1.txt ./costs/Xcosts_N$1_K$1.txt

rm *.nsys-rep *.sqlite # Results/*.txt