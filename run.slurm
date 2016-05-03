#!/bin/bash
#SBATCH --nodelist=udc-ba30-7,udc-ba30-5
#SBATCH --time=00:10:00 
#SBATCH --account=parallelcomputing
#SBATCH --part=training
#SBATCH --output=output


module load cuda

nvcc -O3 -o mmm mmm.cu

for size in `seq 1000 1000 10000`
do
    {
        echo "mmm"
        ./mmm $size
        echo -e "\nyanMMM.o"
        ./yanMMM.o $size
        echo -e "\ncublasMMM.o"
        ./cublasMMM.o $size
    } > data/rivanna/size-$size
done
