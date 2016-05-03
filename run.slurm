#!/bin/bash
#SBATCH --nodelist=udc-ba30-7,udc-ba30-5
#SBATCH --time=00:10:00 
#SBATCH --account=parallelcomputing
#SBATCH --part=training
#SBATCH --output=output

{
    hostname
    bin/rivanna/deviceQuery
} > data/rivanna/hw-spec

for size in `seq 1000 1000 10000`
do
    {
        echo "mmm"
        bin/rivanna/mmm $size
        echo -e "\nyanMMM"
        bin/rivanna/yanMMM $size
        echo -e "\ncublasMMM"
        bin/rivanna/cublasMMM $size
    } > data/rivanna/size-$size
done
