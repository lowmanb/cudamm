#!/bin/bash
#SBATCH --nodelist=udc-ba30-7
#SBATCH --time=00:15:00 
#SBATCH --account=parallelcomputing
#SBATCH --part=training
#SBATCH --output=output

for i in {4,8,16,32}
do
	echo -e "\n$i"
	for j in {1..5}
	 do
		./mmm$i 10000 | tail -1 | cut -d' ' -f6
	done
done
