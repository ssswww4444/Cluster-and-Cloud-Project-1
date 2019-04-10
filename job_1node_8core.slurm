#!/bin/bash
#SBATCH --account=COMP90024	
#SBATCH --partition=physical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:10:00

# Load required modules
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA9

# Launch multiple process python code
echo "using 1 node with 8 core"
echo "Process tinyTwitter.json"
echo "physical partition"
time mpirun python proj.py -t "tinyTwitter.json" -g "melbGrid.json"
echo
echo "Process smallTwitter.json"
time mpirun python proj.py -t "smallTwitter.json" -g "melbGrid.json"
echo
echo "Process bigTwitterjson"
time mpirun python proj.py -t "bigTwitter.json" -g "melbGrid.json"
echo