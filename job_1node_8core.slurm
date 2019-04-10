#!/bin/bash
#SBATCH --account=COMP90024	
#SBATCH --partition=longcloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:10:00

# Load required modules
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA9

# Launch multiple process python code
echo "using 1 node with 8 core"
echo "Process tinyTwitter.json"
time mpiexec python proj3.py -t "tinyTwitter.json" -g "melbGrid.json"
echo
echo "Process smallTwitter.json"
time mpiexec python proj3.py -t "smallTwitter.json" -g "melbGrid.json"
echo
echo "Process bigTwitterjson"
time mpiexec python proj3.py -t "bigTwitter.json" -g "melbGrid.json"