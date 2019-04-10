#!/bin/bash
#SBATCH --account=COMP90024
#SBATCH --partition=cloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:10:00

# Load required modules
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA9

# Launch multiple process python code
echo "using 1 node with 1 core"
echo "Process tinyTwitter.json"
time mpiexec python proj2.py -t "tinyTwitter.json" -g "melbGrid.json"
echo "Process smallTwitter.json"
time mpiexec python proj2.py -t "smallTwitter.json" -g "melbGrid.json"
echo "Process bigTwitterjson"
time mpiexec python proj2.py -t "bigTwitter.json" -g "melbGrid.json"
