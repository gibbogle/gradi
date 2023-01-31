#!/bin/bash 
#SBATCH --gpus-per-node=P100:1
#SBATCH --time=10:00
#SBATCH -A nesi99999         # Project Account
#SBATCH --mem-per-cpu=2G              # Memory
#SBATCH --cpus-per-task=1     # number of threads


module load NVHPC
cd test
srun ../maingpu.exe

