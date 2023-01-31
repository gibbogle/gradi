#!/bin/bash 
#SBATCH --time=1:00:00
#SBATCH -A nesi99999         # Project Account
#SBATCH --mem-per-cpu=5G              # Memory
#SBATCH --cpus-per-task=1     # number of threads
#SBATCH --hint=nomultithread


cd test
srun ../main.exe

