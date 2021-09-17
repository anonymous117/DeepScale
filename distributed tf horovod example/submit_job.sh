#!/bin/sh

# Slurm job configuration
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.er
#SBATCH --time=00:30:00
#SBATCH --job-name=HOROVOD_MNIST
#SBATCH --gres=gpu:1 --partition=dc-gpu-devel

# Load the required modules
module load GCC/9.3.0
module load OpenMPI/4.1.0rc1
module load Horovod/0.20.3-Python-3.8.5
module load TensorFlow/2.3.1-Python-3.8.5

# Run the program
srun python -u mnist_distributed.py
