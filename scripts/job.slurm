#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o slurm_%j.out

srun /lustre/train.sh