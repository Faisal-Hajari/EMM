#!/bin/bash

#SBATCH --job-name=simmim_train_expand_cifar  # Name for your job
#SBATCH --time=04:00:00   # Adjust wallclock time as needed (4 hours here)
#SBATCH --cpus-per-task=32  # Request 8 CPU cores (adjust based on your needs)
#SBATCH --gpus=8           # Request 1 GPU
#SBATCH --mem=32G          # Memory allocation (adjust based on your program's needs)
#SBATCH --output=simmim_train_expand_cifar%j.out  # Output file with job ID (%j)
#SBATCH --error=simmim_train_expand_cifar%j.err   # Error file with job ID (%j)

# Execute your command using srun for distributed launch
torchrun --nproc_per_node 8 main_simmim.py --cfg config/pretraining_simim_vit_base_image32_800.yaml
