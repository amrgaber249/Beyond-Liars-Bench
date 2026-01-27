#!/bin/bash
#SBATCH --job-name="Linear Probe training with Llama-3.3-70B"

#SBATCH --partition=A40short
#SBATCH --time=04:30:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1

#SBATCH --output="slurm_logs/slurm_%j.out"
#SBATCH --error="slurm_logs/slurm_%j.err"

set -e  # Exit on error

echo "Setting up job environment..."
source ./slurm_job_prepare.sh

echo "Starting probe training script..."
python3 probes.py