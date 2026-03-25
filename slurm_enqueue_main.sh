#!/bin/bash
#SBATCH --job-name="LiarsBench"

#SBATCH --partition=A40medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --exclude=node-02

#SBATCH --output="slurm_logs/slurm_%j.out"
#SBATCH --error="slurm_logs/slurm_%j.out"

set -e  # Exit on error

echo "Setting up job environment..."
source ./slurm_job_prepare.sh

echo "==================== GPU Verification ==="
echo "Checking nvidia-smi..."
nvidia-smi
echo ""
echo "Checking PyTorch GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: No CUDA devices detected!')
"
echo "========================================"
echo ""

echo "Starting probe training script..."
cd final_code
python3 main.py