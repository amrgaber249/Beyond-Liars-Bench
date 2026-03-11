#!/bin/bash
#SBATCH --job-name="Preloading Extractions"

#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1

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

# echo "Starting extraction"
# python3 preload_extractions.py

# echo "Starting probe training script..."

# "mean_probe",       # Uses TRAIN_ACTIVATION_METHOD (Logistic Reg)
# "mass_mean_probe",  # Uses TRAIN_ACTIVATION_METHOD (Centroid)
# "followup_probe",   # Uses specific "followup_token" method (Prompting)
# "upper_bound"       # Uses specific "followup_token" method
# python3 probes.py followup_probe,mean_probe,mass_mean_probe,upper_bound
