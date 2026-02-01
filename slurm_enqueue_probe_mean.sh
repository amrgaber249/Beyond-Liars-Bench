#!/bin/bash
#SBATCH --job-name="Mean Probe Llama-3.3-70B"

#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1
#SBATCH --exclude=node-02

#SBATCH --output="slurm_logs/slurm_%j.out"
#SBATCH --error="slurm_logs/slurm_%j.out"

set -e  # Exit on error

echo "Setting up job environment..."
source ./slurm_job_prepare.sh

echo "Starting probe training script..."

# "mean_probe",       # Uses TRAIN_ACTIVATION_METHOD (Logistic Reg)
# "mass_mean_probe",  # Uses TRAIN_ACTIVATION_METHOD (Centroid)
# "followup_probe",   # Uses specific "followup_token" method (Prompting)
# "upper_bound"       # Uses specific "followup_token" method
python3 probes.py mean_probe