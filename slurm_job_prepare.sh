#!/bin/bash
# Run script with model loaded directly into VRAM using Bender's local NVMe scratch
set -e  # Exit on error


# Set up local job directory and HuggingFace cache
###############################################################
# Use Bender's local NVMe for HuggingFace cache (ignores user's disk quota)
echo "Setting HuggingFace cache to local job directory..."
export JOB_DIRECTORY=/local/nvme/${USER}_${SLURM_JOB_ID}
export HF_HOME=$JOB_DIRECTORY/hf_cache
mkdir -p $HF_HOME

# Copy HuggingFace token from home directory if it exists
if [ -f ~/.cache/huggingface/token ]; then
    cp ~/.cache/huggingface/token $HF_HOME/token
    echo "Copied HF token to job directory"
elif [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN from environment"
else
    echo "No HF token found. Public models will work, but private models may fail."
fi
###############################################################


# Prepare Python environment
###############################################################
# Load conda and activate environment
echo "Loading conda and activating environment..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh" 
conda activate lab-ai

# Disable tokenizers parallelism to avoid issues
export TOKENIZERS_PARALLELISM=false

# Python optimization flags - ensure all output is visible
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$JOB_DIRECTORY/pycache
export PYTHONUNBUFFERED=1

module load CUDA/12.4.0
#################################################################


echo "========================================"
echo "Using local job directory: $JOB_DIRECTORY"
echo "HuggingFace cache: $HF_HOME"
echo "Python pycache: $PYTHONPYCACHEPREFIX"
echo "========================================"