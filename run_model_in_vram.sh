#!/bin/bash
# Run script with model loaded directly into VRAM using Bender's local NVMe scratch
set -e  # Exit on error

# Load CUDA and activate environment
echo "Loading CUDA module..."
module load CUDA/12.4.0

echo "Activating Python environment..."
source /home/s37mfese/MA-NF_4335-Lab_AI_Alignment/liars-bench/.venv/bin/activate

# Use Bender's local NVMe for HuggingFace cache (fastest, auto-cleanup)
export LOCAL_SCRATCH=/local/nvme/${USER}_${SLURM_JOB_ID}
export HF_HOME=$LOCAL_SCRATCH/hf_cache

mkdir -p $HF_HOME

# Copy HuggingFace token from home directory if it exists
if [ -f ~/.cache/huggingface/token ]; then
    mkdir -p $HF_HOME
    cp ~/.cache/huggingface/token $HF_HOME/token
    echo "✓ Copied HF token to job directory"
elif [ -n "$HF_TOKEN" ]; then
    echo "✓ Using HF_TOKEN from environment"
else
    echo "⚠ No HF token found. Public models will work, but private models may fail."
fi

# Disable tokenizers parallelism to avoid issues
export TOKENIZERS_PARALLELISM=false

# Python optimization flags - ensure all output is visible
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$LOCAL_SCRATCH/pycache
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Using Bender local NVMe: $LOCAL_SCRATCH"
echo "HuggingFace cache: $HF_HOME"
echo "========================================"

cd /home/s37mfese/MA-NF_4335-Lab_AI_Alignment/liars-bench/Lab-AI-Alignment
python -u main_template.py

echo "========================================"
echo "Job complete. Bender will auto-cleanup $LOCAL_SCRATCH at job end."
echo "========================================"
