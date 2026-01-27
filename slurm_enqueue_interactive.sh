#!/bin/bash
# Enqueue an interactive SLURM session with the job environment prepared
preparation="./slurm_job_prepare.sh"

echo "After running this command, you will be placed in an interactive SLURM session."
echo "Once inside, run the following commands to set up the environment:"
echo "conda init base"
echo "conda activate lab-ai"

srun --pty --partition=A40devel --time=01:00:00 --gpus=2 --ntasks=1 bash -c ". $preparation && /bin/bash"