#!/bin/bash

sbatch slurm_enqueue_probe_followup.sh 
sbatch slurm_enqueue_probe_mean.sh
sbatch slurm_enqueue_probe_mass_mean.sh
sbatch slurm_enqueue_probe_upper_bound.sh