#!/bin/bash
#SBATCH --mem=20GB
#SBATCH --job-name=percept
#SBATCH --cpus-per-task=20
#SBATCH --partition=guppy,chipmunk,mammoth
#SBATCH --time=7-00:00:00
#SBATCH --qos=big_batch_tier
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --array=0-48

python compute_spectra_fooof_bandpower.py $SLURM_ARRAY_TASK_ID