#!/bin/bash
#SBATCH --job-name=pof-manual-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GKSwstype=100
export GKS_WSTYPE=100

export POF_BACKEND="${POF_BACKEND:-slurm}"
export POF_PARALLEL="${POF_PARALLEL:-auto}"
export POF_RESOLUTION="${POF_RESOLUTION:-50}"
export POF_NUM_OBS="${POF_NUM_OBS:-3000}"
export POF_NUM_FEATURES="${POF_NUM_FEATURES:-8}"
export POF_SEED="${POF_SEED:-42}"
export POF_NOISE_SCALE="${POF_NOISE_SCALE:-0.5}"
export POF_RANGE_FRAC="${POF_RANGE_FRAC:-0.5}"
export POF_OUTDIR="${POF_OUTDIR:-slurm_manual_test/outputs}"
export POF_BASENAME="${POF_BASENAME:-objective_plot_manual}"
export POF_SUMMARY_NAME="${POF_SUMMARY_NAME:-objective_plot_manual_summary}"

julia --project=. slurm_manual_test/run_plot.jl
