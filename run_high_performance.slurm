#!/usr/bin/env bash
#SBATCH --job-name=rt-marabou
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mode=${1:?Pass small or large}
case "$mode" in small|large) ;; *) exit 2;; esac
source "$SLURM_SUBMIT_DIR/runtime.sh"
srun "$PYTHON_BIN" -u pilot.py "$@"
