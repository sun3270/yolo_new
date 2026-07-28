#!/bin/bash
#SBATCH --job-name=coffee26n_full_b96
#SBATCH --partition=Gnode
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=7:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$EXP_ROOT/submit_coffee26n.slurm"
