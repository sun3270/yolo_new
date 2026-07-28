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

ROOT=/public/home/2024505440209/ultralytics-8.4.43
PYTHON=/public/home/2024505440209/.conda/envs/yolo26/bin/python
EXP_ROOT="$ROOT/coffee26n_experiment"
DATA_YAML="${DATA_YAML:-$ROOT/coffee3000/coffee3000.yaml}"
WEIGHTS="${WEIGHTS:-$EXP_ROOT/yolo26n.pt}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT="$ROOT/coffee26n_runs/full_coffee_l4_seed0_${RUN_ID}"

export OMP_NUM_THREADS=1
export TMPDIR="/public/home/2024505440209/tmp/slurm-${RUN_ID}"
mkdir -p "$TMPDIR"
cd "$ROOT"

echo "===== Environment ====="
hostname
date
nvidia-smi

"$PYTHON" -c "
import torch
print('torch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
assert torch.__version__.startswith('2.4.1'), torch.__version__
assert torch.cuda.is_available(), 'CUDA is unavailable'
"

"$PYTHON" -c "
from coffee26n_experiment.model_utils import setup_local_imports
paths = setup_local_imports()
print(paths)
assert all('$EXP_ROOT/local_ultralytics' in path for path in paths.values()), paths
"

test -f "$DATA_YAML" || { echo "Dataset YAML not found: $DATA_YAML" >&2; exit 2; }
test -f "$WEIGHTS" || { echo "Weights not found: $WEIGHTS" >&2; exit 2; }

echo "===== Dataset preflight ====="
echo "DATA_YAML=$DATA_YAML"
echo "Coffee3000 is accepted only after this job parses the actual YAML, labels, split integrity and fresh data lock."

echo "===== Structure preflight ====="
mkdir -p "$ROOT/coffee26n_preflight"
"$PYTHON" "$EXP_ROOT/verify_structure.py" \
  --data "$DATA_YAML" \
  --variant full \
  --profile auto_full \
  --imgsz 960 \
  --weights "$WEIGHTS" \
  --output "$ROOT/coffee26n_preflight/full_${RUN_ID}.json" >/dev/null

echo "===== Training ====="
"$PYTHON" "$EXP_ROOT/train_coffee26n.py" \
  --data "$DATA_YAML" \
  --variant full \
  --profile auto_full \
  --loss coffee_l4 \
  --bgmix auto \
  --weights "$WEIGHTS" \
  --imgsz 960 \
  --epochs 300 \
  --batch 96 \
  --workers 8 \
  --device 0 \
  --seed 0 \
  --optimizer MuSGD \
  --lr0 0.01 \
  --lrf 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --warmup-epochs 4 \
  --warmup-momentum 0.8 \
  --warmup-bias-lr 0.1 \
  --patience 80 \
  --nbs 64 \
  --mosaic 0.7 \
  --close-mosaic 150 \
  --bgmix-close-epoch 150 \
  --hsv-h 0.005 \
  --hsv-s 0.20 \
  --hsv-v 0.10 \
  --scale 0.35 \
  --erasing 0.0 \
  --no-cache \
  --amp \
  --deterministic \
  --no-cos-lr \
  --output "$OUTPUT"

echo "===== Training completed ====="
date
