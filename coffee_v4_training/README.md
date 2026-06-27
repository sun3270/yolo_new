# Coffee V4 Training

This folder is the unified entrypoint for the active `coffee_self_sum` V4 work.
The abandoned `lgmsf_lite_experiment` path is intentionally not used here.

## Defaults

All experiments default to:

```text
data=coffee_self_sum/coffee_self_sum.yaml
epochs=300
imgsz=960
batch=64
cache=ram
workers=8
amp=True
patience=30
save_period=20
close_mosaic=10
lr0=0.012
lrf=0.01
momentum=0.937
weight_decay=0.0005
warmup_epochs=4.0
box=7.5
cls=0.5
dfl=1.5
```

## Commands

List experiments:

```bash
python coffee_v4_training/train.py --list
```

Check model YAML construction and dummy forward:

```bash
python coffee_v4_training/check_builds.py
```

Run the current V4 main baseline:

```bash
python coffee_v4_training/train.py --exp edgelite_bibridge_wiou
```

Resume behavior:

```bash
# Default: if runs/train/<run_name>/weights/last.pt exists, training resumes it.
python coffee_v4_training/train.py --exp edgelite_bibridge_wiou_sanitize_copypaste_context --epochs 300

# Force resume from the default same-name run, and fail if last.pt is missing.
python coffee_v4_training/train.py --exp edgelite_bibridge_wiou_sanitize_copypaste_context --resume --epochs 300

# Resume a specific checkpoint.
python coffee_v4_training/train.py --exp edgelite_bibridge_wiou_sanitize_copypaste_context --resume-from runs/train/my_run/weights/last.pt --epochs 300

# Start a new run even if last.pt exists.
python coffee_v4_training/train.py --exp edgelite_bibridge_wiou_sanitize_copypaste_context --fresh --epochs 300
```

`epochs` is the total target epoch count when resuming. For example, resuming a
96-epoch run with `--epochs 300` continues from epoch 97 until 300. To disable
default same-run auto-resume globally, set `COFFEE_AUTO_RESUME=0` or pass
`--no-auto-resume`.

Run the core ablation matrix sequentially:

```bash
python coffee_v4_training/run_ablation_matrix.py --group core
```

Run only selected experiments:

```bash
python coffee_v4_training/run_ablation_matrix.py --exps edgelite_bibridge_wiou,edgelite_bibridge_elteb_wiou
```

Generate recall-focused diagnostics:

```bash
python coffee_v4_training/dataset_diagnostics.py \
  --data /path/to/coffee_self_sum/coffee_self_sum.yaml \
  --output runs/coffee_v4_diagnostics \
  --splits val,test \
  --tail-classes 4,6,7,8
```

The diagnostics command writes `dataset_report.json`, `hard_cases.csv`, and
`paper_requirements_report.md`. The Markdown report maps the scan back to the
three yolo.md miss-detection causes: bad borders/small edges, cluttered
backgrounds, and multi-instance scenes.

Build recall-oriented derived datasets without modifying the source dataset:

```bash
python coffee_v4_training/build_recall_datasets.py --variant sanitize_labels --data /path/to/coffee_self_sum/coffee_self_sum.yaml
python coffee_v4_training/build_recall_datasets.py --variant sahi --data /path/to/coffee_self_sum/coffee_self_sum.yaml
python coffee_v4_training/build_recall_datasets.py --variant copypaste --data /path/to/coffee_self_sum/coffee_self_sum.yaml
python coffee_v4_training/build_recall_datasets.py --variant roi --data /path/to/coffee_self_sum/coffee_self_sum.yaml
```

Each derived dataset writes `recall_dataset_manifest.json` next to the dataset
YAML so the original data path, variant parameters, and generated image counts
are auditable.

Recommended local data-cleaning sequence:

```powershell
$DATA="C:\Users\ssema\Desktop\coffee\datasets\coffee_self_sum\coffee_self_sum.yaml"

python coffee_v4_training\build_recall_datasets.py `
  --variant sanitize_labels `
  --data $DATA `
  --output elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels `
  --overwrite

python coffee_v4_training\dataset_diagnostics.py `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml `
  --output runs\dataset_audit_sanitize `
  --splits train,val,test `
  --tail-classes 1,2,3,5,4,6,7,8

python coffee_v4_training\build_recall_datasets.py `
  --variant copypaste_context `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml `
  --output elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_copypaste_context `
  --copy-class-weights 1:2,2:3,3:2,5:3 `
  --max-paste-iou 0.10 `
  --overwrite
```

SAM2-assisted label review is an offline dataset tool, not a deployment-time
dependency. Without SAM2 installed it falls back to a lightweight leaf-mask
candidate so the review CSV and previews can still be generated locally:

```powershell
python coffee_v4_training\sam2_refine_labels.py `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml `
  --output elteb_experiment\preprocessed_data\coffee_self_sum_sam_refined `
  --mode review `
  --iou-gate 0.55 `
  --overwrite
```

If SAM2 is installed, add `--sam2-config`, `--sam2-checkpoint`, and optionally
`--require-sam2`. Review `review_candidates.csv` and preview images before
using the derived YAML for training.

Sweep inference settings for hard-case recall triage:

```bash
python coffee_v4_training/inference_recall_sweep.py \
  --data /path/to/coffee_self_sum/coffee_self_sum.yaml \
  --split val \
  --output runs/coffee_v4_recall_sweep
```

When `--model` is provided, the sweep matches predictions to YOLO labels and
writes per-config recall, missed counts, and low-confidence candidates to
`recall_sweep_summary.json`. `soft` NMS is explicitly marked unsupported until
a raw-prediction postprocessor is added, so it cannot be mistaken for real
Soft-NMS.

## High-compute overrides

Use environment variables or CLI flags when paths differ on the server:

```bash
COFFEE_DATA=/path/to/coffee_self_sum/coffee_self_sum.yaml \
COFFEE_WEIGHTS=/path/to/yolo26n.pt \
COFFEE_PROJECT=/path/to/runs/train \
COFFEE_DEVICE=0 \
python coffee_v4_training/train.py --exp edgelite_bibridge_wiou
```

For multi-GPU Ultralytics launch:

```bash
COFFEE_DEVICE=0,1 python coffee_v4_training/train.py --exp edgelite_bibridge_wiou
```

When a derived YAML was generated on another Windows machine, remap absolute
paths at runtime instead of editing the source YAML:

```powershell
$env:COFFEE_PATH_REMAP="C:/Users/1/Desktop/coffee=>//BRUCE/Users/1/Desktop/coffee"

python coffee_v4_training\train.py `
  --exp edgelite_bibridge_wiou_sanitize_copypaste_context `
  --data \\BRUCE\Users\1\Desktop\coffee\yolo_new\elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_copypaste_context\coffee_self_sum_sanitize_copypaste_context.yaml `
  --dry-run
```

The training entrypoint writes a temporary remapped YAML under
`runs/train/_remapped_data/<run_name>/` and leaves the source dataset read-only.

## Recommended ablation order

1. `native_wiou`
2. `edgelite_wiou`
3. `edgelite_bibridge_wiou`
4. `edgelite_bibridge_clahe_wiou`
5. `edgelite_bibridge_laplacian_wiou`
6. `edgelite_bibridge_elteb_wiou`
7. `edgelite_bibridge_elteb_lite_wiou`

Recall-focused follow-up experiments:

1. `edgelite_bibridge_wiou_sahi`
2. `edgelite_bibridge_wiou_copypaste`
3. `edgelite_bibridge_wiou_roi`
4. `edgelite_bibridge_wiou_sanitize_labels`
5. `edgelite_bibridge_wiou_sanitize_copypaste_context`
6. `edgelite_bibridge_wiou_sam_refined`
7. `edgelite_bibridge_wiou_copypaste_context`
8. `edgelite_bibridge_wiou_roi_leafmask`
9. `edgelite_bibridge_nwd`

`edgelite_bibridge_wiou_sahi_trainonly` remains registered for debugging but
is intentionally excluded from the default recall group until sliced-label
quality is fixed.

The preprocessing variants build generated datasets under
`elteb_experiment/preprocessed_data` by default. Use `--overwrite-preprocess`
when you need to rebuild them.

The recall group can be checked with:

```bash
python coffee_v4_training/run_ablation_matrix.py --group recall --dry-run
```

Strong recall/model experiments should be screened in this order:

1. `edgelite_bibridge_wiou_y26_recipe`
2. `edgelite_bibridge_wiou_stal_lite`
3. `edgelite_bibridge_wiou_p2`
4. `edgelite_hyperace_wiou`
5. `edgelite_hyperace_p2_wiou`

Recommended 80-epoch screen on the classmate machine:

```powershell
cd C:\Users\1\Desktop\coffee\yolo_new
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$DATA="C:\Users\1\Desktop\coffee\yolo_new\elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_copypaste_context\coffee_self_sum_sanitize_copypaste_context.yaml"
$WEIGHTS="C:\Users\1\Desktop\coffee\yolo_new\yolo26n.pt"

python coffee_v4_training\run_ablation_matrix.py `
  --group strong `
  --data $DATA `
  --weights $WEIGHTS `
  --device 0 `
  --epochs 80 `
  --imgsz 960 `
  --batch 16 `
  --workers 4 `
  --cache disk `
  --continue-on-error
```

After a screen, summarize comparable metrics:

```powershell
python coffee_v4_training\summarize_training_runs.py `
  --runs-dir runs\train `
  --output runs\strong_summary
```

Promote only experiments that beat `v4_edgelite_bibridge_wiou_copypaste_context_coffee_self_sum4`
on either `mAP50-95 > 0.7708` or `Recall > 0.812` with `Precision >= 0.80`.

