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

Run the core ablation matrix sequentially:

```bash
python coffee_v4_training/run_ablation_matrix.py --group core
```

Run only selected experiments:

```bash
python coffee_v4_training/run_ablation_matrix.py --exps edgelite_bibridge_wiou,edgelite_bibridge_elteb_wiou
```

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

## Recommended ablation order

1. `native_wiou`
2. `edgelite_wiou`
3. `edgelite_bibridge_wiou`
4. `edgelite_bibridge_clahe_wiou`
5. `edgelite_bibridge_laplacian_wiou`
6. `edgelite_bibridge_elteb_wiou`
7. `edgelite_bibridge_elteb_lite_wiou`

The preprocessing variants build generated datasets under
`elteb_experiment/preprocessed_data` by default. Use `--overwrite-preprocess`
when you need to rebuild them.

