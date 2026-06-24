# WIoU + ProgLoss Loss Experiment

This project-level folder manages the WIoU + ProgLoss loss-function route for
YOLO26n architecture ablations.

## Folder Location

- Experiment folder: `wiou_progloss_experiment/`
- Default model YAML: `edgelite_experiment/configs/yolo26n_edgelite_bibridge.yaml`
- Native comparison YAML: `edgelite_experiment/configs/yolo26n_original_copy.yaml`
- EdgeLite comparison YAML: `edgelite_experiment/configs/yolo26n_edgelite.yaml`
- Target local package: `edgelite_experiment/local_ultralytics`

The folder is intentionally placed at the repository root for easier management,
but the planned loss changes still run through the isolated local package under
`edgelite_experiment/`. The root package `ultralytics/` should stay unchanged.
The native local loss file
`edgelite_experiment/local_ultralytics/ultralytics/utils/loss.py` should also
stay native, so later experiments can use the original loss directly.

## Applicable Version

This experiment can be applied to these architecture routes:

- `native`: original YOLO26n
- `edgelite`: original EdgeLite
- `edgelite_simam`: EdgeLite with SimAM on the P3 -> P5 bridge
- `edgelite_bibridge`: latest EdgeLite-BiBridge, default
- `edgelite_simam_bibridge`: latest EdgeLite-BiBridge with SimAM on the P3 -> P5 bridge

## Goal

Introduce a dynamic non-monotonic WIoU box loss plus a progressive loss schedule
to reduce training imbalance from the coffee leaf long-tail distribution while
allowing fair native YOLO, EdgeLite, and EdgeLite-BiBridge comparisons.

## Files

- `design.md`: concrete design, integration path, and feasibility notes
- `loss_config.yaml`: flat switches and default parameters loaded by the experiment script
- `implementation_checklist.md`: safe implementation and validation order
- `dataset_class_counts.py`: automatic train-label class counting for ProgLoss
- `loss_extensions.py`: reusable WIoU and ProgLoss math helpers
- `wiou_progloss_loss.py`: custom criterion classes and runtime patch hook
- `train_coffee_edgelite_wiou_progloss.py`: isolated training entry with architecture selection

## Training Data

The training entry reads the dataset YAML from `WIOU_PROGLOSS_DATA` when set,
counts train-label instances per class, and passes those counts to ProgLoss. It
defaults to `coffee_self_sum/coffee_self_sum.yaml`, `epochs=300`, `imgsz=960`,
`batch=64`, and `workers=8`.

```powershell
$env:WIOU_PROGLOSS_ARCH="native"             # native, edgelite, edgelite_bibridge, edgelite_simam, edgelite_simam_bibridge
$env:WIOU_PROGLOSS_DATA="E:\ultralytics-8.4.43\coffee_self_sum\coffee_self_sum.yaml"
python wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py
```

## Management Rule

Do not modify native loss files directly. Keep WIoU + ProgLoss as a runtime
patch from this root-level folder, then compare routes with the same data split,
image size, batch size, epochs, and pretrained weights.
