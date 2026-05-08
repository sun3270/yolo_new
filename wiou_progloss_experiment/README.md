# WIoU + ProgLoss Loss Experiment

This project-level folder manages the loss-function improvement plan for the
validated YOLO26n-EdgeLite route.

## Folder Location

- Experiment folder: `wiou_progloss_experiment/`
- Baseline model YAML: `edgelite_experiment/configs/yolo26n_edgelite.yaml`
- Baseline training script: `edgelite_experiment/train_coffee_edgelite.py`
- Target local package: `edgelite_experiment/local_ultralytics`

The folder is intentionally placed at the repository root for easier management,
but the planned loss changes still target the modified EdgeLite version under
`edgelite_experiment/`. The root package `ultralytics/` should stay unchanged.

## Applicable Version

This experiment applies to the tested no-SimAM EdgeLite variant:

- Architecture route: LDSConv downsampling + P3 texture stream + P5 semantic stream + FastNormFuse2
- Fusion variant: SimAM disabled
- Current config: `end2end: True`
- Current regression config: `reg_max: 1`

## Goal

Introduce a dynamic non-monotonic WIoU box loss plus a progressive loss schedule
to reduce training imbalance from the coffee leaf long-tail distribution while
preserving the lighter EdgeLite architecture.

## Files

- `design.md`: concrete design, integration path, and feasibility notes
- `loss_config.yaml`: flat proposed switches and default parameters
- `implementation_checklist.md`: safe implementation and validation order

## Management Rule

Do not modify the verified EdgeLite baseline training script directly. Implement
and test this loss route as a separate experiment from this root-level folder,
then compare it against the baseline with the same data split, image size, batch
size, epochs, and pretrained weights.
