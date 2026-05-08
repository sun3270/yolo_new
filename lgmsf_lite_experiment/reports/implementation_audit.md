# LGMSF-Lite Implementation Audit

Audit date: 2026-04-30

## Result

The PC-training LGMSF-Lite route described in `LGMSF_Lite_PC_Training_Codex_Guide.md` is implemented.

## Checklist

- [x] All experiment files are under `lgmsf_lite_experiment/`.
- [x] Project root `ultralytics/` has no new diff from this LGMSF-Lite work.
- [x] Original YOLO26 YAML is copied to `configs/yolo26n_original_copy.yaml` before generating the experiment YAML.
- [x] Local Ultralytics copy exists at `local_ultralytics/ultralytics/`.
- [x] YOLO26n structure was inspected before YAML editing.
- [x] P3/P4/P5 feature indices are confirmed as `4 / 6 / 10`.
- [x] P3/P4/P5 downsample Conv indices are confirmed as `3 / 5 / 7`.
- [x] Those three downsample layers are replaced by `LDSConv` in `configs/yolo26n_lgmsf_lite.yaml`.
- [x] P1/P2 stem downsample layers remain native `Conv`.
- [x] `LGMSFBridge` is inserted at index `11`.
- [x] `LGMSFBridge` input is `[4, 10]`, meaning P3 texture plus P5 semantic.
- [x] Head positive indices were remapped; final Detect input is `[17, 20, 23]`.
- [x] Local `tasks.py` imports `LDSConv` and `LGMSFBridge`.
- [x] Local `parse_model()` handles `LDSConv` through `base_modules`.
- [x] Local `parse_model()` has a dedicated branch for two-input `LGMSFBridge`.
- [x] `LDSConv` is depthwise separable and keeps stride as a parameter.
- [x] `TextureBranch` uses two depthwise-separable 3x3 blocks with residual behavior.
- [x] `SemanticBranch` uses Ghost-style 1x1, depthwise 5x5, Ghost-style 1x1 with residual behavior.
- [x] `FastNormFuse2` uses two learnable ReLU-normalized weights.
- [x] `SimAM` uses the correct `x * sigmoid(y)` formula, not `sigmoid(1 - energy)`.
- [x] No CoordAtt module was added to LGMSF-Lite.
- [x] No deployment, export, quantization, or ablation script was added outside the copied upstream package.
- [x] Build check passed.
- [x] FLOPs are non-zero: `7.314329600000001`.
- [x] Dummy forward passed.
- [x] Training entry exists: `06_train_lgmsf_pc.py`.

## Key Numbers

- Parameters: `2,468,190`
- P3/P4/P5 feature indices: `4 / 6 / 10`
- Replaced downsample indices: `3 / 5 / 7`
- LGMSFBridge index: `11`
- Detect input after remap: `[17, 20, 23]`

## Notes

The copied clean Ultralytics package naturally contains upstream exporter and deployment-related source files. The LGMSF-Lite experiment did not add any new deployment or ablation scripts.
