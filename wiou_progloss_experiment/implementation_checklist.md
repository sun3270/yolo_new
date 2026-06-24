# Implementation Checklist

## Preparation

- [x] Keep this management folder at `wiou_progloss_experiment/`.
- [x] Keep `edgelite_experiment/configs/yolo26n_edgelite.yaml` unchanged.
- [x] Keep `edgelite_experiment/train_coffee_edgelite.py` as the baseline script.
- [x] Keep native `edgelite_experiment/local_ultralytics/ultralytics/utils/loss.py` available for baseline tests.
- [x] Implement experimental loss code inside `wiou_progloss_experiment/`.
- [x] Add the new train script at `wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py`.
- [x] Support `native`, `edgelite`, and `edgelite_bibridge` architecture selection from the loss train script.

## Code Changes

- [x] Keep local `ultralytics/cfg/default.yaml` free of experiment-only loss keys.
- [x] Keep local `ultralytics/cfg/__init__.py` free of experiment-only type checks.
- [x] Add reusable WIoU and ProgLoss helpers in `wiou_progloss_experiment/loss_extensions.py`.
- [x] Add custom criterion classes in `wiou_progloss_experiment/wiou_progloss_loss.py`.
- [x] Register WIoU running mean as a custom `BboxWIoUProgLoss` buffer.
- [x] Use raw IoU for WIoU beta and focus calculation.
- [x] Keep the native L1-style branch for `reg_max: 1` inside the custom criterion.
- [x] Add progressive positive-only class weights around BCE classification loss.
- [x] Patch `DetectionModel.init_criterion` only in the experiment training process.
- [x] Propagate epoch progress through custom `E2EWIoUProgLoss.update()`.
- [x] Keep WIoU and ProgLoss outside the native loss file unless explicitly enabled by the experiment script.
- [x] Keep architecture selection outside the loss module so the same loss can attach to native YOLO and EdgeLite variants.

## First Smoke Test

- [x] Run `edgelite_experiment/check_edgelite_build.py`.
- [x] Compile the new experiment modules and restored native loss file.
- [x] Run custom WIoU bbox-loss smoke test.
- [x] Run custom ProgLoss classification smoke test.
- [x] Confirm `loss_config.yaml` loads and patches the experiment criterion in process.
- [x] Confirm the loss patch can initialize on native YOLO, EdgeLite, and EdgeLite-BiBridge.
- [ ] Train for 3 epochs with WIoU disabled and ProgLoss disabled to confirm baseline still works.
- [ ] Train for 3 epochs with WIoU only.
- [ ] Train for 3 epochs with ProgLoss only.
- [ ] Train for 3 epochs with WIoU + ProgLoss.

## Full Comparison

- [ ] Baseline EdgeLite full run.
- [ ] WIoU-only full run.
- [ ] ProgLoss-only full run.
- [ ] WIoU + ProgLoss full run.
- [ ] Compare mAP50-95, mAP50, per-class AP, rare-class recall, and false positives.

## Stop Conditions

- [ ] Stop and reduce `tail_lambda_max` if rare-class precision collapses.
- [ ] Disable `use_distance_gain` if box loss spikes or becomes NaN.
- [ ] Reduce `focus_max` from 3.0 to 2.0 if one-to-one loss becomes unstable.
- [ ] Revert to CIoU base plus WIoU focus only if raw WIoU degrades all-class mAP.
