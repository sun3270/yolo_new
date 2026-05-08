# Implementation Checklist

## Preparation

- [x] Keep this management folder at `wiou_progloss_experiment/`.
- [x] Keep `edgelite_experiment/configs/yolo26n_edgelite.yaml` unchanged.
- [x] Keep `edgelite_experiment/train_coffee_edgelite.py` as the baseline script.
- [x] Implement only inside `edgelite_experiment/local_ultralytics`.
- [x] Add the new train script at `wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py`.

## Code Changes

- [x] Add flat loss config keys to `edgelite_experiment/local_ultralytics/ultralytics/cfg/default.yaml`.
- [x] Add the new switch and scalar keys to `edgelite_experiment/local_ultralytics/ultralytics/cfg/__init__.py` type sets where needed.
- [x] Add a WIoU helper in `edgelite_experiment/local_ultralytics/ultralytics/utils/loss.py`.
- [x] Update `BboxLoss` to support baseline CIoU and WIoU modes.
- [x] Register WIoU running mean as a `BboxLoss` buffer.
- [x] Use raw IoU for WIoU beta and focus calculation.
- [x] Keep the existing L1-style branch for `reg_max: 1`.
- [x] Add progressive positive-only class weights around BCE classification loss.
- [x] Pass `model.args` into `BboxLoss` from `v8DetectionLoss`.
- [x] Propagate epoch progress through `E2ELoss.update()`.
- [x] Make WIoU and ProgLoss disabled by default unless explicitly enabled.

## First Smoke Test

- [x] Run `edgelite_experiment/check_edgelite_build.py`.
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
