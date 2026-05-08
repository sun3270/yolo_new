# WIoU + ProgLoss Design

## Project Placement

This design is managed from the repository root:

`wiou_progloss_experiment/`

The implementation target is still the tested no-SimAM EdgeLite route:

- Model YAML: `edgelite_experiment/configs/yolo26n_edgelite.yaml`
- Baseline training script: `edgelite_experiment/train_coffee_edgelite.py`
- Editable local package: `edgelite_experiment/local_ultralytics`
- Package that should remain unchanged: `ultralytics/`

The new training entry should be created at:

`wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py`

That script should prepend `edgelite_experiment/local_ultralytics` to `sys.path`,
the same way the baseline EdgeLite script does.

## Current Baseline

The current verified route is `YOLO26n-EdgeLite` without SimAM:

- Uses `edgelite_experiment/local_ultralytics`, so the root package is isolated.
- Uses `end2end: True`, so training combines one-to-many and one-to-one losses.
- Uses `reg_max: 1`, so DFL is disabled. The reported `dfl_loss` branch is an L1-style box distance branch.
- Current box loss is CIoU-based: `loss_iou = (1 - iou) * target_score_weight`.

Coffee3000 label counts show a real long-tail pattern:

| split | algal_spot | brown_eye_spot | healthy | miner | phoma | powdery_mildew |
|---|---:|---:|---:|---:|---:|---:|
| train | 501 | 606 | 332 | 618 | 709 | 163 |
| valid | 150 | 198 | 111 | 193 | 121 | 52 |
| test | 46 | 58 | 162 | 51 | 72 | 14 |

The rarest training class is `powdery_mildew`, about 4.35x smaller than `phoma`.
On test it is even more sparse. A box-only loss change will not fully solve this,
so WIoU should be paired with progressive class reweighting.

## Design Target

Use two complementary mechanisms:

1. WIoU dynamic non-monotonic focusing for box regression.
   This changes which positive boxes receive stronger regression gradients.
2. ProgLoss for progressive reweighting.
   This avoids applying aggressive class and box focusing too early, then
   gradually shifts training toward tail classes and high-value localization
   samples once assignments become more reliable.

## Component A: WIoU Box Loss

Replace the current CIoU box term with a WIoU-v3-style term in the EdgeLite local
loss file:

`edgelite_experiment/local_ultralytics/ultralytics/utils/loss.py`

Recommended first version:

```text
raw_iou = IoU(pred_box, target_box)
base_loss = 1 - raw_iou
beta = detach(base_loss) / running_mean(base_loss)
focus = beta / (delta * alpha ** (beta - delta))
distance_gain = exp(center_distance_sq / enclosing_diagonal_sq)
wiou_loss = base_loss * detach(focus) * detach(distance_gain)
```

Recommended defaults:

- `alpha: 1.7`
- `delta: 2.7`
- `momentum: 0.0001`
- `focus_clip: [0.5, 3.0]`
- `distance_gain_clip: 1.8`

Reasoning:

- Low-quality boxes are not allowed to dominate forever.
- Very easy boxes are not over-optimized.
- Mid-quality boxes, which are most useful for mAP improvement, receive stronger gradients.
- Clipping is important because this project uses end-to-end detection and two assignment branches.

Fallback if the first version is unstable:

- Keep existing CIoU as the base loss.
- Apply only the WIoU non-monotonic focus factor.
- Disable `distance_gain` first, then re-enable it after a stable smoke test.

Important implementation detail:

- The WIoU focus calculation should use raw IoU, not CIoU, because CIoU can be
  negative after distance and aspect penalties.
- If a CIoU fallback is kept, clamp the base term before computing beta:
  `base_loss = (1 - ciou).clamp(min=0, max=2)`.
- Register the running mean as a buffer inside `BboxLoss`, so it moves with the
  model device and is saved in checkpoints.

## Component B: ProgLoss Schedule

ProgLoss should control when WIoU and long-tail class weighting become active.

Use epoch progress:

```text
p = current_epoch / max(epochs - 1, 1)
smooth(p) = p * p * (3 - 2 * p)
```

Recommended schedule:

| phase | epoch range | behavior |
|---|---:|---|
| warmup | 0-10% | keep baseline loss, no class reweighting |
| transition | 10-60% | ramp WIoU focus and tail class weights |
| consolidation | 60-100% | full WIoU + capped tail reweighting |

Classification weighting:

```text
class_weight[c] = (max_class_count / class_count[c]) ** tail_power
class_weight = normalize_to_mean_1(class_weight)
active_weight = 1 + lambda_tail(p) * (class_weight - 1)
positive_scale = 1 + (active_weight - 1) * target_scores
cls_loss = BCE(pred_scores, target_scores) * positive_scale
```

Recommended defaults:

- `tail_power: 0.5`
- `tail_lambda_max: 0.8`
- `tail_weight_clip: [0.75, 1.8]`

This gives tail classes more positive gradient without letting rare classes
explode false positives.

Important implementation detail:

- Apply class weights only through positive targets, not to all negative columns.
  Multiplying the full class column would also increase negative loss for rare
  classes and can suppress rare-class recall.

## Integration Points

Implementation should be isolated to the EdgeLite local package while managed
from this root-level folder:

1. Add new config keys to:
   `edgelite_experiment/local_ultralytics/ultralytics/cfg/default.yaml`

2. Add the new keys to the local config type sets where useful:
   `edgelite_experiment/local_ultralytics/ultralytics/cfg/__init__.py`

   Use boolean keys for switches and float keys for scalar parameters. This is
   required if the values are ever passed through CLI-style overrides.

3. Update `BboxLoss` in:
   `edgelite_experiment/local_ultralytics/ultralytics/utils/loss.py`

4. Pass `model.args` into `BboxLoss` from `v8DetectionLoss`.

5. Add progressive class reweighting around the existing BCE classification loss.

6. Extend `v8DetectionLoss` with an epoch-progress state, then extend
   `E2ELoss.update()` so it also advances that state inside both:
   - `one2many`
   - `one2one`

7. Add a separate training script after implementation:
   `wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py`

## Proposed Flat Config Keys

Use flat keys so they can be added to `default.yaml` and read as `self.hyp.*`:

```yaml
loss_name: wiou_progloss
wiou_enabled: false
wiou_alpha: 1.7
wiou_delta: 2.7
wiou_momentum: 0.0001
wiou_focus_min: 0.5
wiou_focus_max: 3.0
wiou_use_distance_gain: true
wiou_distance_gain_max: 1.8
wiou_fallback_base: raw_iou
progloss_enabled: false
progloss_warmup_ratio: 0.10
progloss_ramp_end_ratio: 0.60
progloss_tail_power: 0.5
progloss_tail_lambda_max: 0.8
progloss_tail_weight_min: 0.75
progloss_tail_weight_max: 1.8
progloss_class_counts: [501, 606, 332, 618, 709, 163]
```

Keep both switches disabled by default in `default.yaml`. The new experiment
script should enable them explicitly, so the verified baseline remains intact.

## Why Not Change Architecture

The last EdgeLite result already improved compute cost by reducing parameters and
GFLOPs. This experiment should keep architecture frozen and only change loss
behavior. That keeps attribution clean:

- If metrics improve, the gain is from loss design.
- If metrics regress, the EdgeLite architecture is still preserved as a stable baseline.

## Ablation Plan

Run the following four experiments with identical training settings:

| run | WIoU | ProgLoss class weights | purpose |
|---|---|---|---|
| baseline | off | off | verified EdgeLite result |
| wiou_only | on | off | isolate localization effect |
| prog_only | off | on | isolate long-tail classification effect |
| wiou_progloss | on | on | final combined candidate |

Primary metrics:

- mAP50-95
- mAP50
- per-class AP for `powdery_mildew`, `healthy`, and `phoma`
- recall for rare classes
- false positives on `healthy`

Stability checks:

- No NaN or Inf loss in first 3 epochs.
- Box loss should not spike more than 2x baseline after warmup.
- Rare-class recall should improve without a large precision collapse.
- If `dfl_loss` remains near zero or represents L1 behavior, treat it as a
  secondary regression branch because `reg_max: 1` disables true DFL.

## Expected Result

The most realistic target is not a large global mAP jump. The expected useful
improvement is better rare-class AP and recall, especially for `powdery_mildew`,
while keeping baseline mAP50-95 flat or slightly higher.

## Feasibility Verdict

This plan is implementable in the current codebase because:

- `BboxLoss` is already the single box-loss entry for the detection branch.
- `v8DetectionLoss` owns both box and classification terms, so ProgLoss can be
  added without changing the model head.
- `E2ELoss.update()` is already called once per epoch by the trainer, so epoch
  progress can be propagated without modifying the training loop.
- The EdgeLite experiment already uses an isolated local package, so the root
  package and verified baseline can remain untouched.
