# P5Slim-512 Distillation Experiment

Version 3.0: second-stage lightweight release for mobile edge deployment.

This folder trains a smaller YOLO26n-EdgeLite student for mobile deployment.

- Teacher: `android_phone_deploy/model/best_edgelite_wiou_progloss.pt`
- Student YAML: `configs/yolo26n_edgelite_p5slim512.yaml`
- Main change: final P5 head `C3k2 [1024]` -> `C3k2 [512]`
- Loss: WIoU + ProgLoss + teacher distillation
- Positioning: this is the second lightweight stage after EdgeLite. EdgeLite is used as the teacher and P5Slim-512 is the smaller student.

## Expected Size

Approximate model-build numbers:

| Model | Params |
|---|---:|
| Current EdgeLite train graph | 1.95M |
| P5Slim-512 train graph | 1.57M |
| P5Slim-512 deploy effective graph | 1.48M |

## Attention Mechanisms

The active P5Slim-512 student keeps the YOLO26/EdgeLite attention design conservative:

- `C2PSA` is enabled at backbone layer 10 on the P5 semantic source. It splits channels, applies stacked `PSABlock` modules to one branch, then concatenates and projects the result. Each `PSABlock` contains multi-head self-attention plus a feed-forward block, so it strengthens global semantic modeling at the lowest-resolution feature map where the compute cost is acceptable.
- `EdgeLGMSFBridge` is enabled at layer 11 as a lightweight P3/P5 fusion bridge. It is not a pure attention block, but it includes learnable normalized two-input fusion weights, so it can adaptively balance P3 texture detail and P5 semantic context before the final P5 detection path.
- `SimAM` is implemented in the local EdgeLite module, but it is disabled in the active P5Slim-512 YAML by `EdgeLGMSFBridge [256, False, 8]`. The SimAM-enabled variant exists in the earlier EdgeLite config, but this 3.0 student keeps it off to avoid extra runtime risk on mobile.

Not active in this student YAML: `CBAM`, `ChannelAttention`, `SpatialAttention`, `C2fAttn`, `ImagePoolingAttn`, and `MaxSigmoidAttnBlock`.

## Run

Full run:

```powershell
python p5slim512_distill_experiment/train_coffee_p5slim512_distill.py
```

Smoke run:

```powershell
$env:P5SLIM_DISTILL_EPOCHS = "3"
python p5slim512_distill_experiment/train_coffee_p5slim512_distill.py
```

Optional overrides:

```powershell
$env:P5SLIM_DISTILL_BATCH = "16"
$env:P5SLIM_DISTILL_WORKERS = "2"
$env:P5SLIM_DISTILL_NAME = "v3_p5slim512_second_lightweight_coffee3000"
```

## Validation Target

Use the current EdgeLite + WIoU/ProgLoss best checkpoint as the baseline:

- mAP50-95 target: at least `0.92179`
- mAP50 target: at least `0.97675`
- Check minority class recall, especially `powdery_mildew`
- Re-run ONNX/NCNN consistency validation before Android replacement
