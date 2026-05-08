# LGMSF-Lite PC Training Workspace

This directory is an isolated experiment workspace for YOLO26n-LGMSF-Lite.

Core route:

```text
LGMSF-Lite = LDSConv downsample + P3 texture stream + P5 semantic stream + FastNormFuse2 + SimAM
```

Rules:

- Do not modify the project root `ultralytics/` package.
- Do not modify the original YOLO26 YAML.
- Modify only the copied package under `local_ultralytics/`.
- Keep reports under `reports/` and patch notes under `patches/`.
- This stage is PC training only: no deployment, export, quantization, or ablation scripts.

Default dataset:

```text
coffee3000/coffee3000.yaml
```
