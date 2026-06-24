# YOLO26n-EdgeLite Experiment

This is an isolated experiment workspace for the edge-oriented YOLO26n route.
It does not modify the project root `ultralytics/` package or the existing
`lgmsf_lite_experiment/` workspace.

## Routes

- `YOLO26n-EdgeLite`: LDSConv downsampling + P3 texture stream + P5 semantic stream + FastNormFuse2.
- `YOLO26n-EdgeLite-SimAM`: same structure, with SimAM enabled after the P3/P5 weighted fusion.
- `YOLO26n-EdgeLite-BiBridge`: keeps the original P3 -> P5 bridge and adds P5 -> P3 semantic feedback.
- `YOLO26n-EdgeLite-SimAM-BiBridge`: BiBridge route with SimAM enabled only on the original P3 -> P5 bridge.

## Key Structure

- P3 feature index: `4`
- P5 feature index: `10`
- Backbone downsample Conv replacements: `3 / 5 / 7`
- Head downsample Conv replacements: original `17 / 20`
- Fusion bridge: `EdgeLGMSFBridge([P3=4, P5=10])`
- BiBridge P3 feedback: `P5ToP3SemanticFuse([P3=4, P5=10])`
- BiBridge detection head: `Detect(P3_enhanced, P4, P5_enhanced)`

Training scripts:

```bash
python edgelite_experiment/train_coffee_edgelite.py
python edgelite_experiment/train_coffee_edgelite_simam.py
python edgelite_experiment/train_coffee_edgelite_bibridge.py
python edgelite_experiment/train_coffee_edgelite_simam_bibridge.py
```

All EdgeLite training scripts default to `coffee_self_sum/coffee_self_sum.yaml`,
`epochs=300`, `imgsz=960`, `batch=64`, and `workers=8`. The SimAM-BiBridge
entry loads `yolo26n_edgelite_simam_bibridge.yaml` directly.

BiBridge is currently enabled only in the EdgeLite route. ELTEB and ELTEBLite
remain separate ablation branches.
