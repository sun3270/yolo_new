# YOLO26n-EdgeLite Experiment

This is an isolated experiment workspace for the edge-oriented YOLO26n route.
It does not modify the project root `ultralytics/` package or the existing
`lgmsf_lite_experiment/` workspace.

## Routes

- `YOLO26n-EdgeLite`: LDSConv downsampling + P3 texture stream + P5 semantic stream + FastNormFuse2.
- `YOLO26n-EdgeLite-SimAM`: same structure, with SimAM enabled after the P3/P5 weighted fusion.

## Key Structure

- P3 feature index: `4`
- P5 feature index: `10`
- Backbone downsample Conv replacements: `3 / 5 / 7`
- Head downsample Conv replacements: original `17 / 20`
- Fusion bridge: `EdgeLGMSFBridge([P3=4, P5=10])`

Training scripts:

```bash
python edgelite_experiment/train_coffee_edgelite.py
python edgelite_experiment/train_coffee_edgelite_simam.py
```
