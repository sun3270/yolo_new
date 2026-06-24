# YOLO26n-EdgeLite Design Summary

## Implemented Routes

- `YOLO26n-EdgeLite`: P3 texture stream + P5 semantic stream + FastNormFuse2, SimAM disabled.
- `YOLO26n-EdgeLite-SimAM`: same model, SimAM enabled after weighted fusion.
- `YOLO26n-EdgeLite-BiBridge`: original P3 -> P5 bridge plus P5 -> P3 semantic feedback.
- `YOLO26n-EdgeLite-SimAM-BiBridge`: BiBridge route with SimAM enabled on the P3 -> P5 bridge.

## Structural Intent

- P3 texture stream keeps high-resolution shallow detail for early tiny rust spot edges.
- P5 semantic stream compresses and models global leaf structure and illumination context.
- FastNormFuse2 learns the relative contribution of P3 texture and P5 semantic signals.
- P5ToP3SemanticFuse returns P5 semantic context to P3 with a gated residual path, so early lesion texture is not overwritten at initialization.
- SimAM is only enabled in the SimAM variant to test whether lightweight attention suppresses complex background interference.

## Build Metrics

| model | params | GFLOPs | layers |
|---|---:|---:|---:|
| Native YOLO26n | 2,507,310 | 5.7889792 | 260 |
| YOLO26n-EdgeLite | 1,953,040 | 4.6690816 | 324 |
| YOLO26n-EdgeLite-SimAM | 1,953,040 | 4.6690816 | 324 |
| YOLO26n-EdgeLite-BiBridge | 1,961,555 | 4.7322112 | 335 |
| YOLO26n-EdgeLite-SimAM-BiBridge | 1,961,555 | 4.7322112 | 335 |
| YOLO26n-LGMSF-Lite | 2,468,190 | 7.3143296 | 317 |

Compared with native YOLO26n, EdgeLite reduces parameters by about `22.12%` and GFLOPs by about `19.37%`.
Compared with the previous LGMSF-Lite, EdgeLite reduces parameters by about `20.92%` and GFLOPs by about `36.25%`.
Compared with EdgeLite, BiBridge adds `8,515` parameters and about `0.063` GFLOPs at 640-pixel dummy build size.

## Notes

- The two EdgeLite YAML files differ only in the `use_simam` argument.
- BiBridge YAMLs keep `EdgeLGMSFBridge([P3=4, P5=10])` for P5 enhancement and add `P5ToP3SemanticFuse([P3=4, P5=10])` for P3 enhancement.
- BiBridge Detect input is `[18, 21, 24]`, i.e. `Detect(P3_enhanced, P4, P5_enhanced)`.
- BiBridge is currently only enabled in the EdgeLite route; ELTEB remains a separate ablation branch.
- Original `ultralytics/` and `lgmsf_lite_experiment/` were not modified.
- This stage does not add deployment, quantization, export, or mobile benchmark scripts.
