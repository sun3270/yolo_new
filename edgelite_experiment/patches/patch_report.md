# YOLO26n-EdgeLite Patch Report

- Local package patched: `edgelite_experiment/local_ultralytics/ultralytics`
- Original project package: not modified.
- Existing `lgmsf_lite_experiment/`: not modified.

## Added Modules

- `LDSConv`
- `TextureStreamP3`
- `SemanticStreamP5`
- `FastNormFuse2`
- `SimAM`
- `EdgeLGMSFBridge`
- `P5ToP3SemanticFuse`

## parse_model Changes

- `LDSConv` is handled as a standard base module.
- `EdgeLGMSFBridge` has a dedicated two-input branch for `[P3, P5]`.
- `P5ToP3SemanticFuse` has a dedicated two-input branch for `[P3, P5]`.
