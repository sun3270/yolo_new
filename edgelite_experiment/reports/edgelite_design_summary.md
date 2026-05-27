# YOLO26n-EdgeLite Design Summary

## Implemented Routes

- `YOLO26n-EdgeLite`: P3 texture stream + P5 semantic stream + FastNormFuse2, SimAM disabled.
- `YOLO26n-EdgeLite-SimAM`: same model, SimAM enabled after weighted fusion.

## Structural Intent

- P3 texture stream keeps high-resolution shallow detail for early tiny rust spot edges.
- P5 semantic stream compresses and models global leaf structure and illumination context.
- FastNormFuse2 learns the relative contribution of P3 texture and P5 semantic signals.
- SimAM is only enabled in the SimAM variant to test whether lightweight attention suppresses complex background interference.

## Build Metrics

| model                  |    params |    GFLOPs | layers |
| ---------------------- | --------: | --------: | -----: |
| Native YOLO26n         | 2,506,140 |  5.782528 |    260 |
| YOLO26n-EdgeLite       | 1,951,870 | 4.6626304 |    324 |
| YOLO26n-EdgeLite-SimAM | 1,951,870 | 4.6626304 |    324 |
| YOLO26n-LGMSF-Lite     | 2,468,190 | 7.3143296 |    317 |

Compared with native YOLO26n, EdgeLite reduces parameters by about `22.12%` and GFLOPs by about `19.37%`.
Compared with the previous LGMSF-Lite, EdgeLite reduces parameters by about `20.92%` and GFLOPs by about `36.25%`.

## Notes

- The two EdgeLite YAML files differ only in the `use_simam` argument.
- Original `ultralytics/` and `lgmsf_lite_experiment/` were not modified.
- This stage does not add deployment, quantization, export, or mobile benchmark scripts.
