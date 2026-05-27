# Phone Benchmark Statistical Analysis

Data source: Android benchmark logs pulled from two phones. Each phone has 8 batches, 4 models x 2 batches x 100 images = 800 image-level runs.

## Devices
- 23127PN0CC: Xiaomi 23127PN0CC, n=800, total_ms mean=70.27, p95=76.09, max=129.62, <200ms=100.0%
- M2012K10C: Xiaomi M2012K10C, n=800, total_ms mean=149.32, p95=162.19, max=341.41, <200ms=98.5%

## Per-phone per-model speed
| Phone | Model | n | total mean ms | total p95 | total max | inference mean ms | inference p95 | <200ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 23127PN0CC | Base | 200 | 75.40 | 78.93 | 129.62 | 66.30 | 70.04 | 100.0% |
| 23127PN0CC | EdgeLite | 200 | 68.54 | 71.94 | 88.02 | 59.54 | 63.26 | 100.0% |
| 23127PN0CC | EdgeLite+SimAM | 200 | 68.58 | 71.25 | 90.24 | 59.73 | 62.31 | 100.0% |
| 23127PN0CC | EdgeLite+WIoU/ProgLoss | 200 | 68.54 | 71.07 | 98.88 | 59.79 | 62.31 | 100.0% |
| M2012K10C | Base | 200 | 144.26 | 162.37 | 207.73 | 89.98 | 104.79 | 99.5% |
| M2012K10C | EdgeLite | 200 | 146.39 | 160.30 | 341.41 | 89.72 | 96.75 | 99.5% |
| M2012K10C | EdgeLite+SimAM | 200 | 151.20 | 161.06 | 306.94 | 92.34 | 100.84 | 98.0% |
| M2012K10C | EdgeLite+WIoU/ProgLoss | 200 | 155.44 | 163.71 | 319.42 | 94.03 | 98.68 | 97.0% |

## Phone-to-phone differences by model
Positive percent means 23127PN0CC is slower; negative percent means 23127PN0CC is faster.
| Model | total mean M2012 | total mean 23127 | total diff % | total p(FDR) | inference mean M2012 | inference mean 23127 | inference diff % | inference p(FDR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 144.26 | 75.40 | -47.73% | <0.001 | 89.98 | 66.30 | -26.32% | <0.001 |
| EdgeLite | 146.39 | 68.54 | -53.18% | <0.001 | 89.72 | 59.54 | -33.64% | <0.001 |
| EdgeLite+SimAM | 151.20 | 68.58 | -54.64% | <0.001 | 92.34 | 59.73 | -35.31% | <0.001 |
| EdgeLite+WIoU/ProgLoss | 155.44 | 68.54 | -55.90% | <0.001 | 94.03 | 59.79 | -36.42% | <0.001 |

## Model speed improvement versus Base
| Phone | Metric | Model | Base mean ms | Model mean ms | Improvement ms | Improvement % | p(FDR) | Significant faster? |
|---|---|---|---:|---:|---:|---:|---:|---|
| M2012K10C | total_ms | EdgeLite | 144.26 | 146.39 | -2.13 | -1.48% | 0.146 | no |
| M2012K10C | total_ms | EdgeLite+SimAM | 144.26 | 151.20 | -6.94 | -4.81% | <0.001 | no |
| M2012K10C | total_ms | EdgeLite+WIoU/ProgLoss | 144.26 | 155.44 | -11.18 | -7.75% | <0.001 | no |
| M2012K10C | inference_ms | EdgeLite | 89.98 | 89.72 | 0.25 | 0.28% | 0.791 | no |
| M2012K10C | inference_ms | EdgeLite+SimAM | 89.98 | 92.34 | -2.36 | -2.62% | 0.014 | no |
| M2012K10C | inference_ms | EdgeLite+WIoU/ProgLoss | 89.98 | 94.03 | -4.05 | -4.50% | <0.001 | no |
| 23127PN0CC | total_ms | EdgeLite | 75.40 | 68.54 | 6.86 | 9.10% | <0.001 | yes |
| 23127PN0CC | total_ms | EdgeLite+SimAM | 75.40 | 68.58 | 6.82 | 9.05% | <0.001 | yes |
| 23127PN0CC | total_ms | EdgeLite+WIoU/ProgLoss | 75.40 | 68.54 | 6.86 | 9.10% | <0.001 | yes |
| 23127PN0CC | inference_ms | EdgeLite | 66.30 | 59.54 | 6.76 | 10.20% | <0.001 | yes |
| 23127PN0CC | inference_ms | EdgeLite+SimAM | 66.30 | 59.73 | 6.56 | 9.90% | <0.001 | yes |
| 23127PN0CC | inference_ms | EdgeLite+WIoU/ProgLoss | 66.30 | 59.79 | 6.51 | 9.82% | <0.001 | yes |

## Under 200 ms requirement
23127PN0CC has 100% of image-level total_ms values below 200 ms. M2012K10C averages below 200 ms and has p95 below 200 ms, but still has a few outliers above 200 ms, so it does not satisfy a strict every-image <200 ms guarantee. Current configuration: ONNX Runtime Android CPU, input 640x640, confidence 0.25, NMS IoU 0.70, one image at a time. For a robust <200 ms target, keep p95 well below 200 ms and p99 below 200 ms; the newer phone already satisfies this with a wide margin.
