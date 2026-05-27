# Consistency Validation Report

- Generated at: 2026-05-09T21:22:01
- Python: `C:\Anaconda3\envs\yolo\python.exe`
- Sample images: 8
- Reference: PyTorch `.pt` output
- Compared formats: ONNX and NCNN

| Model | ONNX | ONNX min IoU | ONNX max conf diff | NCNN | NCNN min IoU | NCNN max conf diff | NCNN max box diff |
|---|---|---:|---:|---|---:|---:|---:|
| YOLO26n base | PASS | 0.999901 | 0.000053 | PASS | 0.977280 | 0.055815 | 5.897px |
| YOLO26n EdgeLite | PASS | 0.999849 | 0.000087 | PASS | 0.965875 | 0.060574 | 8.632px |
| YOLO26n EdgeLite + SimAM | PASS | 0.999905 | 0.004549 | PASS | 0.974386 | 0.060154 | 7.892px |
| YOLO26n EdgeLite + WIoU/ProgLoss | PASS | 0.999938 | 0.001075 | PASS | 0.976690 | 0.030192 | 7.117px |

## Interpretation

- ONNX should be nearly identical to PyTorch.
- NCNN may have small numeric differences because it uses a mobile-oriented graph and runtime.
- A final same-class NMS pass is applied to all three outputs before comparison, matching the Android app behavior.
- PASS means class IDs match, detection counts match, and box/confidence differences are within the thresholds recorded in `consistency_report.json`.
