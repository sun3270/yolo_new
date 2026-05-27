# Phone Detection Log Analysis

Source ZIP:

`D:/weixin文件/xwechat_files/wxid_ybe0k83edk3122_9e20/msg/file/2026-05/detection_logs.zip`

## Summary

- Device: `Xiaomi 23127PN0CC`
- Runs: 34
- Detections: 35
- Models tested: 4
- Images tested: 6
- Fully comparable disease image across all four models: `1109_jpg.rf.8f0bbc86151c2df858a86e2af7324405.jpg`

Ground truth for `1109...jpg` is class `4`, `phoma`. All four models detected `phoma`.

## Fair Comparison On Disease Image 1109

| Model | Runs | Avg Inference | Median Inference | Avg Total | Result | Avg Confidence |
|---|---:|---:|---:|---:|---|---:|
| YOLO26n base | 6 | 64.89 ms | 64.20 ms | 76.52 ms | phoma | 0.9676 |
| YOLO26n EdgeLite | 7 | 60.52 ms | 58.48 ms | 71.35 ms | phoma | 0.9608 |
| YOLO26n EdgeLite + SimAM | 7 | 61.30 ms | 59.50 ms | 72.57 ms | phoma | 0.9544 |
| YOLO26n EdgeLite + WIoU/ProgLoss | 7 | 64.61 ms | 60.29 ms | 75.67 ms | phoma | 0.9695 |

## Interpretation

- Accuracy on the tested disease image is good: all four models predicted `phoma`.
- EdgeLite is the fastest in this phone log.
- EdgeLite + SimAM is slightly slower than pure EdgeLite.
- EdgeLite + WIoU/ProgLoss has the highest average confidence on the disease image, but is not the fastest in this run.
- The difference is small. Median latency suggests all EdgeLite variants are faster than the base model, but the WIoU/ProgLoss version has more timing fluctuation.

## Testing Gaps

- `1185_jpg.rf.ef02249b4333ffb5c11d39a75cc86420.jpg` was transferred to the phone but does not appear in this log.
- Healthy sample images were only tested with the base model, not all four models.
- One screenshot was tested with the base model; screenshots are not good evaluation samples because they include app UI and may cause misleading detections.

## Next Test Recommendation

For a fair model comparison, test the same image set with every model:

1. `1109...jpg`
2. `1185...jpg`
3. Two healthy images
4. Two additional disease images from different classes if available

Run each image-model pair 5 to 10 times, then compare median inference time and detection correctness.

