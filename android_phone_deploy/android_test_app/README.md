# YOLO Coffee Android Test App

This is a standalone Android Studio project for local phone testing of the four exported coffee-leaf detection models.

## Built-In Models

| Model ID | Name | Runtime |
|---|---|---|
| `01_yolo26n_base` | YOLO26n base | ONNX Runtime Android |
| `02_yolo26n_edgelite` | YOLO26n EdgeLite | ONNX Runtime Android |
| `03_yolo26n_edgelite_simam` | YOLO26n EdgeLite + SimAM | ONNX Runtime Android |
| `04_yolo26n_edgelite_wiou_progloss` | YOLO26n EdgeLite + WIoU/ProgLoss | ONNX Runtime Android |

## How To Run

1. Open this folder in Android Studio: `android_phone_deploy/android_test_app`.
2. Wait for Gradle sync to finish.
3. Connect an Android phone and press Run.
4. Select one of the four models in the app.
5. Press `Run sample` to detect a bundled sample, or press `Pick image` to select a phone image.
6. Press `Batch 100` to randomly select 100 images from the built-in labeled benchmark set and run them sequentially.
7. Press `Export logs` to share the generated CSV/JSONL logs, or pull them from the computer with ADB.

## Built-In Benchmark

The app embeds all 3,791 labeled images from `coffee3000/train`, `coffee3000/valid`, and `coffee3000/test` under:

- `app/src/main/assets/benchmark/images/`
- `app/src/main/assets/benchmark/manifest.json`

`manifest.json` stores the correct YOLO labels for each image and keeps the original split name. During `Batch 100`, the app samples 100 images without replacement from all 3,791 embedded images, runs the selected model on each image, matches predictions to ground truth at IoU `0.50`, and logs per-image accuracy, latency, and match details.

From the repository root, a computer can wait for a phone batch to finish and automatically pull logs:

```powershell
powershell -ExecutionPolicy Bypass -File .\android_phone_deploy\auto_pull_batch_logs.ps1 -WaitForBatch
```

After the phone prints `BATCH_COMPLETE`, logs are copied to:

`android_phone_deploy/analysis_from_phone_logs/detection_logs/`

## Log Files

Every detection run appends records to:

- `runs.csv`: one row per detection run, including model, image, timing, and detection count.
- `detections.csv`: one row per detected box, including class, confidence, coordinates, size, and area.
- `events.jsonl`: complete JSON event per detection run, convenient for scripts or Pandas analysis.
- `benchmark_runs.csv`: one row per benchmark image with ground-truth count, matches, precision, recall, IoU, best confidence, and timing.
- `benchmark_matches.csv`: one row per ground-truth box with the matched prediction box when found.
- `batch_summary.csv`: one row per `Batch 100` run with aggregate precision, recall, IoU, and average latency.

Default phone-side log directory:

`Android/data/com.example.yolocoffee/files/detection_logs/`

## Parameters

- Input size: `640x640`
- Confidence threshold: `0.25`
- NMS IoU threshold: `0.70`
- Classes: `algal_spot`, `brown_eye_spot`, `healthy`, `miner`, `phoma`, `powdery_mildew`

## Notes

- This test app uses ONNX Runtime Android to make model switching, logging, and export straightforward.
- NCNN files are still available in `../android_app_assets` and `../exports` for a later native NCNN version.
- First use of a model includes model loading. Run the same model several times before judging stable latency.
