# App Completion Report

## Status

- Android Studio project created.
- Four ONNX models are included under `app/src/main/assets/models/`.
- Four bundled test images are included under `app/src/main/assets/samples/`.
- Model switching is driven by `app/src/main/assets/model_index.json`.
- Detection classes are loaded from `app/src/main/assets/classes.txt`.
- Every detection run writes analysis logs.

## Runtime

- Runtime: ONNX Runtime Android
- Input size: `640x640`
- Confidence threshold: `0.25`
- NMS IoU threshold: `0.70`

## Logged Files

The app writes these files under:

`Android/data/com.example.yolocoffee/files/detection_logs/`

- `runs.csv`
- `detections.csv`
- `events.jsonl`

## Export Button

The `Export logs` button shares all non-empty log files through Android's system share sheet.

## Local Build Note

This machine does not expose `gradle`, `java`, `adb`, `ANDROID_HOME`, or `ANDROID_SDK_ROOT` on the command line, so the project was not compiled locally in this session. It is ready to open and sync in Android Studio.

