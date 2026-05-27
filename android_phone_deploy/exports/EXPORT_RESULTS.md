# Export Results

- Generated at: 2026-05-09T11:58:31
- Input size: 640x640
- Classes: algal_spot, brown_eye_spot, healthy, miner, phoma, powdery_mildew

| Model | PT | ONNX | NCNN |
|---|---:|---:|---:|
| 01_yolo26n_base | 5.15 MB | ok (9.31 MB) | ok: `model.ncnn.param` + `model.ncnn.bin` (9.16 MB bin) |
| 02_yolo26n_edgelite | 4.15 MB | ok (7.21 MB) | ok: `model.ncnn.param` + `model.ncnn.bin` (7.04 MB bin) |
| 03_yolo26n_edgelite_simam | 4.15 MB | ok (7.21 MB) | ok: `model.ncnn.param` + `model.ncnn.bin` (7.04 MB bin) |
| 04_yolo26n_edgelite_wiou_progloss | 4.16 MB | ok (7.21 MB) | ok: `model.ncnn.param` + `model.ncnn.bin` (7.04 MB bin) |

## Notes

- WIoU/ProgLoss only affects training. Phone-side inference only needs the exported model and YOLO postprocess.
- EdgeLite models require the custom Ultralytics modules during export, but the Android app should use the exported NCNN/TFLite/ONNX files.
- For Android local testing, prefer NCNN when available.
- On Windows, PNNX exited with a non-zero status after writing the NCNN files. The required Android-side NCNN files are present and non-empty.
