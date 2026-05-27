# Detection Log Schema

## runs.csv

One row is appended for every detection run.

| Field | Meaning |
|---|---|
| `run_id` | Unique detection-run ID |
| `timestamp` | ISO timestamp |
| `model_id` | Model ID |
| `model_name` | Human-readable model name |
| `image_label` | Image name, URI label, or bundled sample path |
| `image_width` / `image_height` | Original image size |
| `input_width` / `input_height` | Model input size |
| `confidence_threshold` | Confidence threshold |
| `nms_iou_threshold` | NMS IoU threshold |
| `preprocess_ms` | Preprocess time |
| `inference_ms` | Model inference time |
| `postprocess_ms` | Postprocess time |
| `total_ms` | End-to-end detection time |
| `detection_count` | Number of final boxes |
| `device` | Phone model |

## detections.csv

One row is appended for every detected box.

| Field | Meaning |
|---|---|
| `run_id` | Detection-run ID from `runs.csv` |
| `rank` | Confidence-order rank |
| `class_id` | Class ID |
| `class_name` | Class name |
| `confidence` | Confidence score |
| `x1`, `y1`, `x2`, `y2` | Box coordinates in the original image coordinate system |
| `width`, `height`, `area` | Box dimensions and area |

## events.jsonl

Each line is one complete detection event with the run summary and a `detections` array.

## benchmark_runs.csv

One row is appended for every image processed by `Batch 100`.

| Field | Meaning |
|---|---|
| `batch_id` / `batch_index` | Batch ID and 1-based image order within the batch |
| `run_id` | Detection-run ID from `runs.csv` |
| `model_id` / `model_name` | Model used for this image |
| `image_label` / `sample_id` | Built-in benchmark image identity |
| `ground_truth_count` | Number of correct-answer boxes embedded for this image |
| `detection_count` | Number of predicted boxes after NMS |
| `matched_count` | Predictions matched to ground truth at IoU `0.50` with the same class |
| `false_positive_count` / `missed_count` | Unmatched predictions and unmatched ground-truth boxes |
| `precision` / `recall` / `mean_iou` | Per-image benchmark metrics |
| `best_confidence` | Highest confidence prediction for this image |
| `preprocess_ms` / `inference_ms` / `postprocess_ms` / `total_ms` | Timing for this image |

## benchmark_matches.csv

One row is appended for every ground-truth box in a benchmark image.

| Field | Meaning |
|---|---|
| `truth_class_id` / `truth_class_name` | Correct class |
| `truth_cx`, `truth_cy`, `truth_w`, `truth_h` | Correct YOLO-normalized label |
| `truth_x1`, `truth_y1`, `truth_x2`, `truth_y2` | Correct box converted to original image coordinates |
| `matched` | `1` when a same-class prediction matched at IoU `0.50`, otherwise `0` |
| `detection_*` | Matched prediction details when a match exists |
| `iou` | IoU between the correct box and matched prediction, or best same-class IoU for missed boxes |

## batch_summary.csv

One row is appended when a `Batch 100` run finishes.

| Field | Meaning |
|---|---|
| `batch_id` | Batch ID emitted in Android logcat as `BATCH_COMPLETE` |
| `sample_count` | Number of images processed |
| `ground_truth_count` / `detection_count` / `matched_count` | Aggregate counts |
| `false_positive_count` / `missed_count` | Aggregate errors |
| `precision` / `recall` / `mean_iou` | Aggregate benchmark metrics |
| `avg_*_ms` | Average per-image timing |
| `total_elapsed_ms` | Whole batch wall-clock time on the phone |
