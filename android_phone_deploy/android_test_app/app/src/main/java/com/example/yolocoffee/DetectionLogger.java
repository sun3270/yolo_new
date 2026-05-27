package com.example.yolocoffee;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

final class DetectionLogger {
    private final Context context;
    private final File logDir;
    private final File runsCsv;
    private final File detectionsCsv;
    private final File eventsJsonl;
    private final File benchmarkRunsCsv;
    private final File benchmarkMatchesCsv;
    private final File batchSummaryCsv;

    DetectionLogger(Context context) {
        this.context = context.getApplicationContext();
        this.logDir = new File(context.getExternalFilesDir(null), "detection_logs");
        if (!logDir.exists()) {
            logDir.mkdirs();
        }
        this.runsCsv = new File(logDir, "runs.csv");
        this.detectionsCsv = new File(logDir, "detections.csv");
        this.eventsJsonl = new File(logDir, "events.jsonl");
        this.benchmarkRunsCsv = new File(logDir, "benchmark_runs.csv");
        this.benchmarkMatchesCsv = new File(logDir, "benchmark_matches.csv");
        this.batchSummaryCsv = new File(logDir, "batch_summary.csv");
        ensureHeaders();
    }

    String log(YoloOnnxDetector.DetectionRun run) throws IOException {
        String runId = UUID.randomUUID().toString();
        String timestamp = Instant.now().toString();
        String device = deviceName();

        appendLine(runsCsv, String.join(",",
                csv(runId),
                csv(timestamp),
                csv(run.modelId),
                csv(run.modelName),
                csv(run.imageLabel),
                String.valueOf(run.imageWidth),
                String.valueOf(run.imageHeight),
                String.valueOf(run.inputWidth),
                String.valueOf(run.inputHeight),
                fmt(YoloOnnxDetector.CONF_THRESHOLD),
                fmt(YoloOnnxDetector.NMS_IOU_THRESHOLD),
                fmt(run.preprocessMs),
                fmt(run.inferenceMs),
                fmt(run.postprocessMs),
                fmt(run.totalMs),
                String.valueOf(run.detections.size()),
                csv(device)
        ));

        int rank = 0;
        for (YoloOnnxDetector.Detection det : run.detections) {
            appendLine(detectionsCsv, String.join(",",
                    csv(runId),
                    String.valueOf(rank),
                    String.valueOf(det.classId),
                    csv(det.className),
                    fmt(det.confidence),
                    fmt(det.x1),
                    fmt(det.y1),
                    fmt(det.x2),
                    fmt(det.y2),
                    fmt(det.width()),
                    fmt(det.height()),
                    fmt(det.area())
            ));
            rank++;
        }

        appendLine(eventsJsonl, toJsonLine(runId, timestamp, device, run));
        return runId;
    }

    void logBenchmarkRun(
            String batchId,
            int batchIndex,
            String runId,
            YoloOnnxDetector.DetectionRun run,
            BenchmarkSample sample,
            BenchmarkSample.Evaluation evaluation
    ) throws IOException {
        String timestamp = Instant.now().toString();
        String device = deviceName();
        appendLine(benchmarkRunsCsv, String.join(",",
                csv(batchId),
                String.valueOf(batchIndex),
                csv(runId),
                csv(timestamp),
                csv(run.modelId),
                csv(run.modelName),
                csv(run.imageLabel),
                csv(sample.id),
                String.valueOf(run.imageWidth),
                String.valueOf(run.imageHeight),
                String.valueOf(evaluation.groundTruthCount),
                String.valueOf(evaluation.detectionCount),
                String.valueOf(evaluation.matchedCount),
                String.valueOf(evaluation.falsePositiveCount),
                String.valueOf(evaluation.missedCount),
                fmt(evaluation.precision),
                fmt(evaluation.recall),
                fmt(evaluation.meanIou),
                fmt(evaluation.bestConfidence),
                fmt(run.preprocessMs),
                fmt(run.inferenceMs),
                fmt(run.postprocessMs),
                fmt(run.totalMs),
                csv(device)
        ));

        for (BenchmarkSample.Match match : evaluation.matches) {
            BenchmarkSample.GroundTruth truth = match.truth;
            YoloOnnxDetector.Detection detection = match.detection;
            appendLine(benchmarkMatchesCsv, String.join(",",
                    csv(batchId),
                    String.valueOf(batchIndex),
                    csv(runId),
                    csv(run.imageLabel),
                    csv(sample.id),
                    String.valueOf(match.truthIndex),
                    String.valueOf(truth.classId),
                    csv(truth.className),
                    fmt(truth.cx),
                    fmt(truth.cy),
                    fmt(truth.boxW),
                    fmt(truth.boxH),
                    fmt(truth.x1(run.imageWidth)),
                    fmt(truth.y1(run.imageHeight)),
                    fmt(truth.x2(run.imageWidth)),
                    fmt(truth.y2(run.imageHeight)),
                    match.matched ? "1" : "0",
                    match.matched ? String.valueOf(match.detectionIndex) : "",
                    detection == null ? "" : String.valueOf(detection.classId),
                    detection == null ? "" : csv(detection.className),
                    detection == null ? "" : fmt(detection.confidence),
                    detection == null ? "" : fmt(detection.x1),
                    detection == null ? "" : fmt(detection.y1),
                    detection == null ? "" : fmt(detection.x2),
                    detection == null ? "" : fmt(detection.y2),
                    detection == null ? "" : fmt(detection.width()),
                    detection == null ? "" : fmt(detection.height()),
                    detection == null ? "" : fmt(detection.area()),
                    fmt(match.iou)
            ));
        }
    }

    void logBenchmarkSummary(
            String batchId,
            ModelConfig config,
            int sampleCount,
            int groundTruthCount,
            int detectionCount,
            int matchedCount,
            int falsePositiveCount,
            int missedCount,
            double precision,
            double recall,
            double meanIou,
            double avgPreprocessMs,
            double avgInferenceMs,
            double avgPostprocessMs,
            double avgTotalMs,
            double elapsedMs
    ) throws IOException {
        appendLine(batchSummaryCsv, String.join(",",
                csv(batchId),
                csv(Instant.now().toString()),
                csv(config.id),
                csv(config.displayName),
                String.valueOf(sampleCount),
                String.valueOf(groundTruthCount),
                String.valueOf(detectionCount),
                String.valueOf(matchedCount),
                String.valueOf(falsePositiveCount),
                String.valueOf(missedCount),
                fmt(precision),
                fmt(recall),
                fmt(meanIou),
                fmt(avgPreprocessMs),
                fmt(avgInferenceMs),
                fmt(avgPostprocessMs),
                fmt(avgTotalMs),
                fmt(elapsedMs),
                csv(deviceName())
        ));
    }

    void shareLogs(Activity activity) {
        ArrayList<Uri> uris = new ArrayList<>();
        for (File file : getLogFiles()) {
            if (file.exists() && file.length() > 0) {
                uris.add(FileProvider.getUriForFile(context, context.getPackageName() + ".fileprovider", file));
            }
        }
        if (uris.isEmpty()) {
            return;
        }
        Intent intent = new Intent(Intent.ACTION_SEND_MULTIPLE);
        intent.setType("text/*");
        intent.putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        activity.startActivity(Intent.createChooser(intent, "Export detection logs"));
    }

    void clearLogs() {
        for (File file : getLogFiles()) {
            if (file.exists()) {
                file.delete();
            }
        }
        ensureHeaders();
    }

    File getLogDir() {
        return logDir;
    }

    private List<File> getLogFiles() {
        List<File> files = new ArrayList<>();
        files.add(runsCsv);
        files.add(detectionsCsv);
        files.add(eventsJsonl);
        files.add(benchmarkRunsCsv);
        files.add(benchmarkMatchesCsv);
        files.add(batchSummaryCsv);
        return files;
    }

    private void ensureHeaders() {
        try {
            if (!runsCsv.exists() || runsCsv.length() == 0) {
                appendLine(runsCsv, "run_id,timestamp,model_id,model_name,image_label,image_width,image_height,input_width,input_height,confidence_threshold,nms_iou_threshold,preprocess_ms,inference_ms,postprocess_ms,total_ms,detection_count,device");
            }
            if (!detectionsCsv.exists() || detectionsCsv.length() == 0) {
                appendLine(detectionsCsv, "run_id,rank,class_id,class_name,confidence,x1,y1,x2,y2,width,height,area");
            }
            if (!eventsJsonl.exists()) {
                eventsJsonl.createNewFile();
            }
            if (!benchmarkRunsCsv.exists() || benchmarkRunsCsv.length() == 0) {
                appendLine(benchmarkRunsCsv, "batch_id,batch_index,run_id,timestamp,model_id,model_name,image_label,sample_id,image_width,image_height,ground_truth_count,detection_count,matched_count,false_positive_count,missed_count,precision,recall,mean_iou,best_confidence,preprocess_ms,inference_ms,postprocess_ms,total_ms,device");
            }
            if (!benchmarkMatchesCsv.exists() || benchmarkMatchesCsv.length() == 0) {
                appendLine(benchmarkMatchesCsv, "batch_id,batch_index,run_id,image_label,sample_id,truth_rank,truth_class_id,truth_class_name,truth_cx,truth_cy,truth_w,truth_h,truth_x1,truth_y1,truth_x2,truth_y2,matched,detection_rank,detection_class_id,detection_class_name,confidence,detection_x1,detection_y1,detection_x2,detection_y2,detection_width,detection_height,detection_area,iou");
            }
            if (!batchSummaryCsv.exists() || batchSummaryCsv.length() == 0) {
                appendLine(batchSummaryCsv, "batch_id,timestamp,model_id,model_name,sample_count,ground_truth_count,detection_count,matched_count,false_positive_count,missed_count,precision,recall,mean_iou,avg_preprocess_ms,avg_inference_ms,avg_postprocess_ms,avg_total_ms,total_elapsed_ms,device");
            }
        } catch (IOException ignored) {
        }
    }

    private static void appendLine(File file, String line) throws IOException {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (FileWriter writer = new FileWriter(file, true)) {
            writer.write(line);
            writer.write('\n');
        }
    }

    private static String deviceName() {
        return Build.MANUFACTURER + " " + Build.MODEL;
    }

    private static String toJsonLine(String runId, String timestamp, String device, YoloOnnxDetector.DetectionRun run) {
        StringBuilder builder = new StringBuilder();
        builder.append('{')
                .append("\"run_id\":").append(json(runId)).append(',')
                .append("\"timestamp\":").append(json(timestamp)).append(',')
                .append("\"model_id\":").append(json(run.modelId)).append(',')
                .append("\"model_name\":").append(json(run.modelName)).append(',')
                .append("\"image_label\":").append(json(run.imageLabel)).append(',')
                .append("\"image_width\":").append(run.imageWidth).append(',')
                .append("\"image_height\":").append(run.imageHeight).append(',')
                .append("\"input_width\":").append(run.inputWidth).append(',')
                .append("\"input_height\":").append(run.inputHeight).append(',')
                .append("\"confidence_threshold\":").append(fmt(YoloOnnxDetector.CONF_THRESHOLD)).append(',')
                .append("\"nms_iou_threshold\":").append(fmt(YoloOnnxDetector.NMS_IOU_THRESHOLD)).append(',')
                .append("\"preprocess_ms\":").append(fmt(run.preprocessMs)).append(',')
                .append("\"inference_ms\":").append(fmt(run.inferenceMs)).append(',')
                .append("\"postprocess_ms\":").append(fmt(run.postprocessMs)).append(',')
                .append("\"total_ms\":").append(fmt(run.totalMs)).append(',')
                .append("\"detection_count\":").append(run.detections.size()).append(',')
                .append("\"device\":").append(json(device)).append(',')
                .append("\"detections\":[");
        for (int i = 0; i < run.detections.size(); i++) {
            if (i > 0) {
                builder.append(',');
            }
            YoloOnnxDetector.Detection det = run.detections.get(i);
            builder.append('{')
                    .append("\"rank\":").append(i).append(',')
                    .append("\"class_id\":").append(det.classId).append(',')
                    .append("\"class_name\":").append(json(det.className)).append(',')
                    .append("\"confidence\":").append(fmt(det.confidence)).append(',')
                    .append("\"x1\":").append(fmt(det.x1)).append(',')
                    .append("\"y1\":").append(fmt(det.y1)).append(',')
                    .append("\"x2\":").append(fmt(det.x2)).append(',')
                    .append("\"y2\":").append(fmt(det.y2)).append(',')
                    .append("\"width\":").append(fmt(det.width())).append(',')
                    .append("\"height\":").append(fmt(det.height())).append(',')
                    .append("\"area\":").append(fmt(det.area()))
                    .append('}');
        }
        builder.append("]}");
        return builder.toString();
    }

    private static String csv(String value) {
        if (value == null) {
            return "";
        }
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    private static String json(String value) {
        if (value == null) {
            return "null";
        }
        return "\"" + value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r") + "\"";
    }

    private static String fmt(double value) {
        return String.format(Locale.US, "%.6f", value);
    }
}
