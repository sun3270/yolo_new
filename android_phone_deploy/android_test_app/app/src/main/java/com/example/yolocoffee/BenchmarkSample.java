package com.example.yolocoffee;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

final class BenchmarkSample {
    static final float MATCH_IOU_THRESHOLD = 0.50f;

    final String id;
    final String fileName;
    final String imageAssetPath;
    final List<GroundTruth> groundTruths;

    private BenchmarkSample(String id, String fileName, String imageAssetPath, List<GroundTruth> groundTruths) {
        this.id = id;
        this.fileName = fileName;
        this.imageAssetPath = imageAssetPath;
        this.groundTruths = groundTruths;
    }

    static List<BenchmarkSample> load(Context context) throws Exception {
        JSONObject root = new JSONObject(readAssetText(context, "benchmark/manifest.json"));
        JSONArray samples = root.getJSONArray("samples");
        List<BenchmarkSample> result = new ArrayList<>();
        for (int i = 0; i < samples.length(); i++) {
            JSONObject item = samples.getJSONObject(i);
            JSONArray labels = item.getJSONArray("labels");
            List<GroundTruth> groundTruths = new ArrayList<>();
            for (int j = 0; j < labels.length(); j++) {
                JSONObject label = labels.getJSONObject(j);
                groundTruths.add(new GroundTruth(
                        label.getInt("class_id"),
                        label.optString("class_name", String.valueOf(label.getInt("class_id"))),
                        (float) label.getDouble("cx"),
                        (float) label.getDouble("cy"),
                        (float) label.getDouble("w"),
                        (float) label.getDouble("h")
                ));
            }
            result.add(new BenchmarkSample(
                    item.getString("id"),
                    item.getString("file_name"),
                    item.getString("image"),
                    groundTruths
            ));
        }
        return result;
    }

    Evaluation evaluate(YoloOnnxDetector.DetectionRun run) {
        List<YoloOnnxDetector.Detection> detections = run.detections;
        boolean[] usedDetections = new boolean[detections.size()];
        List<Match> matches = new ArrayList<>();
        int matched = 0;
        double matchedIouSum = 0.0;
        float bestConfidence = 0f;

        for (YoloOnnxDetector.Detection detection : detections) {
            bestConfidence = Math.max(bestConfidence, detection.confidence);
        }

        for (int truthIndex = 0; truthIndex < groundTruths.size(); truthIndex++) {
            GroundTruth truth = groundTruths.get(truthIndex);
            int bestIndex = -1;
            float bestIou = 0f;
            for (int detIndex = 0; detIndex < detections.size(); detIndex++) {
                if (usedDetections[detIndex]) {
                    continue;
                }
                YoloOnnxDetector.Detection detection = detections.get(detIndex);
                if (detection.classId != truth.classId) {
                    continue;
                }
                float iou = iou(truth, detection, run.imageWidth, run.imageHeight);
                if (iou > bestIou) {
                    bestIou = iou;
                    bestIndex = detIndex;
                }
            }

            if (bestIndex >= 0 && bestIou >= MATCH_IOU_THRESHOLD) {
                usedDetections[bestIndex] = true;
                matched++;
                matchedIouSum += bestIou;
                matches.add(new Match(truthIndex, truth, true, bestIndex, detections.get(bestIndex), bestIou));
            } else {
                matches.add(new Match(truthIndex, truth, false, -1, null, bestIou));
            }
        }

        int detectionCount = detections.size();
        int groundTruthCount = groundTruths.size();
        int falsePositiveCount = detectionCount - matched;
        int missedCount = groundTruthCount - matched;
        double precision = detectionCount == 0 ? 0.0 : matched / (double) detectionCount;
        double recall = groundTruthCount == 0 ? 1.0 : matched / (double) groundTruthCount;
        double meanIou = matched == 0 ? 0.0 : matchedIouSum / matched;
        return new Evaluation(
                groundTruthCount,
                detectionCount,
                matched,
                falsePositiveCount,
                missedCount,
                precision,
                recall,
                meanIou,
                bestConfidence,
                matches
        );
    }

    private static float iou(GroundTruth truth, YoloOnnxDetector.Detection detection, int imageWidth, int imageHeight) {
        float tx1 = truth.x1(imageWidth);
        float ty1 = truth.y1(imageHeight);
        float tx2 = truth.x2(imageWidth);
        float ty2 = truth.y2(imageHeight);
        float ix1 = Math.max(tx1, detection.x1);
        float iy1 = Math.max(ty1, detection.y1);
        float ix2 = Math.min(tx2, detection.x2);
        float iy2 = Math.min(ty2, detection.y2);
        float iw = Math.max(0f, ix2 - ix1);
        float ih = Math.max(0f, iy2 - iy1);
        float inter = iw * ih;
        float union = truth.area(imageWidth, imageHeight) + detection.area() - inter;
        return union > 0f ? inter / union : 0f;
    }

    private static String readAssetText(Context context, String assetPath) throws Exception {
        StringBuilder builder = new StringBuilder();
        try (InputStream input = context.getAssets().open(assetPath);
             BufferedReader reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                builder.append(line).append('\n');
            }
        }
        return builder.toString();
    }

    static final class GroundTruth {
        final int classId;
        final String className;
        final float cx;
        final float cy;
        final float boxW;
        final float boxH;

        GroundTruth(int classId, String className, float cx, float cy, float boxW, float boxH) {
            this.classId = classId;
            this.className = className;
            this.cx = cx;
            this.cy = cy;
            this.boxW = boxW;
            this.boxH = boxH;
        }

        float x1(int imageWidth) {
            return clamp((cx - boxW / 2f) * imageWidth, 0f, imageWidth - 1f);
        }

        float y1(int imageHeight) {
            return clamp((cy - boxH / 2f) * imageHeight, 0f, imageHeight - 1f);
        }

        float x2(int imageWidth) {
            return clamp((cx + boxW / 2f) * imageWidth, 0f, imageWidth - 1f);
        }

        float y2(int imageHeight) {
            return clamp((cy + boxH / 2f) * imageHeight, 0f, imageHeight - 1f);
        }

        float area(int imageWidth, int imageHeight) {
            return Math.max(0f, x2(imageWidth) - x1(imageWidth))
                    * Math.max(0f, y2(imageHeight) - y1(imageHeight));
        }

        private static float clamp(float value, float min, float max) {
            return Math.max(min, Math.min(max, value));
        }
    }

    static final class Evaluation {
        final int groundTruthCount;
        final int detectionCount;
        final int matchedCount;
        final int falsePositiveCount;
        final int missedCount;
        final double precision;
        final double recall;
        final double meanIou;
        final float bestConfidence;
        final List<Match> matches;

        Evaluation(
                int groundTruthCount,
                int detectionCount,
                int matchedCount,
                int falsePositiveCount,
                int missedCount,
                double precision,
                double recall,
                double meanIou,
                float bestConfidence,
                List<Match> matches
        ) {
            this.groundTruthCount = groundTruthCount;
            this.detectionCount = detectionCount;
            this.matchedCount = matchedCount;
            this.falsePositiveCount = falsePositiveCount;
            this.missedCount = missedCount;
            this.precision = precision;
            this.recall = recall;
            this.meanIou = meanIou;
            this.bestConfidence = bestConfidence;
            this.matches = matches;
        }
    }

    static final class Match {
        final int truthIndex;
        final GroundTruth truth;
        final boolean matched;
        final int detectionIndex;
        final YoloOnnxDetector.Detection detection;
        final float iou;

        Match(
                int truthIndex,
                GroundTruth truth,
                boolean matched,
                int detectionIndex,
                YoloOnnxDetector.Detection detection,
                float iou
        ) {
            this.truthIndex = truthIndex;
            this.truth = truth;
            this.matched = matched;
            this.detectionIndex = detectionIndex;
            this.detection = detection;
            this.iou = iou;
        }
    }
}
