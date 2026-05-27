package com.example.yolocoffee;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.SystemClock;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

final class YoloOnnxDetector implements AutoCloseable {
    static final int INPUT_SIZE = 640;
    static final float CONF_THRESHOLD = 0.25f;
    static final float NMS_IOU_THRESHOLD = 0.70f;

    private final Context context;
    private final ModelConfig config;
    private final List<String> classNames;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    YoloOnnxDetector(Context context, ModelConfig config, List<String> classNames) throws Exception {
        this.context = context.getApplicationContext();
        this.config = config;
        this.classNames = classNames;
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        File modelFile = copyModelToInternalStorage(config);
        this.session = env.createSession(modelFile.getAbsolutePath(), options);
        this.inputName = session.getInputNames().iterator().next();
    }

    DetectionRun detect(Bitmap original, String imageLabel) throws Exception {
        long t0 = SystemClock.elapsedRealtimeNanos();
        PreprocessResult prep = preprocess(original);
        long t1 = SystemClock.elapsedRealtimeNanos();

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(prep.input),
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}
        )) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, tensor);
            long t2 = SystemClock.elapsedRealtimeNanos();
            try (OrtSession.Result ortResult = session.run(inputs)) {
                long t3 = SystemClock.elapsedRealtimeNanos();
                Object value = ortResult.get(0).getValue();
                float[][][] output = (float[][][]) value;
                List<Detection> detections = postprocess(output[0], prep, original.getWidth(), original.getHeight());
                long t4 = SystemClock.elapsedRealtimeNanos();
                return new DetectionRun(
                        config.id,
                        config.displayName,
                        imageLabel,
                        original.getWidth(),
                        original.getHeight(),
                        INPUT_SIZE,
                        INPUT_SIZE,
                        nanosToMillis(t1 - t0),
                        nanosToMillis(t3 - t2),
                        nanosToMillis(t4 - t3),
                        nanosToMillis(t4 - t0),
                        detections
                );
            }
        }
    }

    private PreprocessResult preprocess(Bitmap original) {
        int srcW = original.getWidth();
        int srcH = original.getHeight();
        float scale = Math.min(INPUT_SIZE / (float) srcW, INPUT_SIZE / (float) srcH);
        int resizedW = Math.round(srcW * scale);
        int resizedH = Math.round(srcH * scale);
        float padX = (INPUT_SIZE - resizedW) / 2f;
        float padY = (INPUT_SIZE - resizedH) / 2f;

        Bitmap inputBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(inputBitmap);
        canvas.drawColor(Color.rgb(114, 114, 114));
        RectF dst = new RectF(padX, padY, padX + resizedW, padY + resizedH);
        canvas.drawBitmap(original, null, dst, null);

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        inputBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        float[] input = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int plane = INPUT_SIZE * INPUT_SIZE;
        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            input[i] = ((color >> 16) & 0xFF) / 255.0f;
            input[plane + i] = ((color >> 8) & 0xFF) / 255.0f;
            input[2 * plane + i] = (color & 0xFF) / 255.0f;
        }
        inputBitmap.recycle();
        return new PreprocessResult(input, scale, padX, padY);
    }

    private List<Detection> postprocess(float[][] rows, PreprocessResult prep, int imageW, int imageH) {
        List<Detection> detections = new ArrayList<>();
        for (float[] row : rows) {
            if (row.length < 6) {
                continue;
            }
            float confidence = row[4];
            if (confidence < CONF_THRESHOLD) {
                continue;
            }
            int classId = Math.round(row[5]);
            float x1 = clamp((row[0] - prep.padX) / prep.scale, 0, imageW - 1);
            float y1 = clamp((row[1] - prep.padY) / prep.scale, 0, imageH - 1);
            float x2 = clamp((row[2] - prep.padX) / prep.scale, 0, imageW - 1);
            float y2 = clamp((row[3] - prep.padY) / prep.scale, 0, imageH - 1);
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }
            String className = classId >= 0 && classId < classNames.size() ? classNames.get(classId) : String.valueOf(classId);
            detections.add(new Detection(classId, className, confidence, x1, y1, x2, y2));
        }
        Collections.sort(detections, (a, b) -> Float.compare(b.confidence, a.confidence));
        return nms(detections);
    }

    private List<Detection> nms(List<Detection> sorted) {
        List<Detection> kept = new ArrayList<>();
        for (Detection candidate : sorted) {
            boolean duplicate = false;
            for (Detection old : kept) {
                if (candidate.classId == old.classId && iou(candidate, old) >= NMS_IOU_THRESHOLD) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                kept.add(candidate);
            }
        }
        return kept;
    }

    private File copyModelToInternalStorage(ModelConfig config) throws Exception {
        File dir = new File(context.getFilesDir(), "models");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Unable to create model directory: " + dir);
        }
        File target = new File(dir, config.id + ".onnx");
        try (InputStream input = context.getAssets().open(config.onnxAssetPath);
             FileOutputStream output = new FileOutputStream(target, false)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        return target;
    }

    private static float iou(Detection a, Detection b) {
        float ix1 = Math.max(a.x1, b.x1);
        float iy1 = Math.max(a.y1, b.y1);
        float ix2 = Math.min(a.x2, b.x2);
        float iy2 = Math.min(a.y2, b.y2);
        float iw = Math.max(0f, ix2 - ix1);
        float ih = Math.max(0f, iy2 - iy1);
        float inter = iw * ih;
        float union = a.area() + b.area() - inter;
        return union > 0f ? inter / union : 0f;
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    private static final class PreprocessResult {
        final float[] input;
        final float scale;
        final float padX;
        final float padY;

        PreprocessResult(float[] input, float scale, float padX, float padY) {
            this.input = input;
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
        }
    }

    static final class Detection {
        final int classId;
        final String className;
        final float confidence;
        final float x1;
        final float y1;
        final float x2;
        final float y2;

        Detection(int classId, String className, float confidence, float x1, float y1, float x2, float y2) {
            this.classId = classId;
            this.className = className;
            this.confidence = confidence;
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }

        float width() {
            return x2 - x1;
        }

        float height() {
            return y2 - y1;
        }

        float area() {
            return Math.max(0f, width()) * Math.max(0f, height());
        }
    }

    static final class DetectionRun {
        final String modelId;
        final String modelName;
        final String imageLabel;
        final int imageWidth;
        final int imageHeight;
        final int inputWidth;
        final int inputHeight;
        final double preprocessMs;
        final double inferenceMs;
        final double postprocessMs;
        final double totalMs;
        final List<Detection> detections;

        DetectionRun(
                String modelId,
                String modelName,
                String imageLabel,
                int imageWidth,
                int imageHeight,
                int inputWidth,
                int inputHeight,
                double preprocessMs,
                double inferenceMs,
                double postprocessMs,
                double totalMs,
                List<Detection> detections
        ) {
            this.modelId = modelId;
            this.modelName = modelName;
            this.imageLabel = imageLabel;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.preprocessMs = preprocessMs;
            this.inferenceMs = inferenceMs;
            this.postprocessMs = postprocessMs;
            this.totalMs = totalMs;
            this.detections = detections;
        }
    }
}

