package com.example.yolocoffee;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.database.Cursor;
import android.util.Log;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.security.SecureRandom;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_IMAGE_REQUEST = 41;
    private static final int BATCH_SAMPLE_COUNT = 100;
    private static final String BATCH_LOG_TAG = "YoloCoffeeBatch";

    private Spinner modelSpinner;
    private DetectionOverlayView overlayView;
    private TextView statusView;
    private Button pickButton;
    private Button sampleButton;
    private Button detectButton;
    private Button batchButton;
    private Button exportButton;
    private Button clearButton;
    private Bitmap currentBitmap;
    private String currentImageLabel = "none";
    private List<ModelConfig> modelConfigs;
    private List<String> classNames;
    private List<BenchmarkSample> benchmarkSamples = new ArrayList<>();
    private String[] sampleNames = new String[0];
    private int sampleIndex = 0;
    private boolean batchRunning = false;
    private DetectionLogger logger;
    private final Map<String, YoloOnnxDetector> detectors = new HashMap<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            modelConfigs = ModelConfig.loadModels(this);
            classNames = ModelConfig.loadClassNames(this);
            sampleNames = getAssets().list("samples");
            benchmarkSamples = BenchmarkSample.load(this);
            logger = new DetectionLogger(this);
            buildUi();
            setStatus("Ready. Pick an image, run a sample, or batch-test 100 of "
                    + benchmarkSamples.size() + " bundled labeled images.");
        } catch (Exception e) {
            TextView error = new TextView(this);
            error.setPadding(dp(16), dp(16), dp(16), dp(16));
            error.setText("Initialization failed:\n" + e.getMessage());
            setContentView(error);
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(12), dp(12), dp(12), dp(12));

        TextView title = new TextView(this);
        title.setText("YOLO Coffee Local Test");
        title.setTextSize(20f);
        title.setGravity(Gravity.CENTER_VERTICAL);
        root.addView(title, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        modelSpinner = new Spinner(this);
        ArrayAdapter<ModelConfig> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_dropdown_item,
                modelConfigs
        );
        modelSpinner.setAdapter(adapter);
        root.addView(modelSpinner, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        HorizontalScrollView scroll = new HorizontalScrollView(this);
        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);
        scroll.addView(buttons);

        pickButton = makeButton("Pick image");
        pickButton.setOnClickListener(v -> pickImage());
        buttons.addView(pickButton);

        sampleButton = makeButton("Run sample");
        sampleButton.setOnClickListener(v -> loadNextSampleAndDetect());
        buttons.addView(sampleButton);

        detectButton = makeButton("Detect");
        detectButton.setOnClickListener(v -> runDetection());
        buttons.addView(detectButton);

        batchButton = makeButton("Batch 100");
        batchButton.setOnClickListener(v -> runBenchmarkBatch());
        buttons.addView(batchButton);

        exportButton = makeButton("Export logs");
        exportButton.setOnClickListener(v -> logger.shareLogs(this));
        buttons.addView(exportButton);

        clearButton = makeButton("Clear logs");
        clearButton.setOnClickListener(v -> {
            logger.clearLogs();
            setStatus("Logs cleared. Log directory: " + logger.getLogDir().getAbsolutePath());
        });
        buttons.addView(clearButton);

        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        overlayView = new DetectionOverlayView(this);
        root.addView(overlayView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
        ));

        statusView = new TextView(this);
        statusView.setTextSize(14f);
        statusView.setPadding(0, dp(8), 0, 0);
        root.addView(statusView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        setContentView(root);
    }

    private Button makeButton(String text) {
        Button button = new Button(this);
        button.setText(text);
        button.setAllCaps(false);
        button.setMinWidth(dp(112));
        return button;
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            loadImageFromUri(data.getData());
        }
    }

    private void loadImageFromUri(Uri uri) {
        try (InputStream input = getContentResolver().openInputStream(uri)) {
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) {
                setStatus("Unable to decode selected image.");
                return;
            }
            setCurrentImage(bitmap, displayNameForUri(uri));
            setStatus("Image loaded: " + currentImageLabel + " (" + bitmap.getWidth() + "x" + bitmap.getHeight() + ")");
        } catch (Exception e) {
            setStatus("Image load failed: " + e.getMessage());
        }
    }

    private void loadNextSampleAndDetect() {
        if (sampleNames == null || sampleNames.length == 0) {
            setStatus("No bundled samples found.");
            return;
        }
        String sample = sampleNames[sampleIndex % sampleNames.length];
        sampleIndex++;
        try (InputStream input = getAssets().open("samples/" + sample)) {
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) {
                setStatus("Unable to decode sample: " + sample);
                return;
            }
            setCurrentImage(bitmap, "sample/" + sample);
            runDetection();
        } catch (Exception e) {
            setStatus("Sample load failed: " + e.getMessage());
        }
    }

    private void runDetection() {
        if (currentBitmap == null) {
            setStatus("Load an image first.");
            return;
        }
        ModelConfig config = (ModelConfig) modelSpinner.getSelectedItem();
        Bitmap bitmap = currentBitmap;
        String imageLabel = currentImageLabel;
        setStatus("Running " + config.displayName + "...");
        executor.execute(() -> {
            try {
                YoloOnnxDetector detector = detectorFor(config);
                YoloOnnxDetector.DetectionRun run = detector.detect(bitmap, imageLabel);
                String runId = logger.log(run);
                runOnUiThread(() -> {
                    overlayView.setDetections(run.detections);
                    setStatus(summaryText(runId, run));
                });
            } catch (Exception e) {
                runOnUiThread(() -> setStatus("Detection failed: " + e.getMessage()));
            }
        });
    }

    private void runBenchmarkBatch() {
        if (batchRunning) {
            setStatus("Batch test is already running.");
            return;
        }
        if (benchmarkSamples == null || benchmarkSamples.size() < BATCH_SAMPLE_COUNT) {
            setStatus("Need at least " + BATCH_SAMPLE_COUNT + " bundled benchmark images. Found: "
                    + (benchmarkSamples == null ? 0 : benchmarkSamples.size()));
            return;
        }

        ModelConfig config = (ModelConfig) modelSpinner.getSelectedItem();
        String batchId = UUID.randomUUID().toString();
        List<BenchmarkSample> shuffled = new ArrayList<>(benchmarkSamples);
        Collections.shuffle(shuffled, new SecureRandom());
        final List<BenchmarkSample> selected = new ArrayList<>(shuffled.subList(0, BATCH_SAMPLE_COUNT));

        batchRunning = true;
        setControlsEnabled(false);
        setStatus("Batch " + batchId.substring(0, 8) + " started: 100 random images from "
                + benchmarkSamples.size() + " bundled images with " + config.displayName + ".");
        Log.i(BATCH_LOG_TAG, "BATCH_START " + batchId + " model=" + config.id);

        executor.execute(() -> {
            BatchAccumulator accumulator = new BatchAccumulator();
            long batchStart = SystemClock.elapsedRealtimeNanos();
            int processed = 0;
            try {
                YoloOnnxDetector detector = detectorFor(config);
                for (BenchmarkSample sample : selected) {
                    if (Thread.currentThread().isInterrupted()) {
                        break;
                    }
                    Bitmap bitmap = loadBitmapFromAsset(sample.imageAssetPath);
                    if (bitmap == null) {
                        continue;
                    }
                    String imageLabel = "benchmark/" + sample.fileName;
                    YoloOnnxDetector.DetectionRun run = detector.detect(bitmap, imageLabel);
                    BenchmarkSample.Evaluation evaluation = sample.evaluate(run);
                    String runId = logger.log(run);
                    logger.logBenchmarkRun(batchId, processed + 1, runId, run, sample, evaluation);
                    accumulator.add(run, evaluation);
                    processed++;

                    int done = processed;
                    Bitmap displayBitmap = bitmap;
                    YoloOnnxDetector.DetectionRun displayRun = run;
                    BenchmarkSample.Evaluation displayEvaluation = evaluation;
                    runOnUiThread(() -> {
                        setCurrentImage(displayBitmap, imageLabel, true);
                        overlayView.setDetections(displayRun.detections);
                        setStatus(String.format(Locale.US,
                                "Batch %s | %d/%d | match %d/%d | precision %.3f | recall %.3f | total %.2f ms",
                                batchId.substring(0, 8),
                                done,
                                BATCH_SAMPLE_COUNT,
                                displayEvaluation.matchedCount,
                                displayEvaluation.groundTruthCount,
                                displayEvaluation.precision,
                                displayEvaluation.recall,
                                displayRun.totalMs));
                    });
                }

                double elapsedMs = nanosToMillis(SystemClock.elapsedRealtimeNanos() - batchStart);
                logger.logBenchmarkSummary(
                        batchId,
                        config,
                        processed,
                        accumulator.groundTruthCount,
                        accumulator.detectionCount,
                        accumulator.matchedCount,
                        accumulator.falsePositiveCount,
                        accumulator.missedCount,
                        accumulator.precision(),
                        accumulator.recall(),
                        accumulator.meanIou(),
                        accumulator.avgPreprocessMs(),
                        accumulator.avgInferenceMs(),
                        accumulator.avgPostprocessMs(),
                        accumulator.avgTotalMs(),
                        elapsedMs
                );
                Log.i(BATCH_LOG_TAG, "BATCH_COMPLETE " + batchId + " logs=" + logger.getLogDir().getAbsolutePath());

                int finalProcessed = processed;
                runOnUiThread(() -> {
                    batchRunning = false;
                    setControlsEnabled(true);
                    setStatus(String.format(Locale.US,
                            "Batch %s finished | images %d | precision %.3f | recall %.3f | avg total %.2f ms%nLogs: %s",
                            batchId.substring(0, 8),
                            finalProcessed,
                            accumulator.precision(),
                            accumulator.recall(),
                            accumulator.avgTotalMs(),
                            logger.getLogDir().getAbsolutePath()));
                });
            } catch (Exception e) {
                Log.e(BATCH_LOG_TAG, "BATCH_FAILED " + batchId, e);
                runOnUiThread(() -> {
                    batchRunning = false;
                    setControlsEnabled(true);
                    setStatus("Batch failed: " + e.getMessage());
                });
            }
        });
    }

    private synchronized YoloOnnxDetector detectorFor(ModelConfig config) throws Exception {
        YoloOnnxDetector existing = detectors.get(config.id);
        if (existing != null) {
            return existing;
        }
        YoloOnnxDetector created = new YoloOnnxDetector(this, config, classNames);
        detectors.put(config.id, created);
        return created;
    }

    private String summaryText(String runId, YoloOnnxDetector.DetectionRun run) {
        StringBuilder builder = new StringBuilder();
        builder.append("Run ").append(runId.substring(0, 8))
                .append(" | ").append(run.modelName)
                .append(" | detections: ").append(run.detections.size())
                .append(" | total ").append(fmt(run.totalMs)).append(" ms")
                .append(" | preprocess ").append(fmt(run.preprocessMs)).append(" ms")
                .append(" | inference ").append(fmt(run.inferenceMs)).append(" ms")
                .append(" | postprocess ").append(fmt(run.postprocessMs)).append(" ms")
                .append('\n')
                .append("Logged to: ").append(logger.getLogDir().getAbsolutePath());
        return builder.toString();
    }

    private Bitmap loadBitmapFromAsset(String assetPath) throws Exception {
        try (InputStream input = getAssets().open(assetPath)) {
            return BitmapFactory.decodeStream(input);
        }
    }

    private void setCurrentImage(Bitmap bitmap, String imageLabel) {
        setCurrentImage(bitmap, imageLabel, false);
    }

    private void setCurrentImage(Bitmap bitmap, String imageLabel, boolean recyclePrevious) {
        Bitmap oldBitmap = currentBitmap;
        currentBitmap = bitmap;
        currentImageLabel = imageLabel;
        overlayView.setImage(currentBitmap);
        if (recyclePrevious && oldBitmap != null && oldBitmap != currentBitmap && !oldBitmap.isRecycled()) {
            oldBitmap.recycle();
        }
    }

    private void setControlsEnabled(boolean enabled) {
        modelSpinner.setEnabled(enabled);
        pickButton.setEnabled(enabled);
        sampleButton.setEnabled(enabled);
        detectButton.setEnabled(enabled);
        batchButton.setEnabled(enabled);
        exportButton.setEnabled(enabled);
        clearButton.setEnabled(enabled);
    }

    private String displayNameForUri(Uri uri) {
        try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    return cursor.getString(index);
                }
            }
        } catch (Exception ignored) {
        }
        return uri.toString();
    }

    private void setStatus(String text) {
        if (statusView != null) {
            statusView.setText(text);
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static String fmt(double value) {
        return String.format(Locale.US, "%.2f", value);
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        for (YoloOnnxDetector detector : detectors.values()) {
            try {
                detector.close();
            } catch (Exception ignored) {
            }
        }
        if (currentBitmap != null) {
            currentBitmap.recycle();
        }
    }

    private static final class BatchAccumulator {
        int sampleCount;
        int groundTruthCount;
        int detectionCount;
        int matchedCount;
        int falsePositiveCount;
        int missedCount;
        double meanIouSum;
        double preprocessMsSum;
        double inferenceMsSum;
        double postprocessMsSum;
        double totalMsSum;

        void add(YoloOnnxDetector.DetectionRun run, BenchmarkSample.Evaluation evaluation) {
            sampleCount++;
            groundTruthCount += evaluation.groundTruthCount;
            detectionCount += evaluation.detectionCount;
            matchedCount += evaluation.matchedCount;
            falsePositiveCount += evaluation.falsePositiveCount;
            missedCount += evaluation.missedCount;
            meanIouSum += evaluation.meanIou * evaluation.matchedCount;
            preprocessMsSum += run.preprocessMs;
            inferenceMsSum += run.inferenceMs;
            postprocessMsSum += run.postprocessMs;
            totalMsSum += run.totalMs;
        }

        double precision() {
            return detectionCount == 0 ? 0.0 : matchedCount / (double) detectionCount;
        }

        double recall() {
            return groundTruthCount == 0 ? 1.0 : matchedCount / (double) groundTruthCount;
        }

        double meanIou() {
            return matchedCount == 0 ? 0.0 : meanIouSum / matchedCount;
        }

        double avgPreprocessMs() {
            return sampleCount == 0 ? 0.0 : preprocessMsSum / sampleCount;
        }

        double avgInferenceMs() {
            return sampleCount == 0 ? 0.0 : inferenceMsSum / sampleCount;
        }

        double avgPostprocessMs() {
            return sampleCount == 0 ? 0.0 : postprocessMsSum / sampleCount;
        }

        double avgTotalMs() {
            return sampleCount == 0 ? 0.0 : totalMsSum / sampleCount;
        }
    }
}
