package com.example.yolocoffee;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

final class DetectionOverlayView extends View {
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private Bitmap bitmap;
    private List<YoloOnnxDetector.Detection> detections = new ArrayList<>();

    DetectionOverlayView(Context context) {
        super(context);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(28f);
        labelPaint.setStyle(Paint.Style.FILL);
        setBackgroundColor(Color.rgb(245, 246, 244));
    }

    void setImage(Bitmap bitmap) {
        this.bitmap = bitmap;
        this.detections = new ArrayList<>();
        invalidate();
    }

    void setDetections(List<YoloOnnxDetector.Detection> detections) {
        this.detections = detections == null ? new ArrayList<>() : detections;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap == null) {
            drawEmpty(canvas);
            return;
        }
        RectF imageRect = fitCenterRect(bitmap.getWidth(), bitmap.getHeight(), getWidth(), getHeight());
        canvas.drawBitmap(bitmap, null, imageRect, null);
        float scaleX = imageRect.width() / bitmap.getWidth();
        float scaleY = imageRect.height() / bitmap.getHeight();
        for (YoloOnnxDetector.Detection det : detections) {
            drawDetection(canvas, det, imageRect, scaleX, scaleY);
        }
    }

    private void drawEmpty(Canvas canvas) {
        textPaint.setColor(Color.rgb(70, 80, 76));
        textPaint.setTextSize(34f);
        canvas.drawText("Select an image or run a sample", 32f, getHeight() / 2f, textPaint);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(28f);
    }

    private void drawDetection(Canvas canvas, YoloOnnxDetector.Detection det, RectF imageRect, float scaleX, float scaleY) {
        int color = colorForClass(det.classId);
        boxPaint.setColor(color);
        labelPaint.setColor(color);
        float left = imageRect.left + det.x1 * scaleX;
        float top = imageRect.top + det.y1 * scaleY;
        float right = imageRect.left + det.x2 * scaleX;
        float bottom = imageRect.top + det.y2 * scaleY;
        canvas.drawRect(left, top, right, bottom, boxPaint);

        String label = det.className + " " + String.format(Locale.US, "%.3f", det.confidence);
        float textWidth = textPaint.measureText(label);
        float labelTop = Math.max(imageRect.top, top - 34f);
        canvas.drawRect(left, labelTop, left + textWidth + 16f, labelTop + 34f, labelPaint);
        canvas.drawText(label, left + 8f, labelTop + 25f, textPaint);
    }

    private static RectF fitCenterRect(int srcW, int srcH, int dstW, int dstH) {
        float scale = Math.min(dstW / (float) srcW, dstH / (float) srcH);
        float width = srcW * scale;
        float height = srcH * scale;
        float left = (dstW - width) / 2f;
        float top = (dstH - height) / 2f;
        return new RectF(left, top, left + width, top + height);
    }

    private static int colorForClass(int classId) {
        int[] colors = {
                Color.rgb(222, 91, 66),
                Color.rgb(41, 128, 185),
                Color.rgb(46, 160, 87),
                Color.rgb(142, 68, 173),
                Color.rgb(230, 126, 34),
                Color.rgb(44, 62, 80)
        };
        return colors[Math.abs(classId) % colors.length];
    }
}

