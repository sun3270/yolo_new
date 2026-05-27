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

final class ModelConfig {
    final String id;
    final String displayName;
    final String onnxAssetPath;

    ModelConfig(String id, String displayName, String onnxAssetPath) {
        this.id = id;
        this.displayName = displayName;
        this.onnxAssetPath = onnxAssetPath;
    }

    @Override
    public String toString() {
        return displayName;
    }

    static List<ModelConfig> loadModels(Context context) throws Exception {
        JSONObject root = new JSONObject(readAssetText(context, "model_index.json"));
        JSONArray models = root.getJSONArray("models");
        List<ModelConfig> result = new ArrayList<>();
        for (int i = 0; i < models.length(); i++) {
            JSONObject item = models.getJSONObject(i);
            result.add(new ModelConfig(
                    item.getString("id"),
                    item.getString("display_name"),
                    item.getString("onnx")
            ));
        }
        return result;
    }

    static List<String> loadClassNames(Context context) throws Exception {
        List<String> names = new ArrayList<>();
        String text = readAssetText(context, "classes.txt");
        for (String line : text.split("\\R")) {
            String trimmed = line.trim();
            if (!trimmed.isEmpty()) {
                names.add(trimmed);
            }
        }
        return names;
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
}

