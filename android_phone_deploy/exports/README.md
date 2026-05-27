# Exported Models

这里放已经导出的安卓部署模型文件。

当前每款模型都有：

- `.onnx`：电脑端一致性验证和中间格式。
- `_ncnn_model/model.ncnn.param`：Android NCNN 模型结构文件。
- `_ncnn_model/model.ncnn.bin`：Android NCNN 权重文件。

辅助文件：

- `model_index.json`：四款模型的切换索引。
- `classes.txt`：6 个类别名。
- `EXPORT_RESULTS.md`：导出结果和文件大小。
