# Android App Assets

这个目录是给 Android Studio 工程 `app/src/main/assets/` 使用的干净版本。

需要复制的内容：

- `classes.txt`
- `model_index.json`
- `01_yolo26n_base/`
- `02_yolo26n_edgelite/`
- `03_yolo26n_edgelite_simam/`
- `04_yolo26n_edgelite_wiou_progloss/`

每个模型目录只包含 NCNN 推理需要的两个核心文件：

- `model.ncnn.param`
- `model.ncnn.bin`

