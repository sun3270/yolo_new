# Android Phone Deployment Package

这个文件夹用于整理 `EdgeLite + WIoU/ProgLoss` 模型在安卓手机本地运行前需要的文件和说明。

## 当前推荐模型

- 训练结果目录：`run_down/yolo26n_edgelite_wiou_progloss_coffee30003`
- 权重文件：`model/best_edgelite_wiou_progloss.pt`
- 模型结构：`configs/yolo26n_edgelite.yaml`
- 类别数：6
- 类别名：`algal_spot`、`brown_eye_spot`、`healthy`、`miner`、`phoma`、`powdery_mildew`

## 已放入的文件

| 路径 | 用途 |
|---|---|
| `model/best_edgelite_wiou_progloss.pt` | 当前准备用于导出的最佳 PyTorch 权重 |
| `configs/yolo26n_edgelite.yaml` | EdgeLite 模型结构配置 |
| `reports/training_results.csv` | 最后一次训练全过程指标 |
| `reports/build_compare_report.md` | 原始 YOLO26n 与 EdgeLite 的参数量、FLOPs 对比 |
| `reports/edgelite_design_summary.md` | EdgeLite 结构设计说明 |
| `reports/wiou_progloss_training_readme.md` | WIoU/ProgLoss 训练阶段说明 |
| `custom_ultralytics_reference/block_with_edgelite.py` | 包含 `LDSConv`、`EdgeLGMSFBridge` 等自定义模块的参考文件 |
| `custom_ultralytics_reference/tasks_with_edgelite.py` | 支持解析 EdgeLite YAML 的参考文件 |
| `custom_ultralytics_reference/modules_init_with_edgelite.py` | 自定义模块导出注册参考文件 |
| `exports/` | 后续放导出的 ONNX、NCNN、TFLite 等部署模型 |
| `android_app_assets/` | Android 工程 `assets` 可直接使用的干净 NCNN 文件 |
| `android_app_notes/` | 后续放 Android 工程接入说明、测试记录、截图等 |

## 关键提醒

1. WIoU/ProgLoss 是训练阶段改动，手机端推理不需要实现这两个损失函数。
2. EdgeLite 是推理结构改动，导出模型时必须使用带自定义模块的 Ultralytics 版本。
3. 当前根目录普通 `ultralytics/` 不能直接加载这个 EdgeLite 权重；应使用 `edgelite_experiment/local_ultralytics` 作为导出环境参考。
4. 手机本地运行建议优先走 NCNN 或 TFLite。若使用 Android CPU，NCNN 通常更直接；若要利用 NNAPI/NPU，再评估 TFLite 或厂商 SDK。

## 当前模型摘要

| 指标 | 数值 |
|---|---:|
| 参数量 | 1,951,870 |
| GFLOPs | 4.6626 |
| best.pt 大小 | 4.16 MB |
| mAP50 | 97.68% |
| mAP50-95 | 92.18% |
