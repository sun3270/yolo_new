# YOLO Coffee Leaf Disease Detection

本仓库基于 Ultralytics YOLO26n，面向咖啡叶病害检测、轻量化训练和 Android 端部署验证。当前主要实验路线包括：

- 原始 YOLO26n 基线训练
- YOLO26n-EdgeLite 轻量化结构
- EdgeLite + WIoU/ProgLoss 损失函数优化
- P5Slim-512 第二阶段轻量化蒸馏
- ONNX/NCNN 导出与 Android 本地测试工程

## 当前版本

| 版本 | 分支或标签 | 说明 |
|---|---|---|
| `v1.0` | tag | 初始可复现实验版本 |
| `v2.0` | tag / `main` 起点 | EdgeLite + WIoU/ProgLoss 隔离实验版本 |
| `v3.0` | tag | P5Slim-512 第二阶段轻量化版本 |
| `v3.1` | tag / `codex/v3-p5slim512-second-lightweight` | 当前上传版本，包含 Android 部署文件夹、进度报告、NCNN 资源和日志分析 |

GitHub 首页默认显示 `main` 分支；如果要查看最新 3.1 内容，请切换到：

```text
codex/v3-p5slim512-second-lightweight
```

或在 Tags 中选择：

```text
v3.1
```

## 任务与数据集

任务是 6 类咖啡叶病害目标检测：

| 类别编号 | 类别名 |
|---:|---|
| 0 | `algal_spot` |
| 1 | `brown_eye_spot` |
| 2 | `healthy` |
| 3 | `miner` |
| 4 | `phoma` |
| 5 | `powdery_mildew` |

数据配置文件位于：

```text
coffee3000/coffee3000.yaml
```

注意：训练图片和标签目录通常不提交到 Git，需本地放置在 `coffee3000/train`、`coffee3000/valid`、`coffee3000/test`。

## 核心结果

当前综合效果最好的训练路线是：

```text
YOLO26n-EdgeLite + WIoU/ProgLoss
```

验证集最佳结果：

| 指标 | 结果 |
|---|---:|
| 参数量 | 1,951,870 |
| GFLOPs | 4.6626 |
| best epoch | 272 |
| Precision | 95.60% |
| Recall | 94.18% |
| mAP50 | 97.675% |
| mAP50-95 | 92.179% |

结构对比：

| 模型 | 参数量 | GFLOPs | 层数 |
|---|---:|---:|---:|
| 原始 YOLO26n | 2,506,140 | 5.7825 | 260 |
| YOLO26n-EdgeLite | 1,951,870 | 4.6626 | 324 |

EdgeLite 相比原始 YOLO26n 减少约 22.12% 参数量，GFLOPs 减少约 19.37%。

## 目录说明

| 路径 | 说明 |
|---|---|
| `coffee3000/` | 数据集配置和说明 |
| `edgelite_experiment/` | EdgeLite 结构实验、配置和本地 Ultralytics 修改版 |
| `wiou_progloss_experiment/` | WIoU + ProgLoss 损失函数实验，原生 `loss.py` 保持可用 |
| `p5slim512_distill_experiment/` | P5Slim-512 第二阶段轻量化蒸馏实验 |
| `android_phone_deploy/` | Android 部署准备、NCNN 资源、测试工程和手机日志分析 |
| `URP_YOLO_progress_report_2026-05-27.md` | 当前阶段进度报告 |
| `yolocoffee-v3.0.bundle` | v3.0 打包归档 |

## 主要实验入口

### EdgeLite 训练

```powershell
python edgelite_experiment/train_coffee_edgelite.py
```

### WIoU + ProgLoss 训练

```powershell
python wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py
```

快速 smoke test：

```powershell
$env:WIOU_PROGLOSS_EPOCHS = "3"
python wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py
```

### P5Slim-512 蒸馏训练

```powershell
python p5slim512_distill_experiment/train_coffee_p5slim512_distill.py
```

快速 smoke test：

```powershell
$env:P5SLIM_DISTILL_EPOCHS = "3"
python p5slim512_distill_experiment/train_coffee_p5slim512_distill.py
```

## Android 部署

Android 相关文件位于：

```text
android_phone_deploy/
```

已经整理好的 Android 端模型资源位于：

```text
android_phone_deploy/android_app_assets/
```

包含 4 个 NCNN 模型：

| 模型 ID | 说明 | NCNN bin 大小 |
|---|---|---:|
| `01_yolo26n_base` | 原始 YOLO26n | 9.16 MB |
| `02_yolo26n_edgelite` | EdgeLite | 7.04 MB |
| `03_yolo26n_edgelite_simam` | EdgeLite + SimAM | 7.04 MB |
| `04_yolo26n_edgelite_wiou_progloss` | EdgeLite + WIoU/ProgLoss | 7.04 MB |

一致性验证结果显示 ONNX 和 NCNN 均通过与 PyTorch 输出的对比，其中当前推荐模型 `YOLO26n EdgeLite + WIoU/ProgLoss` 的 NCNN 对比结果为 PASS。

## 版本使用建议

- 复现实验结构和损失函数：使用 `v2.0`
- 查看第二阶段轻量化：使用 `v3.0`
- 查看当前完整上传内容和 Android 部署文件夹：使用 `v3.1`

## 说明

本项目保留了 Ultralytics 原始工程结构，但核心实验代码集中放在独立实验文件夹中，便于比较不同版本和回退：

- 原生损失函数仍可用于对照实验。
- WIoU/ProgLoss 只在对应实验入口中启用。
- Android 端只需要导出的模型文件，不需要实现训练阶段损失函数。

上游项目：Ultralytics YOLO。原始许可证见 `LICENSE`。
