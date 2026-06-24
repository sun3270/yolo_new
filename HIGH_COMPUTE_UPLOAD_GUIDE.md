# 高算平台上传与代码结构说明

## 当前结论

- 主训练线已经统一到 `coffee_self_sum`、`epochs=300`、`imgsz=960`、`batch=64`、`workers=8`。
- `coffee_self_sum/coffee_self_sum.yaml` 和 `coffee_self_sum/coffee_self.yaml` 不写 `path`，上传到 Linux/高算平台后会默认以 YAML 所在目录作为数据根目录。
- 已验证 `native`、`edgelite`、`edgelite_bibridge`、`ELTEB`、`ELTEB-Lite`、`WIoU+ProgLoss` 组合能够构建或触发对应功能。
- 不建议上传 `runs/`、`__pycache__/`、分析临时目录、重复数据目录。

## 必须上传

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `edgelite_experiment/` | 本地隔离 Ultralytics、EdgeLite、BiBridge 模块、训练入口 | 必须完整上传，尤其是 `local_ultralytics/` |
| `wiou_progloss_experiment/` | WIoU + ProgLoss 损失、类别统计、训练入口 | 如果跑新损失必须上传 |
| `elteb_experiment/` | ELTEB/ELTEB-Lite 配置、训练入口、预处理工具 | 如果跑 ELTEB 或 V4 ablation 必须上传 |
| `coffee_self_sum/` | 当前公平对比数据集 | 必须包含 `images/`、`labels/`、`coffee_self_sum.yaml` |
| `yolo26n.pt` | 预训练权重 | 不上传也能从 YAML 初始化，但不推荐 |

## 可选上传

| 路径 | 作用 | 什么时候上传 |
| --- | --- | --- |
| `pyproject.toml` | 原 Ultralytics 依赖说明 | 平台需要按项目安装依赖时上传 |
| `train_coffee_base.py` | 旧 coffee3000 原生 baseline 入口 | 只在复现旧 coffee3000 实验时上传 |
| `coffee3000/` | 旧 6 类数据集 | 只在复现旧实验时上传 |
| `coffee_self/`、`coffee_self_de/`、`coffee_self_merged/`、`coffee_self_minority_aug/` | 数据处理中间版本 | 只在重做数据清洗/增强时上传 |
| `runs/` | 本地训练结果 | 一般不要上传；只在需要带旧权重或图表时上传 |

## 不建议上传

| 路径 | 原因 |
| --- | --- |
| `__pycache__/` | Python 缓存，高算平台会自动生成 |
| `analysis_temp/`、`codex_analysis/`、`_analysis_zips/` | 分析临时文件，不参与训练 |
| `coffee_original_hash_report/`、`coffee_self_duplicate_report/` | 数据审计报告，不参与训练 |
| `android_phone_deploy/` | 和当前 YOLO 训练无关 |
| `p5slim512_distill_experiment/` | 当前只看到缓存文件，没有可用源码主线 |
| `lgmsf_lite_experiment/` | 当前工作树显示大量删除状态，不作为最新主线 |

## 文件夹说明

### `edgelite_experiment/`

当前最新架构主线。包含本地隔离版 `ultralytics`，训练脚本通过 `sys.path` 优先加载这里，不依赖根目录 `ultralytics/`。

- `configs/`
  - `yolo26n_original_copy.yaml`: 原生 YOLO26n 对照。
  - `yolo26n_edgelite.yaml`: 原 EdgeLite，单向 `P3 -> P5`。
  - `yolo26n_edgelite_simam.yaml`: 原 EdgeLite + SimAM。
  - `yolo26n_edgelite_bibridge.yaml`: 最新双向 EdgeLite。
  - `yolo26n_edgelite_simam_bibridge.yaml`: 双向 EdgeLite + SimAM。
- `local_ultralytics/`
  - 修改过的本地 Ultralytics 包。
  - 核心模块在 `ultralytics/nn/modules/block.py`。
  - YAML 解析在 `ultralytics/nn/tasks.py`。
- `train_coffee_edgelite.py`
  - 跑原 EdgeLite。
- `train_coffee_edgelite_simam.py`
  - 跑 EdgeLite + SimAM。
- `train_coffee_edgelite_bibridge.py`
  - 跑最新 EdgeLite-BiBridge；设置 `EDGE_BIBRIDGE_SIMAM=1` 可切 SimAM-BiBridge。
- `train_coffee_edgelite_simam_bibridge.py`
  - 直接跑最新 EdgeLite-SimAM-BiBridge，不需要设置 `EDGE_BIBRIDGE_SIMAM`。
- `check_edgelite_build.py`
  - 构建检查脚本；报告文件被占用时会打印结果，不会误判模型失败。
- `reports/`、`patches/`
  - 说明和补丁记录，不是训练必须项，但建议上传便于追踪。

### `wiou_progloss_experiment/`

新损失主线。它不改本地 Ultralytics 原生 loss 文件，而是在训练入口里 runtime patch `DetectionModel.init_criterion`。

- `train_coffee_edgelite_wiou_progloss.py`
  - 统一入口。
  - `WIOU_PROGLOSS_ARCH=native|edgelite|edgelite_simam|edgelite_bibridge|edgelite_simam_bibridge`。
  - 默认 `edgelite_bibridge`。
- `wiou_progloss_loss.py`
  - 自定义 `E2EWIoUProgLoss`、`v8DetectionWIoUProgLoss`、`BboxWIoUProgLoss`。
- `loss_extensions.py`
  - WIoU 和 ProgLoss 数学实现。
- `dataset_class_counts.py`
  - 自动统计训练集类别数，传给 ProgLoss。
- `loss_config.yaml`
  - WIoU/ProgLoss 超参数。

### `elteb_experiment/`

ELTEB/ELTEB-Lite 结构线。现在补齐了 `native + ELTEB`、`edgelite + ELTEB`、`edgelite_bibridge + ELTEB`。

- `configs/`
  - `yolo26n_native_elteb.yaml`
  - `yolo26n_native_elteb_lite.yaml`
  - `yolo26n_edgelite_elteb.yaml`
  - `yolo26n_edgelite_elteb_lite.yaml`
  - `yolo26n_edgelite_bibridge_elteb.yaml`
  - `yolo26n_edgelite_bibridge_elteb_lite.yaml`
- `train_coffee_v4_elteb.py`
  - ELTEB 专用入口。
  - `V4_ELTEB_ARCH=native|edgelite|edgelite_bibridge`。
  - `V4_ELTEB_VARIANT=elteb|elteb_lite`。
  - `V4_ELTEB_LOSS=native|wiou_progloss`。
- `train_coffee_v4_ablation.py`
  - V4 消融入口。
  - `V4_ARCH=native|edgelite|edgelite_bibridge`。
  - `V4_VARIANT=baseline|clahe|laplacian|elteb|elteb_lite`。
- `build_preprocessed_dataset.py`
  - 构建 CLAHE/Laplacian 数据副本，默认源数据已同步到 `coffee_self_sum`。
- `repair_dataset_images.py`、`report_duplicate_images.py`、`build_deduplicated_dataset.py`、`augment_minority_dataset.py`
  - 数据处理工具，不是训练必须项。

### `coffee_self_sum/`

当前建议上传并训练的数据集。检查结果：

- train: 2554 images / 2554 labels
- val: 304 images / 304 labels
- test: 320 images / 320 labels
- missing labels: 0
- empty labels: 0
- train class counts: `[1381, 446, 356, 209, 93, 320, 194, 165, 182]`

### `coffee3000/`

旧 6 类 Roboflow 数据集。当前最新脚本不再默认使用它。只在复现旧结果时上传。

### `runs/`

训练输出目录。高算平台新训练不需要上传。需要续训时，只上传对应 `runs/train/.../weights/best.pt` 或 `last.pt`。

### 根目录脚本

- `train_coffee_base.py`: 旧 coffee3000 baseline，仍是 `200/640/32`，不是最新主线。
- `train_coffee_lgmsf.py`: 旧 LGMSF-Lite 入口，依赖 `lgmsf_lite_experiment/`，当前不建议作为最新上传主线。
- `train_leaf_100.py`: leaf_100 小数据实验，和咖啡病害主线无关。

## 推荐高算平台运行命令

### 最新主线：EdgeLite-BiBridge

```bash
python edgelite_experiment/train_coffee_edgelite_bibridge.py
```

### 最新主线：EdgeLite-SimAM-BiBridge

```bash
python edgelite_experiment/train_coffee_edgelite_simam_bibridge.py
```

### 最新主线 + WIoU/ProgLoss

```bash
export WIOU_PROGLOSS_ARCH=edgelite_bibridge
python wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py
```

### EdgeLite-BiBridge + ELTEB

```bash
export V4_ELTEB_ARCH=edgelite_bibridge
export V4_ELTEB_VARIANT=elteb
export V4_ELTEB_LOSS=native
python elteb_experiment/train_coffee_v4_elteb.py
```

### EdgeLite-BiBridge + ELTEB + WIoU/ProgLoss

```bash
export V4_ELTEB_ARCH=edgelite_bibridge
export V4_ELTEB_VARIANT=elteb
export V4_ELTEB_LOSS=wiou_progloss
python elteb_experiment/train_coffee_v4_elteb.py
```

## 高算平台依赖提醒

建议环境至少包含：

- Python 3.10+
- PyTorch CUDA 版
- `opencv-python`
- `pyyaml`
- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`

训练脚本会优先使用 `edgelite_experiment/local_ultralytics`，所以不需要在平台上单独修改 site-packages 里的 Ultralytics。
