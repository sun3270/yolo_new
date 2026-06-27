# Coffee V4 项目交接与另一台电脑配置指南

生成日期：2026-06-27  
适用项目：`sun3270/yolo_new` / Coffee V4 recall 改造线  
目标：让同学在另一台 Windows 电脑上复现环境、检查数据集、生成派生数据集并继续训练。

## 1. 当前项目结论

当前任务不是优先压误检，而是解决漏检，提高 recall。主要问题来自数据集几何质量、边缘标注、复杂背景和多主体遮挡。

当前有效最好训练结果来自：

```text
runs/train/v4_edgelite_bibridge_wiou_copypaste_context_coffee_self_sum4
```

关键指标：

```text
最佳 mAP50-95：
epoch 158
Precision 0.865
Recall    0.771
mAP50     0.853
mAP50-95  0.771

最高 Recall：
epoch 159
Precision 0.814
Recall    0.809
mAP50     0.844
mAP50-95  0.760
```

已判定异常的实验：

```text
edgelite_bibridge_wiou_sahi_trainonly
```

原因：该方案只用切片图训练，切片后产生大量贴边或截断叶片框，和原图验证分布严重不一致，mAP50-95 只有约 0.083。后续不要直接训练这条线，除非先修复切片标签策略。

## 2. 已经实现的本地工具

核心目录：

```text
C:\Users\ssema\Desktop\coffee\yolo_new
```

新增或已使用的 V4 工具：

```text
coffee_v4_training/dataset_diagnostics.py
coffee_v4_training/build_recall_datasets.py
coffee_v4_training/inference_recall_sweep.py
coffee_v4_training/sam2_refine_labels.py
coffee_v4_training/csv_to_tensorboard.py
coffee_v4_training/train.py
coffee_v4_training/run_ablation_matrix.py
tests/test_coffee_v4_recall_tools.py
```

新增训练实验：

```text
edgelite_bibridge_wiou_sanitize_labels
edgelite_bibridge_wiou_sanitize_copypaste_context
edgelite_bibridge_wiou_sam_refined
```

保留但默认不推荐：

```text
edgelite_bibridge_wiou_sahi_trainonly
```

## 3. 数据集状态

原始数据集：

```text
C:\Users\ssema\Desktop\coffee\datasets\coffee_self_sum\coffee_self_sum.yaml
```

原始数据集检查发现：

```text
train 越界框：149
val   越界框：16
test  越界框：13
```

已生成 clean 基座数据集：

```text
C:\Users\ssema\Desktop\coffee\yolo_new\elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml
```

clean 基座结果：

```text
修正越界框：178
dropped boxes：0
diagnostics warning_count：0
```

已生成 clean + context Copy-Paste 数据集：

```text
C:\Users\ssema\Desktop\coffee\yolo_new\elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_copypaste_context\coffee_self_sum_sanitize_copypaste_context.yaml
```

clean + context Copy-Paste 结果：

```text
source_boxes：1331
scheduled_pastes：3338
accepted_pastes：1050
rejected_paste_attempts：93260
failed_scheduled_pastes：2288
generated_train_images：1050
训练集总图像：3604
训练集总实例：5831
diagnostics warning_count：0
```

已生成 SAM review 数据集：

```text
C:\Users\ssema\Desktop\coffee\yolo_new\elteb_experiment\preprocessed_data\coffee_self_sum_sam_refined\coffee_self_sum_sam_refined.yaml
```

SAM review 结果：

```text
objects_reviewed：4182
needs_review：2153
preview_images：100
review_csv：
C:\Users\ssema\Desktop\coffee\yolo_new\elteb_experiment\preprocessed_data\coffee_self_sum_sam_refined\review_candidates.csv
```

注意：这轮没有真正调用 SAM2 checkpoint，而是使用本地 leaf-mask fallback 生成复核候选。如果同学电脑要真正使用 SAM2，需要额外安装 SAM2 并传入 SAM2 config 和 checkpoint。

## 4. 方法依据

本项目采用两条线：

1. 数据集线：先清洗标签，再做增强。
2. 模型训练线：保留 V4 主线，再做 NWD 和强模型对照。

论文方法对应关系：

```text
SAM2：
用于离线辅助标注和可见区域 mask 复核，不放入最终部署链路。

Simple Copy-Paste：
用于增强低召回类和弱类样本。

SAHI：
适合小目标切片，但当前整叶检测任务里切片会制造大量截断叶片，因此暂不直接使用 trainonly。

NWD：
用于小框或边缘框定位敏感性消融。

YOLOv12 / YOLOv13 / RT-DETRv2 / YOLOv10：
作为后续强模型外部对照，不直接替换当前 V4 主线。
```

## 5. 另一台电脑目录要求

推荐统一目录：

```text
C:\Users\<用户名>\Desktop\coffee
  ├─ yolo_new
  └─ datasets
      └─ coffee_self_sum
          ├─ coffee_self_sum.yaml
          ├─ images
          └─ labels
```

最少必须复制：

```text
yolo_new 项目目录
datasets/coffee_self_sum 数据集目录
yolo_new/yolo26n.pt
```

派生数据集可以复制，也可以在同学电脑上重新生成。推荐重新生成，因为路径会随电脑用户名变化。

## 6. 环境安装

先安装：

```text
Anaconda 或 Miniconda
Git
NVIDIA Driver
```

检查显卡：

```powershell
nvidia-smi
```

创建环境：

```powershell
conda create -n coffee python=3.10 -y
conda activate coffee
python -m pip install --upgrade pip
```

安装 PyTorch GPU 版。若显卡驱动支持 CUDA 12.x，优先用：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

必须看到：

```text
True
显卡名称
```

安装项目依赖：

```powershell
cd C:\Users\<用户名>\Desktop\coffee\yolo_new
pip install -e .
pip install pytest tensorboard
```

如果遇到 OpenMP 报错：

```text
OMP: Error #15: Initializing libiomp5md.dll
```

在当前 PowerShell 里执行：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## 7. 项目验证

进入项目：

```powershell
conda activate coffee
cd C:\Users\<用户名>\Desktop\coffee\yolo_new
```

检查模型构建：

```powershell
python coffee_v4_training\check_builds.py --exps edgelite_bibridge_wiou,edgelite_bibridge_nwd
```

dry-run 检查训练入口：

```powershell
python coffee_v4_training\train.py `
  --exp edgelite_bibridge_wiou `
  --data C:\Users\<用户名>\Desktop\coffee\datasets\coffee_self_sum\coffee_self_sum.yaml `
  --weights C:\Users\<用户名>\Desktop\coffee\yolo_new\yolo26n.pt `
  --device 0 `
  --dry-run
```

运行测试：

```powershell
pytest tests\test_coffee_v4_recall_tools.py -q
```

期望看到：

```text
16 passed
```

## 8. 重新生成 clean 数据集

设置数据路径：

```powershell
$DATA="C:\Users\<用户名>\Desktop\coffee\datasets\coffee_self_sum\coffee_self_sum.yaml"
```

生成 clean 基座：

```powershell
python coffee_v4_training\build_recall_datasets.py `
  --variant sanitize_labels `
  --data $DATA `
  --output elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels `
  --overwrite
```

诊断 clean 数据：

```powershell
python coffee_v4_training\dataset_diagnostics.py `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml `
  --output runs\dataset_audit_sanitize `
  --splits train,val,test `
  --tail-classes 1,2,3,5,4,6,7,8
```

检查：

```text
runs/dataset_audit_sanitize/dataset_report.json
runs/dataset_audit_sanitize/hard_cases.csv
runs/dataset_audit_sanitize/paper_requirements_report.md
```

验收标准：

```text
label_quality.warning_count = 0
```

## 9. 生成 clean + context Copy-Paste 数据集

```powershell
python coffee_v4_training\build_recall_datasets.py `
  --variant copypaste_context `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml `
  --output elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_copypaste_context `
  --copy-class-weights 1:2,2:3,3:2,5:3 `
  --max-paste-iou 0.10 `
  --overwrite
```

诊断增强数据：

```powershell
python coffee_v4_training\dataset_diagnostics.py `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_copypaste_context\coffee_self_sum_sanitize_copypaste_context.yaml `
  --output runs\dataset_audit_sanitize_copypaste_context `
  --splits train,val,test `
  --tail-classes 1,2,3,5,4,6,7,8
```

验收标准：

```text
label_quality.warning_count = 0
recall_dataset_manifest.json 中 accepted_pastes > 0
```

## 10. SAM2 离线复核

不安装 SAM2 时可以先跑 fallback：

```powershell
python coffee_v4_training\sam2_refine_labels.py `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml `
  --output elteb_experiment\preprocessed_data\coffee_self_sum_sam_refined `
  --mode review `
  --iou-gate 0.55 `
  --overwrite
```

输出：

```text
elteb_experiment/preprocessed_data/coffee_self_sum_sam_refined/review_candidates.csv
elteb_experiment/preprocessed_data/coffee_self_sum_sam_refined/review_previews/
elteb_experiment/preprocessed_data/coffee_self_sum_sam_refined/sam2_refine_manifest.json
```

真正使用 SAM2 时，需要安装 SAM2，并加入：

```powershell
--sam2-config <SAM2配置文件路径> `
--sam2-checkpoint <SAM2权重路径> `
--require-sam2
```

注意：

```text
SAM2 脚本只做 review，不自动覆盖标签。
必须人工看 review_candidates.csv 和 review_previews，再决定是否训练 sam_refined 数据集。
```

## 11. 推荐训练命令

优先训练 clean + context Copy-Paste：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

python coffee_v4_training\train.py `
  --exp edgelite_bibridge_wiou_sanitize_copypaste_context `
  --data C:\Users\<用户名>\Desktop\coffee\datasets\coffee_self_sum\coffee_self_sum.yaml `
  --weights C:\Users\<用户名>\Desktop\coffee\yolo_new\yolo26n.pt `
  --device 0 `
  --epochs 300 `
  --imgsz 960 `
  --batch 16 `
  --workers 4 `
  --cache disk
```

如果显存不够：

```powershell
--batch 8
```

如果显存仍然不够：

```powershell
--imgsz 768
```

clean baseline：

```powershell
python coffee_v4_training\train.py `
  --exp edgelite_bibridge_wiou_sanitize_labels `
  --data C:\Users\<用户名>\Desktop\coffee\datasets\coffee_self_sum\coffee_self_sum.yaml `
  --weights C:\Users\<用户名>\Desktop\coffee\yolo_new\yolo26n.pt `
  --device 0 `
  --epochs 300 `
  --imgsz 960 `
  --batch 16 `
  --workers 4 `
  --cache disk
```

NWD 消融：

```powershell
python coffee_v4_training\train.py `
  --exp edgelite_bibridge_nwd `
  --data elteb_experiment\preprocessed_data\coffee_self_sum_sanitize_labels\coffee_self_sum_sanitize_labels.yaml `
  --weights C:\Users\<用户名>\Desktop\coffee\yolo_new\yolo26n.pt `
  --device 0 `
  --epochs 300 `
  --imgsz 960 `
  --batch 16 `
  --workers 4 `
  --cache disk
```

80 epoch 快筛：

```powershell
python coffee_v4_training\run_ablation_matrix.py `
  --group recall `
  --data C:\Users\<用户名>\Desktop\coffee\datasets\coffee_self_sum\coffee_self_sum.yaml `
  --weights C:\Users\<用户名>\Desktop\coffee\yolo_new\yolo26n.pt `
  --device 0 `
  --epochs 80 `
  --imgsz 960 `
  --batch 16 `
  --workers 4 `
  --cache disk `
  --continue-on-error
```

## 12. TensorBoard 实时监控

另开一个 PowerShell：

```powershell
conda activate coffee
cd C:\Users\<用户名>\Desktop\coffee\yolo_new
tensorboard --logdir runs\train --host 127.0.0.1 --port 6006
```

浏览器打开：

```text
http://127.0.0.1:6006
```

如果 TensorBoard 没看到新 run：

```text
1. 确认训练真的写到了 yolo_new/runs/train
2. 确认 TensorBoard 的 --logdir 是 runs/train，不是某个单独旧目录
3. 刷新浏览器
4. 重启 TensorBoard
```

## 13. 训练后检查

每个 run 看：

```text
runs/train/<run_name>/results.csv
runs/train/<run_name>/results.png
runs/train/<run_name>/confusion_matrix_normalized.png
runs/train/<run_name>/BoxPR_curve.png
runs/train/<run_name>/val_batch*_pred.jpg
runs/train/<run_name>/weights/best.pt
```

评价标准：

```text
优先比较当前最好 run：
v4_edgelite_bibridge_wiou_copypaste_context_coffee_self_sum4

目标：
Recall > 0.812
或 mAP50-95 > 0.771 且 Recall 不下降
Precision >= 0.80
BLS_AB 和 CR_AB 不靠误检堆起来
hard set missed count 下降
```

## 14. 不要做的事

```text
不要覆盖原始 datasets/coffee_self_sum。
不要直接训练 sahi_trainonly。
不要只看最后一个 epoch。
不要只看 mAP50，要同时看 Recall、mAP50-95、混淆矩阵和预测图。
不要把 SAM2 放进最终部署链路；SAM2 只是离线标注辅助。
```

## 15. 推荐下一步顺序

第一步：

```text
在同学电脑重新生成 sanitize_labels。
```

第二步：

```text
跑 dataset_diagnostics，确认 warning_count = 0。
```

第三步：

```text
生成 sanitize_copypaste_context。
```

第四步：

```text
训练 edgelite_bibridge_wiou_sanitize_copypaste_context。
```

第五步：

```text
如果结果不够，再看 SAM review 的 CSV 和预览图，人工修正后训练 sam_refined。
```

第六步：

```text
最后再做 NWD、YOLOv12/YOLOv13/RT-DETRv2/YOLOv10 对照。
```

