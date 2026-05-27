# URP YOLO 咖啡叶病害检测阶段进展汇报

汇报日期：2026-05-27  
项目方向：基于 YOLO26n 的咖啡叶病害检测、轻量化和移动端部署

## 1. 当前一句话结论

目前项目已经完成了从原始 YOLO26n 基线训练，到 EdgeLite 结构轻量化、WIoU + ProgLoss 损失函数优化、ONNX/NCNN 导出和 Android 端初步验证的完整闭环。当前综合效果最好的版本是：

**YOLO26n-EdgeLite + WIoU/ProgLoss**

该版本在保持轻量化结构的基础上，验证集最佳结果达到：

| 指标 | 结果 |
|---|---:|
| 参数量 | 1.952M |
| GFLOPs | 4.663 |
| mAP50 | 97.68% |
| mAP50-95 | 92.18% |
| 最佳 epoch | 272 |

相比原始 YOLO26n，EdgeLite 路线参数量减少约 22.12%，GFLOPs 减少约 19.37%；在加入 WIoU + ProgLoss 后，mAP50-95 从基线的 91.73% 提升到 92.18%。

## 2. 数据集与任务设置

使用数据集为 `coffee3000`，任务为 6 类咖啡叶病害目标检测。

| 数据划分 | 图片数量 |
|---|---:|
| train | 2658 |
| val | 744 |
| test | 389 |

类别如下：

| 类别编号 | 英文类别名 | 汇报用中文名 | 训练集标注数 |
|---:|---|---|---:|
| 0 | algal_spot | 藻斑病 | 501 |
| 1 | brown_eye_spot | 褐眼斑病 | 606 |
| 2 | healthy | 健康叶片 | 332 |
| 3 | miner | 潜叶虫害 | 618 |
| 4 | phoma | Phoma 叶斑病 | 709 |
| 5 | powdery_mildew | 白粉病 | 163 |

可以看到 `powdery_mildew` 是明显少样本类别，训练集中只有 163 个标注，和最多的 `phoma` 相差约 4.35 倍。因此后续的损失函数改进重点考虑了长尾类别问题。

## 3. 已做尝试总览

| 阶段 | 尝试内容 | 目的 | 当前结论 |
|---|---|---|---|
| 1 | YOLO26n baseline | 建立原始模型基线 | mAP50-95 为 91.73%，作为对照组 |
| 2 | LGMSF-Lite | 尝试 P3 纹理和 P5 语义融合 | 结构能跑通，但参数和计算量偏高，不作为主路线 |
| 3 | EdgeLite | 面向移动端重新设计轻量结构 | 参数和 GFLOPs 明显下降，是后续主路线 |
| 4 | EdgeLite + SimAM | 验证轻量注意力是否提升复杂背景鲁棒性 | mAP50-95 有恢复，但仍低于最终损失优化路线 |
| 5 | EdgeLite + WIoU/ProgLoss | 针对框回归和长尾类别做损失优化 | 当前最佳结果，mAP50-95 达到 92.18% |
| 6 | P5Slim512 蒸馏 | 第二阶段进一步压缩模型 | 目前完成设计和 smoke run，尚未完成完整训练 |
| 7 | Android 端部署验证 | 导出 ONNX/NCNN 并做手机端测试 | 导出和一致性验证通过，手机日志已能分析 |

## 4. 各阶段详细记录

### 4.1 基线模型 YOLO26n

目的：先用原始 YOLO26n 在 coffee3000 上训练，得到后续结构改进和损失函数改进的对照结果。

模型与结构参数：

| 项目 | 值 |
|---|---:|
| 模型 | YOLO26n |
| 参数量 | 2,506,140 |
| GFLOPs | 5.7825 |
| 检测类别数 | 6 |

主要训练参数：

| 参数 | 值 |
|---|---:|
| 输入尺寸 | 640 |
| 计划 epochs | 200 |
| 实际日志 epochs | 156 |
| batch | 128 |
| optimizer | auto |
| lr0 | 0.012 |
| lrf | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 4.0 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| patience | 30 |
| close_mosaic | 10 |
| AMP | true |
| cache | ram |

结果：

| 指标 | 最佳值 |
|---|---:|
| 最佳 epoch | 130 |
| Precision | 96.22% |
| Recall | 94.39% |
| mAP50 | 96.35% |
| mAP50-95 | 91.73% |

阶段结论：基线精度较高，但模型参数量和计算量对移动端部署仍有压缩空间。

### 4.2 LGMSF-Lite 结构尝试

目的：引入 P3 纹理分支和 P5 语义分支，让模型同时利用细粒度病斑纹理和全局语义信息。

主要结构改动：

| 改动 | 说明 |
|---|---|
| LDSConv | 替换部分下采样卷积，降低计算开销 |
| TextureBranch | 从 P3 提取高分辨率纹理信息 |
| SemanticBranch | 从 P5 提取语义信息 |
| FastNormFuse2 | 使用可学习归一化权重融合两路特征 |
| SimAM | 加入轻量注意力，用于复杂背景抑制 |
| LGMSFBridge | 将 P3 纹理和 P5 语义融合到检测头 |

模型结构参数：

| 项目 | 值 |
|---|---:|
| 参数量 | 2,468,190 |
| GFLOPs | 7.3143 |
| 层数 | 317 |

主要训练参数：

| 参数 | 值 |
|---|---:|
| 输入尺寸 | 640 |
| 计划 epochs | 200 |
| 实际日志 epochs | 5 |
| batch | auto batch，日志中为 -1 |
| optimizer | auto |
| lr0 | 0.012 |
| lrf | 0.01 |
| warmup_epochs | 4.0 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |

短训练结果：

| 指标 | 最佳值 |
|---|---:|
| 最佳 epoch | 4 |
| Precision | 31.31% |
| Recall | 47.03% |
| mAP50 | 34.86% |
| mAP50-95 | 19.89% |

阶段结论：LGMSF-Lite 的结构能构建并完成前向推理，但参数量接近原始模型，GFLOPs 反而更高。该路线更多作为结构探索，没有作为最终主路线。

### 4.3 YOLO26n-EdgeLite 结构轻量化

目的：在保留 P3 纹理和 P5 语义融合思想的基础上，重新设计更适合移动端的轻量化结构。EdgeLite 的重点不是把 YOLO26n 全部推倒重写，而是在原模型上进行“关键层轻量替换 + P3/P5 融合增强”。

#### 4.3.1 EdgeLite 用到的技术模块

| 英文模块名 | 中文名 | 作用 |
|---|---|---|
| LDSConv | 可学习深度可分离下采样卷积 | 替换计算量较大的 stride=2 普通下采样卷积，降低参数量和 GFLOPs |
| DWSeparableConv | 深度可分离卷积 | 将普通卷积分解为 depthwise 空间卷积和 pointwise 1x1 通道卷积 |
| TextureStreamP3 | P3 纹理细节分支 | 从高分辨率 P3 特征中提取病斑边缘、斑点、纹理等细节 |
| SemanticStreamP5 | P5 语义上下文分支 | 从低分辨率 P5 特征中提取整片叶子、光照、背景等全局语义 |
| FastNormFuse2 | 快速归一化加权融合模块 | 用可学习权重自动决定 P3 纹理和 P5 语义各占多少比例 |
| EdgeLGMSFBridge | 边缘轻量多尺度融合桥 | 将 P3 纹理分支和 P5 语义分支对齐并融合，输出增强后的 P5 特征 |
| SimAM | 无参数轻量注意力模块 | 用于测试是否能抑制复杂背景干扰；主版本关闭，消融版本开启 |

#### 4.3.2 LDSConv 的替换方式

LDSConv 的全称可汇报为“可学习深度可分离下采样卷积”。它的基本结构是：

```text
普通下采样 Conv
    ↓
Depthwise 3x3 卷积：每个通道单独提取空间特征
Pointwise 1x1 卷积：进行通道融合和通道数调整
```

这样做的原因是，普通 3x3 卷积会同时做空间卷积和通道混合，计算量较高；深度可分离卷积把这两步拆开，通常能明显降低参数量和计算量。

EdgeLite 没有替换所有卷积，而是只替换计算量较大的关键下采样层：

| 位置 | 原作用 | EdgeLite 替换后 | 目的 |
|---|---|---|---|
| backbone 第 3 层 | P2/4 -> P3/8 下采样 | LDSConv [256, 3, 2] | 降低早期下采样计算量，保留 P3 细节 |
| backbone 第 5 层 | P3/8 -> P4/16 下采样 | LDSConv [512, 3, 2] | 降低中层特征提取成本 |
| backbone 第 7 层 | P4/16 -> P5/32 下采样 | LDSConv [1024, 3, 2] | 降低深层语义特征计算量 |
| head 第 18 层 | P3 检测分支 -> P4 检测分支 | LDSConv [256, 3, 2] | 轻量化检测头中的下采样路径 |
| head 第 21 层 | P4 检测分支 -> P5 检测分支 | LDSConv [512, 3, 2] | 轻量化最终大目标检测路径 |

因此，EdgeLite 的第一层改动可以总结为：

> 只对关键 stride=2 下采样卷积进行轻量替换，不破坏 YOLO 原有的三尺度检测结构。

#### 4.3.3 P3/P5 特征融合方式

EdgeLite 的第二个核心是新增 `EdgeLGMSFBridge`，即“边缘轻量多尺度融合桥”。它融合的是：

| 特征层 | 来源层号 | 中文解释 | 主要价值 |
|---|---:|---|---|
| P3 | 第 4 层 | 高分辨率纹理特征 | 保留病斑边缘、小斑点、局部纹理 |
| P5 | 第 10 层 | 低分辨率语义特征 | 提供整片叶子、光照背景、病害整体语义 |

配置中对应这一行：

```yaml
- [[4, 10], 1, EdgeLGMSFBridge, [256, False, 8]]
```

含义是：把第 4 层 P3 和第 10 层 P5 同时输入到 EdgeLGMSFBridge，输出 256 通道的增强 P5 特征。其中 `False` 表示主版本不启用 SimAM，`8` 是中间通道压缩比例。

具体融合流程如下：

| 步骤 | 操作 | 说明 |
|---:|---|---|
| 1 | P3 输入 TextureStreamP3 | 先用 1x1 卷积压缩通道，再用两个深度可分离 3x3 卷积提取纹理细节 |
| 2 | P3 纹理分支连续两次下采样 | P3 是 P3/8，需要通过两次 stride=2 的 LDSConv 对齐到 P5/32 |
| 3 | P5 输入 SemanticStreamP5 | 用 1x1 卷积压缩通道，再用深度可分离 5x5 卷积增强语义上下文 |
| 4 | 对齐空间尺寸 | 如果 P3 下采样后的尺寸和 P5 不完全一致，用 nearest 插值对齐 |
| 5 | FastNormFuse2 加权融合 | 用两个可学习权重融合 P3 纹理和 P5 语义 |
| 6 | 可选 SimAM | 主版本关闭，SimAM 消融版本开启 |
| 7 | 1x1 输出卷积 | 输出增强后的 256 通道 P5 特征 |

FastNormFuse2 的融合公式可以讲成：

```text
增强特征 = w1 * P3纹理特征 + w2 * P5语义特征
```

其中 `w1` 和 `w2` 是训练过程中自动学习的，并且经过 ReLU 和归一化处理，保证融合权重稳定。这样不是简单拼接，而是让模型自己学习“当前任务更依赖纹理还是语义”。

#### 4.3.4 融合后的特征送到哪里

EdgeLGMSFBridge 输出的是第 11 层增强 P5 特征。后续在检测头中，第 22 层把当前 P5 路径和第 11 层增强 P5 做拼接：

```yaml
- [[-1, 11], 1, Concat, [1]]
```

最终检测仍然保持 YOLO 的三尺度检测：

```text
Detect(P3, P4, P5) = Detect([17, 20, 23])
```

也就是说，EdgeLite 没有改变 YOLO 的最终检测范式，仍然检测小目标、中目标和大目标；它只是让最终 P5 分支融合了来自 P3 的细节信息和 P5 的语义信息。

#### 4.3.5 为什么这样设计

咖啡叶病害图像中既有细小病斑、斑点边缘，也有复杂背景、叶片整体颜色和光照变化。因此：

| 问题 | EdgeLite 对应设计 |
|---|---|
| 小病斑、边缘纹理容易丢失 | 用 P3 纹理分支保留高分辨率细节 |
| 背景复杂、光照变化明显 | 用 P5 语义分支提供全局上下文 |
| 直接增加复杂融合会变慢 | 用轻量 EdgeLGMSFBridge 和 LDSConv 控制计算量 |
| 不同图片对纹理/语义依赖不同 | 用 FastNormFuse2 学习融合权重 |
| 需要适配手机端 | 只改训练和模型结构，不引入手机端难实现的额外后处理 |

#### 4.3.6 汇报时可以这样讲

EdgeLite 的改进主要有两点。第一，我将 YOLO26n 中几个关键 stride=2 下采样卷积替换为 LDSConv，也就是可学习深度可分离下采样卷积，用 depthwise 3x3 加 pointwise 1x1 降低下采样阶段的参数量和计算量。第二，我加入了 EdgeLGMSFBridge，也就是边缘轻量多尺度融合桥，把 P3 的高分辨率纹理信息和 P5 的深层语义信息融合。P3 分支负责保留咖啡叶病斑的边缘、斑点等细节，经过两次轻量下采样对齐到 P5 尺寸；P5 分支负责全局叶片结构和光照背景。两路特征通过 FastNormFuse2 做可学习加权融合，再送入最终 P5 检测头。这样既保留了小病斑细节，又利用了深层语义，同时模型参数量和 GFLOPs 都下降。

模型结构参数：

| 模型 | 参数量 | GFLOPs | 层数 |
|---|---:|---:|---:|
| 原始 YOLO26n | 2,506,140 | 5.7825 | 260 |
| EdgeLite | 1,951,870 | 4.6626 | 324 |
| EdgeLite + SimAM | 1,951,870 | 4.6626 | 324 |

轻量化幅度：

| 对比 | 参数量变化 | GFLOPs 变化 |
|---|---:|---:|
| EdgeLite vs 原始 YOLO26n | 约 -22.12% | 约 -19.37% |
| EdgeLite vs LGMSF-Lite | 约 -20.92% | 约 -36.25% |

主要训练参数：

| 参数 | 值 |
|---|---:|
| 输入尺寸 | 640 |
| 计划 epochs | 300 |
| batch | 128 |
| optimizer | auto |
| 预训练权重 | yolo26n.pt |
| lr0 | 0.012 |
| lrf | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 4.0 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| patience | 30 |
| close_mosaic | 10 |
| AMP | true |
| cache | ram |

EdgeLite 结果：

| 指标 | 最佳值 |
|---|---:|
| 实际日志 epochs | 170 |
| 最佳 epoch | 140 |
| Precision | 96.89% |
| Recall | 93.50% |
| mAP50 | 97.86% |
| mAP50-95 | 90.74% |

阶段结论：EdgeLite 明显降低了参数量和计算量，mAP50 反而提高到 97.86%，但 mAP50-95 下降到 90.74%，说明在更严格 IoU 阈值下定位质量还有改进空间。

### 4.4 EdgeLite + SimAM 注意力消融

目的：测试 SimAM 是否能改善复杂背景下的特征表达。

结构差异：

| 模型 | SimAM |
|---|---|
| EdgeLite | 关闭 |
| EdgeLite + SimAM | 开启 |

主要训练参数与 EdgeLite 保持一致：

| 参数 | 值 |
|---|---:|
| 输入尺寸 | 640 |
| 计划 epochs | 300 |
| batch | 128 |
| lr0 | 0.012 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| patience | 30 |

结果：

| 指标 | 最佳值 |
|---|---:|
| 实际日志 epochs | 209 |
| 最佳 epoch | 179 |
| Precision | 97.85% |
| Recall | 92.88% |
| mAP50 | 97.87% |
| mAP50-95 | 91.46% |

阶段结论：SimAM 对 mAP50 基本持平，对 mAP50-95 有一定恢复，从 90.74% 提升到 91.46%。但是它没有超过后续 WIoU + ProgLoss 的结果，所以最终没有作为最优版本。

### 4.5 EdgeLite + WIoU/ProgLoss 损失函数优化

目的：在不继续改结构的情况下，重点提升定位质量和少样本类别学习效果。

核心思路：

| 模块 | 作用 |
|---|---|
| WIoU | 使用动态非单调聚焦机制，避免过度关注极难样本或极易样本，强化中等质量框的回归 |
| ProgLoss | 按训练进度逐步启用 WIoU 和长尾类别权重，避免训练早期不稳定 |
| 正样本类别重权 | 只增强正样本类别损失，避免 rare class 的负样本损失被错误放大 |

WIoU 参数：

| 参数 | 值 |
|---|---:|
| wiou_alpha | 1.7 |
| wiou_delta | 2.7 |
| wiou_momentum | 0.0001 |
| wiou_focus_min / max | 0.5 / 3.0 |
| wiou_use_distance_gain | true |
| wiou_distance_gain_max | 1.8 |
| wiou_fallback_base | raw_iou |

ProgLoss 参数：

| 参数 | 值 |
|---|---:|
| warmup_ratio | 0.10 |
| ramp_end_ratio | 0.60 |
| tail_power | 0.5 |
| tail_lambda_max | 0.8 |
| tail_weight_min / max | 0.75 / 1.8 |
| class_counts | [501, 606, 332, 618, 709, 163] |

主要训练参数：

| 参数 | 值 |
|---|---:|
| 模型结构 | EdgeLite，SimAM 关闭 |
| 输入尺寸 | 640 |
| 计划 epochs | 300 |
| 实际日志 epochs | 300 |
| batch | 128 |
| optimizer | auto |
| 预训练权重 | yolo26n.pt |
| lr0 | 0.012 |
| lrf | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 4.0 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| patience | 30 |

结果：

| 指标 | 最佳值 |
|---|---:|
| 最佳 epoch | 272 |
| Precision | 95.60% |
| Recall | 94.18% |
| mAP50 | 97.68% |
| mAP50-95 | 92.18% |

与前面路线对比：

| 模型 | mAP50 | mAP50-95 | 说明 |
|---|---:|---:|---|
| YOLO26n baseline | 96.35% | 91.73% | 原始基线 |
| EdgeLite | 97.86% | 90.74% | 更轻，但高 IoU 定位指标下降 |
| EdgeLite + SimAM | 97.87% | 91.46% | 有恢复 |
| EdgeLite + WIoU/ProgLoss | 97.68% | 92.18% | 当前综合最优 |

阶段结论：WIoU + ProgLoss 是目前最有效的改进。它没有增加推理端结构复杂度，因为损失函数只在训练阶段生效；导出到手机端后不需要额外实现 WIoU 或 ProgLoss。

### 4.6 P5Slim512 第二阶段轻量化与蒸馏

目的：在 EdgeLite 已经轻量化的基础上，进一步压缩最终 P5 检测分支，使模型更适合手机端部署。

结构改动：

| 位置 | 原始 EdgeLite | P5Slim512 |
|---|---|---|
| 最终 P5 检测分支 C3k2 | 1024 通道 | 512 通道 |

预估模型规模：

| 模型 | 参数量 |
|---|---:|
| 当前 EdgeLite train graph | 1.95M |
| P5Slim512 train graph | 1.57M |
| P5Slim512 deploy effective graph | 1.48M |

蒸馏设置：

| 参数 | 值 |
|---|---:|
| Teacher | EdgeLite + WIoU/ProgLoss 最佳权重 |
| Student | yolo26n_edgelite_p5slim512.yaml |
| distill_temperature | 2.0 |
| distill_score_weight | 0.35 |
| distill_box_weight | 0.20 |
| distill_feature_weight | 0.08 |
| distill_conf_threshold | 0.20 |
| distill_topk | 600 |
| distill_feature_levels | [0, 1] |

当前 smoke run 参数：

| 参数 | 值 |
|---|---:|
| epochs | 1 |
| batch | 2 |
| imgsz | 640 |
| lr0 | 0.008 |
| workers | 0 |
| 预训练权重 | best_edgelite_wiou_progloss.pt |

smoke run 结果：

| 指标 | 值 |
|---|---:|
| epoch | 1 |
| Precision | 39.57% |
| Recall | 50.45% |
| mAP50 | 44.12% |
| mAP50-95 | 30.95% |

阶段结论：P5Slim512 目前已经完成结构设计、蒸馏损失代码和 1 epoch smoke run，说明训练流程可以跑通。但这个结果不能和完整训练模型直接比较，下一步需要做 300 epoch 完整训练。

## 5. Android 端部署与验证

目前已经完成 4 个模型的导出和初步移动端验证：

| 模型 | PyTorch 权重 | ONNX | NCNN |
|---|---:|---:|---:|
| YOLO26n base | 5.15 MB | 9.31 MB | 9.16 MB bin |
| YOLO26n EdgeLite | 4.15 MB | 7.21 MB | 7.04 MB bin |
| YOLO26n EdgeLite + SimAM | 4.15 MB | 7.21 MB | 7.04 MB bin |
| YOLO26n EdgeLite + WIoU/ProgLoss | 4.16 MB | 7.21 MB | 7.04 MB bin |

导出一致性验证：

| 模型 | ONNX | ONNX min IoU | NCNN | NCNN min IoU | NCNN max box diff |
|---|---|---:|---|---:|---:|
| YOLO26n base | PASS | 0.999901 | PASS | 0.977280 | 5.897 px |
| YOLO26n EdgeLite | PASS | 0.999849 | PASS | 0.965875 | 8.632 px |
| YOLO26n EdgeLite + SimAM | PASS | 0.999905 | PASS | 0.974386 | 7.892 px |
| YOLO26n EdgeLite + WIoU/ProgLoss | PASS | 0.999938 | PASS | 0.976690 | 7.117 px |

Android 测试 App 进展：

| 项目 | 状态 |
|---|---|
| 模型切换 | 已完成，支持 4 个模型 |
| 输入尺寸 | 640 x 640 |
| Runtime | ONNX Runtime Android |
| 置信度阈值 | 0.25 |
| NMS IoU 阈值 | 0.70 |
| 日志导出 | 已完成，包含 runs.csv、detections.csv、events.jsonl |

手机日志分析：

| 项目 | 值 |
|---|---:|
| 测试设备 | Xiaomi 23127PN0CC |
| 总 runs | 34 |
| 总 detections | 35 |
| 测试模型数 | 4 |
| 测试图片数 | 6 |

在同一张 `phoma` 病害图像上的对比：

| 模型 | Avg inference | Median inference | Avg total | 结果 | Avg confidence |
|---|---:|---:|---:|---|---:|
| YOLO26n base | 64.89 ms | 64.20 ms | 76.52 ms | phoma | 0.9676 |
| YOLO26n EdgeLite | 60.52 ms | 58.48 ms | 71.35 ms | phoma | 0.9608 |
| YOLO26n EdgeLite + SimAM | 61.30 ms | 59.50 ms | 72.57 ms | phoma | 0.9544 |
| YOLO26n EdgeLite + WIoU/ProgLoss | 64.61 ms | 60.29 ms | 75.67 ms | phoma | 0.9695 |

阶段结论：四个模型在该病害样本上都识别为 `phoma`。EdgeLite 的平均推理速度最快，EdgeLite + WIoU/ProgLoss 的平均置信度最高。但目前手机端样本量还较小，需要后续做更公平的多图多轮测试。

## 6. 当前项目完成程度

已经完成：

| 内容 | 状态 |
|---|---|
| coffee3000 数据集接入 | 已完成 |
| YOLO26n 基线训练 | 已完成 |
| LGMSF-Lite 结构探索 | 已完成初步验证 |
| EdgeLite 结构轻量化 | 已完成 |
| EdgeLite 与 SimAM 消融 | 已完成 |
| WIoU + ProgLoss 损失函数设计与训练 | 已完成，并取得当前最佳结果 |
| ONNX/NCNN 导出 | 已完成 |
| 导出一致性验证 | 已完成 |
| Android 测试 App | 已完成初版 |
| 手机日志采集和分析 | 已完成初步分析 |
| P5Slim512 二次轻量化设计 | 已完成 |
| P5Slim512 蒸馏完整训练 | 未完成，仅完成 smoke run |
| P5Slim512 Android 导出和测试 | 未完成 |

当前最适合汇报的成果是：

1. 已经构建了咖啡叶病害检测的完整训练和验证流程。
2. EdgeLite 结构在参数量和计算量上明显优于原始 YOLO26n。
3. WIoU + ProgLoss 进一步提升了 mAP50-95，是当前最佳版本。
4. 模型已经完成 ONNX/NCNN 导出，并通过 PyTorch、ONNX、NCNN 一致性验证。
5. Android 端已经能切换模型、运行检测和导出日志，说明部署链路基本打通。

## 7. 当前最优模型建议

现阶段建议把以下模型作为主模型汇报：

`YOLO26n-EdgeLite + WIoU/ProgLoss`

推荐原因：

| 维度 | 表现 |
|---|---|
| 精度 | mAP50-95 最高，为 92.18% |
| 轻量化 | 参数量 1.952M，比原始 YOLO26n 减少约 22.12% |
| 训练端改进 | WIoU + ProgLoss 只影响训练，不增加手机端推理复杂度 |
| 部署可行性 | ONNX/NCNN 导出通过，一致性验证通过 |
| 移动端验证 | 已有 Android 测试 App 和手机日志 |

## 8. 目前存在的问题

1. P5Slim512 只有 1 epoch smoke run，不能作为最终模型结论。
2. 手机端测试样本数量还少，目前只完成初步验证，不能充分代表全部病害类别。
3. 少样本类别 `powdery_mildew` 需要进一步看单类别 recall，而当前整理出的日志主要是整体 mAP。
4. Android 端目前更偏工程验证，后续还需要固定测试集、重复运行次数和统计方式。
5. P5Slim512 还没有完成导出、ONNX/NCNN 一致性验证和手机端替换测试。

## 9. 下一步计划

| 优先级 | 计划 | 目标 |
|---|---|---|
| 1 | 完成 P5Slim512 300 epoch 蒸馏训练 | 判断二次轻量化后是否能接近 EdgeLite + WIoU/ProgLoss 精度 |
| 2 | 做 per-class 指标统计 | 重点检查 `powdery_mildew` 等少样本类别 recall |
| 3 | 导出 P5Slim512 到 ONNX/NCNN | 验证进一步压缩后的部署可行性 |
| 4 | Android 端替换 P5Slim512 | 比较手机端速度、置信度和检测稳定性 |
| 5 | 扩大手机测试集 | 每个模型对相同图片集重复 5 到 10 次，统计 median inference |
| 6 | 准备消融表 | 对比 baseline、EdgeLite、SimAM、WIoU/ProgLoss、P5Slim512 |

## 10. 汇报时可以这样讲

本阶段我围绕咖啡叶病害检测做了三方面工作。第一是建立基线，用原始 YOLO26n 在 coffee3000 数据集上训练，得到 mAP50-95 91.73% 的对照结果。第二是做结构轻量化，我先尝试了 LGMSF-Lite，发现计算量偏高；随后改为 EdgeLite 结构，通过 LDSConv 和轻量 P3/P5 融合桥，将参数量从 2.506M 降到 1.952M，GFLOPs 从 5.783 降到 4.663。第三是做训练损失优化，针对数据集长尾问题加入 WIoU 和 ProgLoss，最终在不增加推理端复杂度的情况下，把 mAP50-95 提升到 92.18%。

目前模型已经完成 ONNX 和 NCNN 导出，并通过 PyTorch、ONNX、NCNN 的一致性验证。我还做了 Android 测试 App，可以在手机端切换模型、运行检测并导出日志。手机端初步结果显示 EdgeLite 版本比原始模型更快，EdgeLite + WIoU/ProgLoss 在测试病害图上置信度最高。下一步我会完成 P5Slim512 二次轻量化的完整蒸馏训练，并补充各类别 recall 和更公平的手机端多图测试。
