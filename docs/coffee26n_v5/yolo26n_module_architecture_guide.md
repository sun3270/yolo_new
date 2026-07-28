# YOLO26n 模块结构手册：基线、论文解读与咖啡病害扩展

## 1. 阅读范围与使用方式

本文以当前仓库中的 YOLO26n 实现、模型 YAML 和实验代码为准，采用统一的模块卡片格式：

1. **目标**：该模块要解决什么表示或效率问题。
2. **结构**：输入经过哪些算子；`+` 表示残差相加，`cat` 表示通道拼接。
3. **数据流**：特征在空间尺度和通道维上的变化。
4. **取舍**：适合保留、替换或单独消融的原因。

论文 *Toward a Deeper Understanding of YOLO26*（doi: `10.20944/preprints202603.2518.v1`）提供了这种逐模块讲解和单因素消融的范式。本文没有逐句复制其受版权保护的文本，而是根据本仓库源码重新组织，并补全其未覆盖的本地模块。论文仍是未同行评审预印本；其 COCO 结果仅作为生成假设的证据，不等同于咖啡病害数据集上的结论。

## 2. 原生 YOLO26n 全图

当前原生配置：`edgelite_experiment/local_ultralytics/ultralytics/cfg/models/26/yolo26.yaml`。

```text
Image
  -> Conv s=2 (P1/2)
  -> Conv s=2 (P2/4) -> C3k2(False)
  -> Conv s=2 (P3/8) -> C3k2(False) -------------------- P3 skip
  -> Conv s=2 (P4/16) -> C3k2(True) -------------------- P4 skip
  -> Conv s=2 (P5/32) -> C3k2(True) -> SPPF -> C2PSA -- P5 skip

P5 -> Upsample -> cat(P4 skip) -> C3k2(True) -> P4 head
P4 -> Upsample -> cat(P3 skip) -> C3k2(True) -> P3 head
P3 -> Conv s=2 -> cat(P4 head) -> C3k2(True) -> P4 head
P4 -> Conv s=2 -> cat(P5 skip) -> C3k2(True, attn=True) -> P5 head
Detect(P3, P4, P5)
```

`P3/P4/P5` 的步长分别为 `8/16/32`。P3 保留较多病斑边缘和小目标细节，P5 的空间图较小但语义和感受野最大；颈部通过上采样、下采样和拼接让三种尺度反复交换信息。

### 2.1 一次 640x640 前向传播的张量总账

下表按 `YOLO26n` 的宽度系数 `0.25` 展开，忽略 batch 维的变化。`B` 是 batch size，实际咖啡图像经 letterbox 后也遵循相同的 stride 关系。

| 节点 | 张量形状 `(B,C,H,W)` | 该张量后续去向 |
|---|---:|---|
| 输入 | `(B,3,640,640)` | 进入骨干 |
| P1 | `(B,16,320,320)` | 下采样至 P2 |
| P2/C3k2(False) | `(B,64,160,160)` | 下采样至 P3；P2P5 实验也把它作为细节源 |
| P3/C3k2(False) | `(B,128,80,80)` | 保存为 P3 skip；进入 P4 |
| P4/C3k2(True) | `(B,128,40,40)` | 保存为 P4 skip；进入 P5 |
| P5/C2PSA 后 | `(B,256,20,20)` | 保存为 P5 skip；向上采样到 P4 |
| neck P4 | `(B,128,40,40)` | 向上采样到 P3；也用于下行路径 |
| Detect P3 | `(B,64,80,80)` | 小目标/病斑检测分支 |
| Detect P4 | `(B,128,40,40)` | 中尺度检测分支 |
| Detect P5 | `(B,256,20,20)` | 大目标/全局语义检测分支 |

这个表是阅读其余 Flow 的坐标系。桥接模块若把 P3 或 P2 送往 P5，必须先下采样到 `20x20`；P5 回馈 P3 时必须先上采样到 `80x80`；任何 `cat` 前两路的 `H,W` 必须完全一致。

## 3. 原生模块卡片

### 3.1 Conv 与 Bottleneck

**目标**：以可融合的卷积单元提取局部模式，并在通道数不变时保留残差信息。

```text
Conv:       x -> Conv2d -> BatchNorm -> SiLU
Bottleneck: x -> 1x1 Conv -> 3x3 Conv -> (+ x, 条件满足时) -> y
```

**数据流**：步长为 2 的 Conv 同时下采样和扩展通道；步长为 1 的 Conv 保持特征图大小。训练结束后 Conv 和 BN 可以融合，减少推理开销。

**取舍**：论文的单因素消融显示，SiLU 相比 ReLU 在其 COCO 设定下多 `1.25` 个 mAP50-95 点（`0.3933` 对 `0.3808`），代价是约 `0.06 ms`。咖啡路线默认保留 SiLU；只有严格的速度约束才把 ReLU 作为独立消融。

### 3.2 C2f、C3k 与 C3k2

**目标**：用 CSP 分流减少重复计算，同时让一条路径持续进行特征精炼。

```text
C2f/C3k2:
x -> cv1(1x1) -> split(y0, y1)
                  y1 -> [refine] x n -> y2 ...
     cat(y0, y1, y2, ...) -> cv2(1x1) -> y

C3k2(False): refine = Bottleneck
C3k2(True):  refine = C3k(内部含两个 Bottleneck)
C3k2(attn=True): refine = Bottleneck -> PSABlock
```

**数据流**：浅层 P2/P3 的空间图很大，原生图采用 `False`，让细粒度特征以较低成本通过；P4/P5 和颈部采用 `True`，把更多计算预算投入语义特征和多尺度融合。最后一个 P5 C3k2 使用 `attn=True`。

**取舍**：论文中“全部 True”与默认混合配置精度几乎相同（`0.3930` 对 `0.3933`），但延迟由 `0.99` 增至 `1.11 ms`；“全部 False”最快（`0.86 ms`）却降至 `0.3813`。因此默认混合布局是应保留的基线，而不是全局开关。

### 3.3 SPPF

**目标**：在 P5 汇聚不同感受野的上下文，避免多分支金字塔的重复计算。

```text
x -> cv1(1x1, C/2) -> p0
p0 -> MaxPool(k) -> p1 -> MaxPool(k) -> p2 -> MaxPool(k) -> p3
cat(p0, p1, p2, p3) -> cv2(1x1) -> (+ x, 若 shortcut=True) -> y
```

连续三次 `k=5` 池化等效覆盖 `5/9/13` 的多尺度范围。当前 YOLO26 配置启用 `shortcut=True`。

**取舍**：移除 SPPF 仅节省约 `0.03 ms`，但 mAP50-95 从 `0.3933` 降至 `0.3866`，应保留。论文的 `k=7` 差异仅 `+0.0002 mAP`、`-0.01 ms`，远小于应有的多随机种子验证阈值，不能作为直接改图的依据。

### 3.4 PSABlock 与 C2PSA

**目标**：在不让全通道都承担注意力代价的前提下，建模远距离空间关系。

```text
PSABlock:
x -> Multi-head Attention -> +x -> FFN(1x1 expand -> 1x1 project) -> + -> y

C2PSA:
x -> cv1(1x1) -> split(a, b)
                  b -> PSABlock x n -> b'
     cat(a, b') -> cv2(1x1) -> y
```

`C2PSA` 位于 SPPF 后、P5 语义源处；一半通道绕行，另一半通道接受 PSA，因此兼顾局部保真和全局关系。

**取舍**：删除 C2PSA 的 COCO mAP50-95 从 `0.3933` 降至 `0.3704`，只换来 `0.07 ms`，不应删除。论文的 `attn_ratio=0.75` 达到 `0.3961` 且延迟未变，是值得在咖啡数据上以多种子单独复验的候选，不是既定结论。

### 3.5 Upsample、Concat 与 Detect

**目标**：恢复空间分辨率、合并不同尺度特征，并分别在 P3/P4/P5 预测目标。

```text
Upsample: x(H,W) -> x(2H,2W)
Concat:   cat(同空间尺度的两个特征, channel)
Detect:   [P3, P4, P5] -> 每尺度分类 + 回归 -> 端到端检测输出
```

**数据流**：P5 向上提供语义，P3 skip 提供边缘和位置；随后再向下构建 P4/P5 检测特征。删除 Concat 虽只省约 `0.02 ms`，mAP50-95 却降至 `0.3743`，说明跨尺度拼接是基础能力。论文中 Bilinear 上采样的单次结果略好于 Nearest（`0.3954` 对 `0.3933`），但增幅只有 `0.0021`，需要独立复验。

## 4. 本仓库自定义模块：基础算子与流

这些类位于 `edgelite_experiment/local_ultralytics/ultralytics/nn/modules/block.py`，并已通过模块导出和 `parse_model()` 注册供 YAML 使用。

### 4.1 ConvBNAct、DWSeparableConv 与 LDSConv

```text
ConvBNAct:      Conv2d -> BatchNorm -> SiLU
DWSeparableConv: Depthwise ConvBNAct -> Pointwise(1x1) ConvBNAct
LDSConv:         DWSeparableConv(k=3, s=2)
```

**目标**：以深度可分离卷积替代原生下采样的标准卷积，降低空间卷积成本。

**实际位置**：EdgeLite YAML 用 `LDSConv` 代替 P3/P4/P5 的下采样和颈部下采样。它改变的是计算方式，不改变 P3/P4/P5 的尺度语义。

### 4.2 TextureStreamP3 与 SemanticStreamP5

```text
TextureStreamP3:
P3 -> 1x1 reduce -> DW 3x3 -> DW 3x3 -> + reduced P3

SemanticStreamP5:
P5 -> 1x1 reduce -> DW 5x5 -> 1x1 -> + reduced P5
```

**目标**：把 P3 解释为纹理/病斑边缘源，把 P5 解释为叶片形态、照明和全局语义源。两者先压到相同中间通道数，再进入跨尺度桥接。

### 4.3 FastNormFuse2、RoleAwareAttnFuse2 与 MonitoredRoleAwareAttnFuse2

```text
FastNormFuse2:  y = relu(w1)/(sum+eps)*x1 + relu(w2)/(sum+eps)*x2

RoleAwareAttnFuse2:
z = cat(x1, x2, abs(x1-x2), x1*x2)
z -> GAP -> 1x1 -> SiLU -> 1x1 -> reshape(2,C) -> softmax(两路)
y = alpha(C)*x1 + beta(C)*x2
```

**目标**：不是给每个像素做昂贵的空间注意力，而是为 P3 纹理流和 P5 语义流的每个通道分配两路权重。投影层零初始化，因此训练开始时两路权重大致各半，避免新桥接立刻破坏预训练表示。

`MonitoredRoleAwareAttnFuse2` 在此基础上累计 P3/P5 平均权重与二元熵，用于判断融合是否塌缩到单一路径。

### 4.4 SimAM

**结构**：逐通道计算空间均值和方差，以无参数能量函数得到权重，再执行 `x * sigmoid(weight)`。

**目标**：压制复杂背景中的低显著响应。它只存在于部分历史 EdgeLite/Bridge 变体，`p5_largekernel_experiment` 明确关闭 SimAM，不能默认视为当前路线组成部分。

## 5. 本仓库自定义模块：跨尺度桥接

### 5.1 EdgeLGMSFBridge

```text
P3 -> TextureStreamP3 -> DW downsample x2 ----\
                                               RoleAwareAttnFuse2 -> 1x1 -> enhanced P5
P5 -> SemanticStreamP5 ------------------------/
```

**目标**：把细粒度纹理下采样至 P5 尺度，与 P5 语义自适应融合。当前 `yolo26n_edgelite.yaml` 使用此单向 P3-to-P5 桥。

### 5.2 ResidualEdgeLGMSFBridge

```text
base = project(P5)
refine = out(RoleAwareFuse(down(Texture(P3)), Semantic(P5)))
y = base + sigmoid(gate) * refine
```

**目标**：把桥接作为受保护的 P5 残差修正而非直接替换。小门值起始意味着先保持 P5，再由数据决定是否放大纹理反馈。

### 5.3 P5ToP3SemanticFuse（BiBridge 的反向支路）

```text
P3 -> 1x1 reduce -------------------------------\
P5 -> 1x1 reduce -> upsample -> FastNormFuse2 -> 1x1 -> gated residual + P3
```

**目标**：把 P5 的语义反馈送回 P3，帮助小病斑检测具有全局叶片上下文。

**状态**：在 `yolo26n_edgelite_bibridge.yaml` 中实现并使用；`p5_largekernel_experiment` 显式关闭该反馈，不能与当前 P5 大核历史最佳图混为一谈。

### 5.4 DistanceAwareP2P5Bridge

```text
P2 -> NearDetailConvBlock -> stride-2 downsample x3 ---\
                                                        RoleAwareAttnFuse2 -> 1x1 -> P5 enhancement
P5 -> FarContextConvBlock -----------------------------/
```

**目标**：P2 只作为极小病斑/边缘细节来源，P5 提供远场上下文，在 P5 尺度相遇；检测头仍只输出 P3/P4/P5。

**状态**：`yolo26n_p2p5_distanceconv_experiment` 的主旨是保留 P2 信息而删除直接 P2 Detect。原因是 `imgsz=960` 下 P2 的 `240x240` 网格代价很高，历史收益仅 `+0.00014 mAP50-95`，不足以证明四尺度检测值得保留。

## 6. 本仓库自定义模块：病斑细节与上下文

### 6.1 NearDetailConvBlock

```text
x -> 1x1 project -> base
base -> RepConv(3x3) -----------------------------\
base -> Conv(1x3) -> Conv(3x1) ------------------- cat -> 1x1 mix -> detail
y = base + sigmoid(gate) * detail
base ----------------------------------------------/
```

**目标**：用局部 3x3 和水平/垂直可分离卷积同时保留斑点边缘、叶脉方向性与短距离纹理。门控残差使其初始是对基础特征的温和补充。

### 6.2 FarContextConvBlock

```text
x -> 1x1 reduce -> base
base -> local 3x3
base -> factorized large kernel (1xK -> Kx1)
base -> dilated 3x3 -> 1x1
base -> GAP -> 1x1 -> broadcast
四路 cat -> GAP selector -> softmax(4) -> 加权和 -> 1x1 -> + base
```

**目标**：为 P5 同时提供局部、较大感受野、空洞上下文和全局上下文，不预设哪一路对任意病斑都最好，而由样本自适应选择。

### 6.3 P5LargeKernelContext

```text
P5 -> base
base -> depthwise 3x3
base -> depthwise (1x13 -> 13x1)
base -> GAP -> 1x1 -> broadcast
三路 selector softmax -> 1x1 -> sigmoid(gate) residual + base
```

**目标**：仅在预训练 P5 `C2PSA` 之后添加 13x13 等效大核上下文，而非重建整个骨干。门初值为 `sigmoid(-3)≈0.047`，保证新支路从小残差开始学习。

**状态**：`p5_largekernel_experiment` 的唯一新增结构；它保留 ELTEB、EdgeLGMSFBridge、WIoU+ProgLoss，关闭 BiBridge 和 SimAM。

## 7. 本仓库自定义模块：ELTEB 纹理注入

### 7.1 ELTEB 与 ELTEBLite

```text
image -> gray -> Sobel-x/Sobel-y -> gradient magnitude -> Laplacian
full mode: 还加入 RGB 与 unsharp(RGB - avgpool(RGB))
texture cues -> 1x1 compress -> 1x1 project -> texture(P3 channels)

P3 = C3k2(feature)
output = P3 + alpha * texture
      or  Conv1x1(cat(P3, alpha * texture))
```

**目标**：在 P3 引入显式病斑边缘、斑点和局部清晰度线索。`ELTEB` 使用完整线索；`ELTEBLite` 只使用灰度、Sobel 和 Laplacian，通道更少。

**安全初始化**：`fusion_alpha=0` 时 add 路径起始为原 P3；concat 路径的 1x1 被初始化成只传递原 P3。`BoundedELTEB` 把纹理融合和 unsharp 强度限制在有限范围，避免训练把人工纹理放大到失控。

## 8. 本仓库自定义训练目标：WIoU + ProgLoss

它们不是网络层，但确实是我们写入训练链路的核心组件，位于 `wiou_progloss_experiment/`。

### 8.1 BboxWIoUProgLoss

```text
CIoU loss = 1 - CIoU(pred, target)
WIoU loss = (1 - raw IoU) * dynamic_focus(beta) * center_distance_gain
box loss = (1 - progress_blend) * CIoU loss + progress_blend * WIoU loss
```

训练维护当前损失的滑动均值，以 `beta = current_loss / running_mean` 计算非单调聚焦权重，并限制其最小/最大值。`ProgLoss` 调度在前 `10%` 训练进度保持原损失，到 `60%` 平滑过渡到 WIoU，避免训练起始阶段过早改变优化地形。

### 8.2 ProgLoss 分类重加权

```text
class_weight = clamp((max_count / class_count)^0.5, 0.75, 1.8)
positive BCE scale = 1 + lambda(progress) * (class_weight - 1)
```

只放大正样本分类项中的尾类权重，避免把背景负样本也错误放大。WIoU/ProgLoss 通过运行时 patch 注入 `DetectionModel.init_criterion`，原生 Ultralytics loss 仍可回退用于公平消融。

## 9. 逐步 Flow：从输入到检测，再到本地扩展

这一节把前面的结构图展开为一次真实前向传播。除非特别说明，`Conv` 均指当前实现中的 `Conv -> BN -> SiLU`，`1x1` 不改变空间尺寸，`s=2` 将高宽减半。

### 9.1 原生骨干和颈部的完整 Flow

```text
1. x0 = image                                      # (B,3,640,640)
2. x1 = Conv3x3,s2(x0)                             # (B,16,320,320), P1
3. x2 = Conv3x3,s2(x1) -> C3k2(False)              # (B,64,160,160), P2
4. x3 = Conv3x3,s2(x2) -> C3k2(False)              # (B,128,80,80), P3 skip
5. x4 = Conv3x3,s2(x3) -> C3k2(True)               # (B,128,40,40), P4 skip
6. x5 = Conv3x3,s2(x4) -> C3k2(True) -> SPPF
        -> C2PSA                                   # (B,256,20,20), P5 skip

7. h4 = C3k2(True)(cat(upsample(x5, 40x40), x4))   # cat: 256+128=384 -> (B,128,40,40)
8. h3 = C3k2(True)(cat(upsample(h4, 80x80), x3))   # cat: 128+128=256 -> (B,64,80,80)
9. h4' = C3k2(True)(cat(Conv3x3,s2(h3), h4))       # cat: 64+128=192  -> (B,128,40,40)
10.h5 = C3k2(True,attn=True)(cat(Conv3x3,s2(h4'), x5))
                                                       # cat: 128+256=384 -> (B,256,20,20)
11.Detect([h3, h4', h5])
```

步骤 7-8 是自顶向下路径：语义由 P5 传向高分辨率特征。步骤 9-10 是自底向上路径：融合过 P3 细节的特征再回到 P4/P5。这里的 P3/P4/P5 是检测头输出尺度，不是把原骨干的 skip 特征直接送进 Detect。

### 9.2 C3k2 的逐步 Flow

以输入 `x:(B,Cin,H,W)`、输出 `y:(B,Cout,H,W)` 为例，令隐藏通道为 `Ch=floor(Cout*e)`。

```text
1. t = cv1_1x1(x)                                  # (B,2*Ch,H,W)
2. [y0, y1] = split(t, channel, equal)             # 两路各 (B,Ch,H,W)
3. states = [y0, y1]
4. repeat n times:
      z = refine(states[-1])
      states.append(z)
5. u = cat(states, channel)                        # (B,(2+n)*Ch,H,W)
6. y = cv2_1x1(u)                                  # (B,Cout,H,W)
```

`refine` 的内部选择决定模块实际复杂度：

```text
c3k=False: refine(a) = Bottleneck(a)
  a -> 1x1 reduce -> 3x3 recover -> (+a, 若 shortcut) -> z

c3k=True:  refine(a) = C3k(a)
  a -> CSP split -> 两个 Bottleneck 串联的变换支路 -> cat(skip, transform) -> 1x1 -> z

attn=True:  refine(a) = PSABlock(Bottleneck(a))
  a -> Bottleneck -> self-attention residual -> FFN residual -> z
```

注意 `C3k2` 不只是“一个残差块”：它会把初始两路和每次 refinement 的中间状态全部拼接。因此删掉 `True` 或 attention 不仅改变单层算子，也改变被 `cv2` 汇总的特征族。

### 9.3 SPPF 和 C2PSA 的逐步 Flow

```text
SPPF(x):
1. p0 = cv1_1x1(x)                                 # 通道减半
2. p1 = maxpool_k5(p0)                              # 空间尺寸不变
3. p2 = maxpool_k5(p1)                              # 顺序池化，不是独立池化
4. p3 = maxpool_k5(p2)
5. u  = cat(p0,p1,p2,p3, channel)                  # 4 倍隐藏通道
6. y  = cv2_1x1(u)
7. return y+x if shortcut and Cin==Cout else y

C2PSA(x):
1. t = cv1_1x1(x)                                  # (B,2*Ch,H,W)
2. [a,b] = split(t, channel)                        # a 是低成本保留支路
3. for each PSABlock:
      q,k,v = attention projections(b)
      A = softmax(q @ k^T / sqrt(d))
      b = b + project(A @ v)                        # 注意力残差
      b = b + FFN_1x1_expand_project(b)             # FFN 残差
4. y = cv2_1x1(cat(a,b, channel))
```

在当前 `P5:(B,256,20,20)` 上，C2PSA 只让处理支路进入 PSA，而不是让 256 个通道全量进入。空间 token 数为 `20*20=400`，这是把注意力安排在 P5 而非 P3 的关键原因：若直接在 P3 做同类全局注意力，token 数会变为 `80*80=6400`，计算量显著增加。

### 9.4 Detect 的逐步 Flow

当前 YAML 设置 `end2end=True`、`reg_max=1`。三个尺度分别接受 `h3/h4'/h5`：

```text
for feature Fi in [P3, P4, P5]:
1. box_i = BoxHead(Fi)
   Fi -> Conv3x3 -> Conv3x3 -> Conv1x1 -> 4*reg_max 个回归通道
2. cls_i = ClassHead(Fi)
   Fi -> DWConv3x3 -> 1x1 -> DWConv3x3 -> 1x1 -> Conv1x1 -> nc 个分类通道
3. flatten box_i 和 cls_i 的 H*W 位置

4. boxes  = cat(flatten(box_P3), flatten(box_P4), flatten(box_P5), location)
5. scores = cat(flatten(cls_P3), flatten(cls_P4), flatten(cls_P5), location)
6. 训练：同时构造 one-to-many 和 one-to-one 两组头；one-to-one 输入是 detach 后的特征。
7. 推理：解码 one-to-one 的 box，scores 经过 sigmoid，再执行端到端后处理。
```

因此 `Detect` 不是一层简单 1x1 卷积。它给每个尺度保留独立的分类与回归路径，再在 location 维合并；这也是为何直接添加 P2 Detect 会使候选位置从 `80*80 + 40*40 + 20*20=8400` 增加 `160*160=25600` 个。

### 9.5 EdgeLite 和单向 P3-to-P5 Bridge 的逐步 Flow

```text
LDSConv(x):
1. depthwise 3x3,s2：每个输入通道独立下采样
2. pointwise 1x1：混合通道并投影到目标通道数

EdgeLGMSFBridge([P3,P5]):
1. p3r = ConvBNAct_1x1(P3)                         # 先压缩 P3 通道
2. p3t = p3r + DW3x3(DW3x3(p3r))                  # 纹理残差提炼
3. p3d = DWSeparable_s2(DWSeparable_s2(p3t))       # 80x80 -> 20x20
4. p5r = ConvBNAct_1x1(P5)
5. p5s = p5r + ConvBNAct_1x1(DW5x5(p5r))           # 语义残差提炼
6. 对齐：若 p3d 与 p5s 的 H,W 仍不同，nearest 插值 p3d 至 p5s
7. f = RoleAwareAttnFuse2(p3d,p5s)
8. y = ConvBNAct_1x1(f)                            # 投影到 P5/neck 目标通道
```

`RoleAwareAttnFuse2` 的 Flow 是：

```text
1. z = cat(p3d, p5s, abs(p3d-p5s), p3d*p5s)        # (B,4C,20,20)
2. g = GAP(z)                                      # (B,4C,1,1)，只提取通道统计
3. logits = Conv1x1(SiLU(Conv1x1(g)))              # (B,2C,1,1)
4. [alpha,beta] = softmax(reshape(logits,B,2,C,1,1), route)
5. f = alpha*p3d + beta*p5s                        # 每通道、全空间共享的两路权重
```

这说明该模块不是空间对齐网络：它假设上游已经完成尺度对齐，然后只决定每个通道更信任纹理还是语义。

### 9.6 BiBridge 的 P5-to-P3 反馈 Flow

```text
P5ToP3SemanticFuse([P3,P5]):
1. a = ConvBNAct_1x1(P3)                            # (B,Cmid,80,80)
2. b = ConvBNAct_1x1(P5)                            # (B,Cmid,20,20)
3. b = nearest_upsample(b, size=a.HW)               # 20x20 -> 80x80
4. w = relu([wP3,wP5]) / (sum(w)+eps)
5. r = ConvBNAct_1x1(wP3*a + wP5*b)
6. y = shortcut(P3) + sigmoid(gate) * r
```

输出 `y` 与原 P3 同尺度，因而可以替换 P3 skip 进入头部。门控保护原 P3；若训练没有证明反馈有效，`sigmoid(gate)` 应保持小值，而不是把 P5 语义强行覆盖到边缘特征上。

### 9.7 P2-P5 DistanceAware Flow

```text
DistanceAwareP2P5Bridge([P2,P5]):
1. n0 = NearDetailConvBlock(P2)
2. n1 = ConvBNAct_3x3,s2(n0)                        # P2 160x160 -> 80x80
3. n2 = ConvBNAct_3x3,s2(n1)                        # 80x80 -> 40x40
4. n3 = ConvBNAct_3x3,s2(n2)                        # 40x40 -> 20x20
5. f0 = FarContextConvBlock(P5)
6. 对齐：若 n3.HW != f0.HW，nearest 插值 n3
7. mix = RoleAwareAttnFuse2(n3,f0)
8. y = ConvBNAct_1x1(mix)
```

`NearDetailConvBlock` 的内部 Flow 是 `project -> {RepConv3x3, 1x3->3x1, identity} -> cat -> 1x1 mix -> gated residual + base`。`FarContextConvBlock` 则把同一个 P5 base 并行送入 local 3x3、大核 `(1xK->Kx1)`、dilated 3x3 和 `GAP->broadcast` 四路；四路拼接后的全局池化统计生成 4 个 softmax 权重，最后加权回注 base。P2 在此设计中没有直接产生预测，只影响 P5 表示。

### 9.8 P5LargeKernelContext 和 ELTEB 的逐步 Flow

```text
P5LargeKernelContext(P5):
1. base = Identity(P5) 或 1x1 project(P5)
2. local  = DW3x3(base)
3. large  = DW1x13(base) -> DW13x1(...)            # 13x13 等效感受野
4. global = broadcast(ConvBNAct_1x1(GAP(base)))
5. route  = softmax(MLP(GAP(cat(local,large,global))))
6. ctx    = route0*local + route1*large + route2*global
7. y      = base + sigmoid(gate) * ConvBNAct_1x1(ctx)

ELTEB([image, backbone_feature]):
1. p3 = C3k2(backbone_feature)
2. image 缩放至 p3.HW；gray = 0.299R+0.587G+0.114B
3. sx = SobelX(gray), sy = SobelY(gray)
4. sobel = tanh(sqrt(sx^2+sy^2)); lap = tanh(abs(Laplacian(gray)))
5. full 模式额外生成 blur=avgpool(image) 与 unsharp=image+a*(image-blur)
6. cues = cat(image,gray,sobel,lap,unsharp) 或 cat(gray,sobel,lap)
7. tex = project_1x1(compress_1x1(cues))           # 通道投影至 p3.C
8. y = p3 + alpha*tex，或 fuse_1x1(cat(p3,alpha*tex))
```

ELTEB 把原始图像的固定滤波线索直接送进 P3，因此与仅处理深层特征的 C2PSA/P5 大核完全不同。`alpha=0` 和 concat identity 初始化保证它在加载预训练权重时先近似原图，再逐步学习是否需要纹理残差。

### 9.9 WIoU + ProgLoss 的训练时 Flow

```text
每个 batch:
1. Detect 输出 boxes、scores、feats；assigner 依据预测和 GT 指派正样本。
2. 对正样本：计算 CIoU loss 与 raw IoU；维护 raw loss 的 running mean。
3. beta = current_box_loss / running_mean -> non-monotonic focus(beta)。
4. WIoU = (1-raw_iou) * focus * bounded_distance_gain。
5. blend = smoothstep(epoch_ratio, 0.10, 0.60)。
6. box = (1-blend)*CIoU + blend*WIoU；DFL/L1 分支沿用原生实现。
7. 从训练集类别计数生成 class_weight；仅在 target_score>0 的 BCE 项上乘逐步增加的尾类权重。
8. box、class、DFL 乘原生 hyp 系数后相加，反向传播。
```

这条 Flow 发生在训练 criterion 内，不会改变导出后的计算图或推理延迟；它改变的是哪些正样本和尾类错误在不同训练阶段获得更大的梯度权重。

## 10. 现有模块完整档案（逐模块 Flow 与统一实验门槛）

本节沿用论文“目标—结构—逐步 Flow—起效原理—单因素消融”的讲解方式，但所有本仓库自定义模块均以实际源码和 YAML 为准，不把结构动机写成已经验证的收益。`待验证` 表示已有实现或候选插入点，但尚未达到至少 3 个随机种子的结论门槛。

阅读尺度约定：默认输入为 `640x640`；因此 P2/P3/P4/P5 典型空间尺度分别为 `160/80/40/20`。通道数按 YOLO26n 或对应实验 YAML 的实际缩放结果描述；`+` 表示逐元素残差相加，`cat` 表示通道拼接，`GAP` 表示全局平均池化。

### 10.1 原生 YOLO26n 模块：逐模块深度解析

#### A01 Conv

##### 名称、位置与输入/输出尺度

**名称：** `Conv`。**所在阶段：** 骨干、颈部和检测头的通用基本算子；原生 YAML 节点 0、1、3、5、7、17、20 等。  
**输入/输出尺度与通道：** 一般形式为 `(B,Cin,H,W) -> (B,Cout,ceil(H/s),ceil(W/s))`；640 输入的首层为 `(B,3,640,640) -> (B,16,320,320)`。

##### 目标：它要解决什么问题

卷积块同时承担局部模式提取、通道重编码和按步长下采样。它要把像素邻域中的边缘、颜色过渡和纹理变成可学习特征，并在进入更深阶段时用更小的空间网格换取更多通道。对于咖啡病斑，浅层卷积主要响应斑点边缘、叶脉和颜色突变；深层卷积则在已有特征上组合出病斑形状和叶片语义。

##### 结构图

```text
x -> Conv2d(k,s,p,d,groups,bias=False) -> BatchNorm2d -> SiLU -> y
推理融合: Conv2d + BN -> fused Conv2d -> SiLU
```

##### 逐步 Flow：数据到底怎样经过它

1. `autopad` 根据核大小和 dilation 计算 padding；stride=1 时通常保持 H、W，stride=2 时把空间尺寸约减半。
2. 无偏置 `Conv2d` 在每个局部窗口上做交叉相关，产生 `Cout` 个响应图；参数在所有空间位置共享，因此同一种病斑边缘在不同位置可由同一组核响应。
3. BatchNorm 对每个输出通道做标准化和仿射变换，缓和不同 batch/层之间的尺度漂移。
4. SiLU 用 `x*sigmoid(x)` 引入平滑非线性，负值不会像 ReLU 那样被全部截断，弱纹理信号仍可能保留梯度。
5. 部署前可把 BN 的均值、方差、缩放和偏置吸收到卷积权重与偏置中；`forward_fuse` 跳过独立 BN，但数学功能保持等价。

##### 起作用的原理

卷积的核心不是简单缩小图片，而是以局部感受野建立平移共享的特征探测器。stride=2 会减少采样点，降低后续计算量；与此同时增加通道数，让每个较粗位置保存更多类型的响应。BN 使不同通道落在更稳定的数值范围内，SiLU 则提供连续梯度。原论文对 SiLU、ReLU 和 LeakyReLU 做单因素比较，说明激活函数必须同时按精度和真实推理延迟评估，不能仅凭理论 FLOPs 判断。

##### 初始化保护策略

原生 Conv 应优先继承预训练权重。若只想比较激活函数，必须保持卷积核、通道、stride、BN、训练配方全部不变；若替换 stride-2 Conv，则要逐层核对输出尺度和 Detect 的 stride。导出时需验证 Conv-BN 融合前后最大绝对误差。

##### YAML 插入点

原生 `yolo26.yaml` 中所有 `Conv` 节点都可直接定位；首轮实验只选择一个角色相同的节点组，例如仅替换骨干 stride-2 Conv，不要同时改检测头下采样。

##### 与基线相比唯一改变

基线为 `Conv2d(bias=False)+BN+SiLU`。单因素消融只能改变一个维度，例如 `SiLU -> ReLU`、标准卷积 -> 深度可分离卷积，或某一节点的 kernel/stride；不能把激活、卷积类型和通道宽度同时改变。

##### 预期收益及潜在代价

**预期收益：** 形成稳定的局部特征层次；stride-2 节点显著降低后续计算；BN 融合后部署开销低。  
**潜在代价与失败模式：** 早期大分辨率卷积主导 FLOPs；过强下采样会让微小病斑在到达 P3 前消失；不同后端对 SiLU、depthwise 和融合卷积的优化程度不同，理论轻量化可能不降低实测延迟。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：每个被替换节点的输出均值/方差、梯度范数、Conv-BN 融合误差、P2/P3 小病斑召回和各后端算子耗时。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自本仓库 `conv.py::Conv`；论文第 5 节提供 Objective、Flow 和激活消融范式。论文的 COCO/H100 数值只能作为设计证据，不能直接当作咖啡数据集结论。

#### A02 Bottleneck

##### 名称、位置与输入/输出尺度

**名称：** `Bottleneck`。**所在阶段：** C3k2、C3k 与注意力版 C3k2 的内部局部精炼单元。  
**输入/输出尺度与通道：** 通常 `(B,C,H,W) -> (B,C,H,W)`；若 `c1 != c2` 则不执行残差相加。隐藏通道为 `Cmid=int(Cout*e)`。

##### 目标：它要解决什么问题

Bottleneck 在不改变空间尺寸的条件下，用两层局部卷积对特征进行压缩、处理和恢复，并通过残差连接保留未变换信息。它解决的是“需要增加非线性与局部感受野，但不能让深层网络难以优化”的问题。

##### 结构图

```text
x -> Conv(k1): Cin->Cmid -> Conv(k2): Cmid->Cout -> f(x)
  \________________________________________ + -> y  (shortcut and Cin=Cout)
```

##### 逐步 Flow：数据到底怎样经过它

1. 第一层卷积把输入映射到 `Cmid=int(Cout*e)`；当 `e<1` 时先压缩通道以降低第二层卷积成本。
2. 第二层卷积在压缩空间内提取局部结构，并恢复到 `Cout`；`g` 可控制分组卷积。
3. 只有 `shortcut=True` 且 `Cin=Cout` 时才执行 `y=x+f(x)`，否则直接返回 `f(x)`，避免形状不匹配。
4. 反向传播时梯度既可经过两层卷积分支，也可沿 identity 路径直接传回，减轻深层串联导致的梯度衰减。

##### 起作用的原理

残差相加把模块表示成对恒等映射的修正：网络只需学习 `f(x)` 中真正有用的变化，而不是重新构造全部特征。如果新卷积分支在训练早期尚未学好，identity 仍能传递基础表示。对病斑检测而言，这有利于在增强局部纹理时保留叶片轮廓和原有颜色信息。压缩比例 `e` 决定容量与成本，过小会形成信息瓶颈，过大则增加卷积计算。

##### 初始化保护策略

继承原生 Bottleneck 权重时保持 `c1/c2/e/k/g/shortcut` 一致。新分支若没有预训练权重，应放在外层受门控残差保护的结构中；不要随意关闭所有 shortcut，因为这同时改变优化路径和表示函数。

##### YAML 插入点

通常不作为独立 YAML 节点，而由 `C3k2`、`C3k`、`ELTEB` 等内部构造。消融应通过外层模块参数控制，而不是在全局源码中改掉所有 Bottleneck。

##### 与基线相比唯一改变

相对纯两层卷积，唯一结构增量是条件残差；相对基线做实验时，应分别测试 `shortcut on/off` 或 `e`，每次只改变其中一项。

##### 预期收益及潜在代价

**预期收益：** 以较小代价增加局部非线性和有效深度，并改善梯度传播与特征复用。  
**潜在代价与失败模式：** 通道压缩过强可能丢失细粒度纹理；残差相加要求形状一致；在高分辨率层堆叠过多仍会显著增加延迟。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：shortcut 开关、隐藏通道比 `e`、残差分支输出与 identity 的范数比、各层梯度范数和小病斑 AP。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自 `block.py::Bottleneck`；论文第 6、7、12 节把它作为 C3k2 局部精炼与残差优化的核心组成。

#### A03 C3k2

##### 名称、位置与输入/输出尺度

**名称：** `C3k2`。**所在阶段：** YOLO26n 骨干 P2-P5 与颈部 P3-P5 的主干精炼模块。  
**输入/输出尺度与通道：** 保持空间尺度；通道由 YAML 的目标通道经宽度系数缩放。内部隐藏通道 `c=int(Cout*e)`，输出为 `(B,Cout,H,W)`。

##### 目标：它要解决什么问题

C3k2 用 CSP 式部分通道处理减少重复计算，同时保留原始、浅变换和深变换三类信息。它还通过 `c3k` 与 `attn` 两个开关，在同一个外壳内选择轻量 Bottleneck、较重 C3k 或 Bottleneck+PSA，从而把计算容量分配到最需要的层级。

##### 结构图

```text
x -> cv1(1x1, 2c) -> split(y0,y1)
                             y1 -> block1 -> y2 -> ... -> blockN
cat(y0,y1,y2,...,yN) -> cv2(1x1) -> y
block = Bottleneck | C3k | Bottleneck+PSABlock
```

##### 逐步 Flow：数据到底怎样经过它

1. `cv1` 用 1x1 卷积把输入映射为 `2c` 通道，并沿通道切成 `y0`、`y1` 两半。
2. `y0` 作为低变换路径直接保留；`y1` 是处理路径的起点。
3. 每个内部单元只接收上一个处理结果，并把新结果追加到列表，因此 concat 同时看到原始两半和不同深度的输出。
4. 当 `attn=True` 时，每个单元是 `Bottleneck -> PSABlock`；否则 `c3k=True` 选择嵌套 C3k，`c3k=False` 选择单 Bottleneck。
5. `cv2` 在 concat 后执行通道混合，把 `(2+n)c` 投影到 `Cout`，学习各深度特征应如何组合。

##### 起作用的原理

CSP 的作用不是简单分支，而是控制重复梯度路径：一部分特征少加工以保真，一部分逐级加工以增添语义，最后一次性融合。早期大网格若使用轻量 `c3k=False`，可限制计算；深层小网格使用 `c3k=True`，能以较低空间成本增加非线性；P5 末端 `attn=True` 则加入全局关系。论文第 8 节的核心观点是按阶段分配复杂度，而不是把最重配置铺满全网。

##### 初始化保护策略

基线不同节点的 `c3k/e/attn/n` 组合必须逐点保留。新增实验应复制 YAML 后只改目标节点；加载权重时检查 matched keys 和每个分支输出形状。不要把所有 C3k2 同时设为同一类型后仍声称是单模块消融。

##### YAML 插入点

原生节点为骨干 2/4/6/8，颈部 13/16/19/22；nano 中浅层使用 `False, e=0.25`，深层和颈部多用 `True`，末端节点 22 使用 `attn=True`。

##### 与基线相比唯一改变

基线的核心是混合配置。实验可只改一个节点的 `c3k`、`attn`、`e` 或 repeats；其余节点、通道和训练配方必须锁定。

##### 预期收益及潜在代价

**预期收益：** 提供高效多深度特征复用，并能把较重计算集中在小分辨率或融合后特征上。  
**潜在代价与失败模式：** concat 增加临时显存；全网启用重 C3k/attention 可能几乎不增益却显著增延迟；过度轻量化则削弱高层语义。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：逐节点 c3k/attn 配置、concat 峰值显存、各节点输出相似度、P3/P4/P5 分支 AP 与目标设备分层延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自 `block.py::C3k2/C2f` 和原生 YAML；论文第 6-8、12 节提供轻/重/注意力三种配置及分阶段放置的论证。

#### A04 C3k

##### 名称、位置与输入/输出尺度

**名称：** `C3k`。**所在阶段：** `C3k2(c3k=True)` 内部的较重 CSP 子块。  
**输入/输出尺度与通道：** 一般 `(B,C,H,W) -> (B,C,H,W)`；内部两条 1x1 分支，处理分支含 `n` 个 kernel=`k` 的 Bottleneck。

##### 目标：它要解决什么问题

C3k 在较小空间尺度上增加分支深度和局部建模能力。它解决轻量单 Bottleneck 在深层语义阶段容量不足的问题，同时仍通过 CSP 旁路保留未经多次卷积的特征。

##### 结构图

```text
x -> cv1 -> Bottleneck(k,k) x n -> a --\
x -> cv2 ---------------------------- cat -> cv3 -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 输入分别经过两个 1x1 投影，形成处理分支与旁路分支。
2. 处理分支串联 `n` 个 Bottleneck，每个内部卷积核由 `k` 指定；当前常用 `k=3`。
3. 两个分支在通道维拼接，不做空间尺寸改变。
4. 最后 1x1 卷积混合两路通道并投影到输出宽度。

##### 起作用的原理

与单 Bottleneck 相比，C3k 允许处理分支学习更深的局部组合；与纯串联卷积相比，旁路减少信息破坏并提供短梯度路径。它适合 P4/P5 或 neck 融合后，因为这些位置空间网格较小，增加深度的成本更可控。

##### 初始化保护策略

C3k 不应在所有高分辨率层无差别替换轻块。若用预训练权重，保持内部 `n/e/k/shortcut`；若改变 k，单独记录实际后端 latency，因为较大核未必被同等优化。

##### YAML 插入点

不直接作为原生 YAML 顶层节点；由 `C3k2(..., c3k=True)` 构造。节点 6/8/13/16/19 等间接使用。

##### 与基线相比唯一改变

相对 `C3k2(c3k=False)`，唯一变化是内部单 Bottleneck 替换为包含多个 Bottleneck 的 C3k 子块。

##### 预期收益及潜在代价

**预期收益：** 在较低分辨率上增加局部层次和语义表达，适合融合后特征整合。  
**潜在代价与失败模式：** 参数、激活和延迟均高于单 Bottleneck；置于早期大网格会放大成本，且可能对小数据集过拟合。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：内部重复数 n、核大小 k、处理/旁路特征范数、阶段延迟和按病斑尺寸分组的 AP。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自 `block.py::C3k`；论文第 7-8 节解释其嵌套 Bottleneck 与阶段化放置。

#### A05 SPPF

##### 名称、位置与输入/输出尺度

**名称：** `SPPF`。**所在阶段：** 骨干最深 P5，位于 C3k2(P5) 之后、C2PSA 之前的原生节点 9。  
**输入/输出尺度与通道：** YOLO26n 640 输入时 `(B,256,20,20) -> (B,256,20,20)`；隐藏通道 `C/2`，空间尺寸不变。

##### 目标：它要解决什么问题

SPPF 在不继续下采样的情况下扩大有效感受野，让 P5 同时看到局部、邻域和更大范围上下文。对于病斑识别，它可帮助区分单个斑点与整片叶片的颜色、遮挡和光照背景，避免只根据局部纹理误判。

##### 结构图

```text
x -> cv1(no act) -> p0 -> MaxPool5 -> p1 -> MaxPool5 -> p2 -> MaxPool5 -> p3
cat(p0,p1,p2,p3) -> cv2 -> y -> (+x if shortcut)
```

##### 逐步 Flow：数据到底怎样经过它

1. `cv1` 用无激活 1x1 卷积把通道减半，先降低连续池化和 concat 的成本。
2. 同一个 stride=1、padding=2 的 5x5 最大池化连续调用三次，得到等效于更大感受野的 p1、p2、p3。
3. p0-p3 保持相同 H、W，因此可沿通道直接拼接；每个位置同时拥有不同上下文范围的强响应。
4. `cv2` 把 `(n+1)*C/2` 融合回 Cout。当前 shortcut=True 且 Cin=Cout，因此最终再与原输入相加。

##### 起作用的原理

连续 5x5 池化复用中间结果，比并行大核池化更省计算；最大池化传播局部最强响应，对突出显著病斑或边缘有利。多尺度 concat 让后续 1x1 卷积自行选择所需上下文。YOLO26 版本的无激活 cv1 与残差意味着模块更接近“在原 P5 上添加多尺度修正”，初始化与优化更稳。

##### 初始化保护策略

保留预训练 SPPF 和 shortcut 作为基线。改变 k、n、shortcut 或 act 必须分别实验；禁止同时删除 SPPF 又增加其他上下文块后把差异归因给某一个模块。

##### YAML 插入点

原生节点 9：`[-1, 1, SPPF, [1024, 5, 3, True]]`。P5LargeKernelContext 应插在 C2PSA 之后作为新增残差，而不是覆盖此节点。

##### 与基线相比唯一改变

原生无变化；候选消融分别为 `k=3/7`、关闭 shortcut、删除 SPPF 或改变池化次数 n，每次只改一项。

##### 预期收益及潜在代价

**预期收益：** 以较低开销聚合多尺度上下文，保持 P5 空间大小，并为后续注意力提供更丰富输入。  
**潜在代价与失败模式：** 最大池化可能抹平弱小响应；concat 增加临时显存；删除可稍降延迟但可能明显伤害精度。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：k、n、shortcut、池化分支激活稀疏度、P5 有效感受野代理、上下文型误检和延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自 `block.py::SPPF`；论文第 9 节详细比较删除、shortcut 与 k=3/5/7，但其最优选择仍需在咖啡数据和目标设备上复验。

#### A06 PSABlock

##### 名称、位置与输入/输出尺度

**名称：** `PSABlock`。**所在阶段：** C2PSA 内部，以及 P5 末端 `C3k2(attn=True)` 的注意力精炼单元。  
**输入/输出尺度与通道：** `(B,C,H,W) -> (B,C,H,W)`；空间位置展平为 `N=H*W` 参与多头注意力，FFN 先扩到 `2C` 再压回 C。

##### 目标：它要解决什么问题

PSABlock 用全局位置间交互补足卷积局部感受野。它要回答的是：某个位置的病斑响应是否应结合叶片另一处的结构、颜色和语义来解释，同时通过 FFN 重新组合通道。

##### 结构图

```text
x -> Attention(Q,K,V + depthwise positional encoding) -> +x -> z
z -> 1x1(C->2C)+SiLU -> 1x1(2C->C,no act) -> +z -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. Attention 用 1x1 卷积一次生成 Q、K、V，并按 `num_heads` 拆分通道。
2. Q 与 K 在所有 N 个空间位置间计算缩放点积，softmax 得到每个查询位置对其他位置的权重。
3. 权重对 V 做加权求和；V 同时经过 depthwise 卷积形成位置编码并加到注意力结果，补充局部空间感。
4. 若 shortcut 开启，注意力输出与输入残差相加。
5. FFN 用两个 1x1 卷积完成逐位置通道扩张与压缩，再做第二次残差相加。

##### 起作用的原理

卷积只能通过堆叠逐步传播远距离信息，注意力可在一次操作中直接建立任意位置的相似关系。多头机制让不同通道子空间关注不同模式；depthwise 位置编码避免完全丢失空间邻接。双残差把注意力和 FFN 都限制为对输入的修正，降低过平滑和训练不稳风险。注意力复杂度随 N 的平方增长，因此只适合 P5 这类小网格。

##### 初始化保护策略

保持 P5 放置和原生 shortcut；不要把 PSABlock 直接移到 P2/P3 而不做复杂度评估。`num_heads` 必须与通道可分，权重加载需检查 QKV/proj 形状。

##### YAML 插入点

通常不直接作为节点；由 `C2PSA` 或 `C3k2(..., attn=True)` 构造。原生 P5 末端节点 22 使用注意力版 C3k2。

##### 与基线相比唯一改变

相对局部 Bottleneck/C3k，唯一增量是全局注意力和通道 FFN；消融可单独关闭 shortcut、改变 attn_ratio 或删除 Attention。

##### 预期收益及潜在代价

**预期收益：** 建立远距离依赖，提升复杂或分散病斑的全局一致性，并在低分辨率上保持可控成本。  
**潜在代价与失败模式：** N² 注意力可能占用显存并增加延迟；小数据集上更易过拟合，过强注意力会削弱局部边界。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：attention ratio、head 数、注意力熵、shortcut 开关、峰值显存、P5 latency 和跨区域误检。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自 `block.py::PSABlock/Attention`；论文第 10、12 节给出 QKV、位置编码、FFN 和双残差的逐步 Flow。

#### A07 C2PSA

##### 名称、位置与输入/输出尺度

**名称：** `C2PSA`。**所在阶段：** 原生骨干与颈部交界，SPPF 后的节点 10。  
**输入/输出尺度与通道：** 640 输入 YOLO26n 中 `(B,256,20,20) -> (B,256,20,20)`；按 `e=0.5` 分成两条 128 通道路径。

##### 目标：它要解决什么问题

C2PSA 把高成本全局注意力限制在部分通道，同时保留另一部分局部基线表示。它在进入 neck 前增强 P5 的长距离语义，使上采样到 P4/P3 的信息不只是局部卷积响应，而带有整叶片范围的上下文。

##### 结构图

```text
x -> cv1(1x1,2c) -> split(a,b)
                         a -------------------\
                         b -> PSABlock x n ---- cat -> cv2(1x1) -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 要求 Cin=Cout，`cv1` 产生 2c 通道并分成 a、b。
2. a 不进入注意力，保存卷积生成的局部/身份信息。
3. b 串联 n 个 PSABlock，逐次加入全局位置关系和 FFN 通道重编码。
4. a 与处理后的 b 拼接，`cv2` 再混合并恢复 Cout。

##### 起作用的原理

部分通道注意力在表达力和成本之间折中：注意力只处理 c 通道，降低 QK 计算和显存，同时 a 路防止全局混合覆盖全部局部细节。放在 SPPF 后意味着输入已经具备多尺度上下文，注意力再决定远距离位置之间的依赖。论文的删除实验显示该模块可能对精度贡献很大，但 attention ratio 的最优值依赖数据与后端。

##### 初始化保护策略

预训练基线必须保留完整 C2PSA；新 P5 上下文模块应加在其后并用小门控残差，不得替换掉它。改变 `e/n/attn_ratio/shortcut` 时一次只动一个参数。

##### YAML 插入点

原生节点 10：`[-1, 2, C2PSA, [1024]]`，nano 深度缩放后实际堆叠数需以解析后的模型为准。

##### 与基线相比唯一改变

基线无变化；候选消融为删除整个模块、改变注意力通道比例、PSABlock 数或 shortcut，每项独立比较。

##### 预期收益及潜在代价

**预期收益：** 以部分通道成本加入全局语义，并通过 CSP 旁路保持局部信息和训练稳定性。  
**潜在代价与失败模式：** 注意力增加峰值显存和导出复杂度；过大比例可能过拟合，删除虽然提速却可能造成明显精度损失。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：attn_ratio、PSABlock 数、注意力熵、P5 激活相似度、峰值显存和目标设备 attention kernel 延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自 `block.py::C2PSA`；论文第 10 节给出完整 Objective/Flow 与比例、shortcut、删除消融。

#### A08 Upsample / Concat

##### 名称、位置与输入/输出尺度

**名称：** `Upsample / Concat`。**所在阶段：** neck 的自顶向下与自底向上多尺度融合路径。  
**输入/输出尺度与通道：** Upsample：`(B,C,H,W)->(B,C,2H,2W)`；Concat：同 H、W 的 `(B,C1,H,W)+(B,C2,H,W)->(B,C1+C2,H,W)`。

##### 目标：它要解决什么问题

Upsample 负责让深层语义回到更细网格，Concat 负责把它与骨干浅层细节并排保留。二者共同解决单一尺度无法兼顾小病斑定位和大范围语义的问题，是 P3/P4/P5 检测头获得互补信息的关键路由。

##### 结构图

```text
P5 --Upsample x2--\
                       cat -> C3k2 -> neck P4
backbone P4 ----------/
neck P4 --Upsample x2--\
                         cat -> C3k2 -> Detect P3
backbone P3 ------------/
```

##### 逐步 Flow：数据到底怎样经过它

1. nearest Upsample 按整数倍复制空间值，通道不变、无可训练参数。
2. 在 concat 前，两路 H、W 必须完全一致；letterbox 和 stride 体系保证 P5->P4、P4->P3 可对齐。
3. Concat 沿通道维保留两路原始特征，不做平均，因此后续 C3k2 可学习选择与重组。
4. 下行路径用 stride-2 Conv 将 P3/P4 再降采样，并与先前 neck 特征 concat，形成 PAN 式双向聚合。

##### 起作用的原理

上采样本身不会恢复被丢失的新细节，它只把深层语义映射到更密坐标；真正的细节来自 backbone skip。Concat 不强迫两路立即相加，避免尺度与统计差异导致信息互相抵消。后续卷积才决定哪些通道有用。论文对插值方式和删除 concat 的消融说明：插值差异通常较小，而去掉跨层融合可能造成远大于延迟收益的精度损失。

##### 初始化保护策略

任何 concat 前必须断言空间尺寸一致；若改插值方式，只改 mode。禁止用 Identity 直接替换 Concat 而不同时修正后续输入通道，否则比较既不等价也可能无法加载权重。

##### YAML 插入点

原生 head 节点 11/12、14/15、17/18、20/21。自定义图必须保留相同 P3/P4/P5 对齐关系，并标清被增强特征来自哪个节点。

##### 与基线相比唯一改变

单因素可比较 nearest/bilinear/bicubic，或在通道匹配的重构图中移除某一条 skip；不得同时改变插值和融合模块。

##### 预期收益及潜在代价

**预期收益：** 融合高层语义与浅层定位细节，显著支持不同尺寸病斑；算子简单且易导出。  
**潜在代价与失败模式：** Concat 增加后续卷积输入通道与激活显存；错误对齐会导致伪影或运行时失败；更复杂插值不一定在目标设备更快。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：插值模式、concat 后通道数与峰值显存、各尺度 Recall/AP、边界误差和目标设备算子延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自原生 YAML、`nn.Upsample` 与 `conv.py::Concat`；论文第 11 节提供 Objective/Flow 和插值/Concat 消融。

#### A09 Detect

##### 名称、位置与输入/输出尺度

**名称：** `Detect`。**所在阶段：** 模型末端，接收 P3/P4/P5 三个尺度并输出框与类别。  
**输入/输出尺度与通道：** 输入为 `[P3:(B,64,80,80), P4:(B,128,40,40), P5:(B,256,20,20)]`（640、n-scale）；每位置输出 `4*reg_max + nc`，当前 YOLO26 YAML 为 `reg_max=1`、`end2end=True`。

##### 目标：它要解决什么问题

Detect 把多尺度特征转成可训练的边界框回归和类别分数，并在 end-to-end 模式下同时维护 one-to-many 与 one-to-one 分支，使推理可直接产生有限检测结果。它是特征工程最终兑现为检测结果的位置。

##### 结构图

```text
P3/P4/P5 -> [BoxHead: Conv->Conv->1x1] -- boxes --\
          each -> [ClsHead: DWConv+1x1 x2->1x1] -- scores -- cat locations
end2end: duplicate one-to-one heads on detached feats -> decode -> postprocess
```

##### 逐步 Flow：数据到底怎样经过它

1. 每个尺度分别进入 box head 和 class head；box head 输出 `4*reg_max`，class head 输出 nc。
2. 各尺度输出展平空间维并沿 location 拼接，640 输入候选位置数为 80²+40²+20²。
3. 训练时 one-to-many 使用密集匹配；end2end 还对 detached feature 运行 one-to-one 副本，形成更稀疏的最终匹配路径。
4. 推理时根据网格 anchor points 与 stride 解码框；reg_max=1 时 DFL 层为 Identity，直接回归四个距离。
5. 类别 logits 经 sigmoid；end2end 路径再按最高分进行固定上限 postprocess。

##### 起作用的原理

P3 的密集网格更适合小病斑，P5 的大感受野更适合大斑块和整叶语义，P4居中。分类与回归分头可减少两个任务的梯度冲突。depthwise 分类头降低高分辨率分支成本。one-to-many 提供丰富正样本促进训练，one-to-one 学习唯一匹配以支持 NMS-free 推理。

##### 初始化保护策略

当前基线固定 P3/P4/P5 和 reg_max=1。新增 P2 Detect 会产生极多候选并显著加重头部，应先作为独立候选而非默认开启；任何 feature source 变更都要检查 stride、通道和 bias 初始化。

##### YAML 插入点

原生末节点 23：`[[16,19,22],1,Detect,[nc]]`。各实验图允许改变输入节点编号，但应保持三尺度集合，除非专门做检测尺度消融。

##### 与基线相比唯一改变

结构实验通常只替换 Detect 的输入特征来源，不修改输出尺度、reg_max、end2end 或头部结构；若研究其中之一，应另建独立实验。

##### 预期收益及潜在代价

**预期收益：** 将多尺度表示转为分离的分类/回归预测；end-to-end 双分支兼顾训练覆盖和 NMS-free 部署。  
**潜在代价与失败模式：** 候选数随分辨率平方增长；额外检测尺度会增加显存、匹配和后处理成本；one-to-one 副本增加训练期参数与计算。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：每尺度正样本数、每尺度 Recall/AP、候选数量、box/cls loss、one-to-one 与 one-to-many 分支差异、端到端后处理延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

结构来自 `head.py::Detect` 和原生 YAML。当前仓库 `reg_max=1` 与论文描述的通用 Detect 需分开理解；最终结论以本仓库解析模型和训练日志为准。

### 10.2 本仓库基础卷积、流与融合模块

#### B01 ConvBNAct

##### 名称、位置与输入/输出尺度

**名称：** `ConvBNAct`。**所在阶段：** 所有 EdgeLite、自定义桥接和上下文支路的内部基础算子。  
**输入/输出尺度与通道：** `(B,Cin,H,W) -> (B,Cout,H/s,W/s)`，支持整数或二维 kernel、groups 与可关闭激活。

##### 目标：它要解决什么问题

为自定义模块提供行为统一、容易检查的 Conv-BN-SiLU 实现，避免每个实验分支各自拼接算子导致 padding、bias、激活和归一化不一致。它本身不是研究贡献，而是保证新增支路具有可重复数值行为的工程基元。

##### 结构图

```text
x -> Conv2d(autopad,bias=False,groups) -> BN -> SiLU/Identity -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 根据 kernel 计算 same padding，并按 stride/groups 创建无偏置卷积。
2. 卷积完成局部空间提取与通道投影。
3. BN 统一通道尺度；`act=True` 时以原位 SiLU 增加非线性，否则保留线性投影。

##### 起作用的原理

BN 已能提供仿射偏置，因此卷积不再使用 bias。关闭末层激活很重要：残差修正、门控输出或 concat 前投影如果强制经过 SiLU，会限制其表达负向校正的能力。统一 helper 也使不同自定义模块的差异更容易归因到结构本身。

##### 初始化保护策略

保持默认权重初始化与 BN；残差输出层通常 `act=False`，中间特征层通常 `act=True`。它没有独立预训练权重时，应依赖外层 identity/小门控保护。

##### YAML 插入点

不直接插入 YAML，由 LDSConv、TextureStreamP3、SemanticStreamP5、各 Bridge、P5LargeKernelContext 和 ELTEB 内部调用。

##### 与基线相比唯一改变

相对原生 `Conv` 主要是为自定义代码提供简洁接口，运算语义基本一致；不得把它的存在单独声称为精度改进。

##### 预期收益及潜在代价

**预期收益：** 统一自定义分支的数值与导出行为，减少实现偏差。  
**潜在代价与失败模式：** 每个调用仍包含 BN 与激活；过度碎片化的小卷积可能增加 kernel launch 和端侧延迟。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：融合前后数值误差、BN running statistics、调用次数、目标后端 kernel 数与单算子耗时。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::ConvBNAct`；其作用应作为支撑性实现说明，而非独立算法结论。

#### B02 DWSeparableConv

##### 名称、位置与输入/输出尺度

**名称：** `DWSeparableConv`。**所在阶段：** EdgeLite 轻量纹理、语义和下采样路径。  
**输入/输出尺度与通道：** `(B,Cin,H,W) -> depthwise:(B,Cin,H/s,W/s) -> pointwise:(B,Cout,H/s,W/s)`。

##### 目标：它要解决什么问题

把空间过滤与通道混合拆开，降低标准 k×k 卷积的参数和乘加量，使高分辨率 P3 纹理支路与多次下采样支路可在较小预算内运行。

##### 结构图

```text
x -> depthwise ConvBNAct(k,s,groups=Cin) -> pointwise ConvBNAct(1,1) -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. depthwise 卷积为每个输入通道单独学习一个空间核，不发生跨通道混合。
2. stride 可在该步完成下采样。
3. 1x1 pointwise 卷积重新组合通道并映射到 Cout。

##### 起作用的原理

标准卷积同时做空间和通道耦合，成本约为 k²CinCout；深度可分离卷积把它拆为 k²Cin + CinCout。节省在大网格上最明显，但空间核无法直接组合不同通道，因此表达上依赖后续 1x1 补偿。

##### 初始化保护策略

只在新支路或明确的 EdgeLite 替换点使用；不要把所有标准卷积一次性替换。检查端侧库对 depthwise 的实际优化，并保持输入/输出尺度与原节点一致。

##### YAML 插入点

通常由其他模块内部调用，不作为独立节点；LDSConv 是其可直接 YAML 化的 stride-2 包装。

##### 与基线相比唯一改变

相对标准 Conv，唯一变化是 `k×k full convolution` 改为 `depthwise k×k + pointwise 1×1`。

##### 预期收益及潜在代价

**预期收益：** 显著减少理论参数和 FLOPs，适合高分辨率轻量分支。  
**潜在代价与失败模式：** 通道交互被推迟，可能损失表达；某些 GPU/CPU 上小 depthwise kernel 的内存访问占主导，延迟收益小于 FLOPs 收益。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：标准/深度可分离卷积的实际 MACs、kernel launch 数、目标设备 latency、通道相关性和小病斑 AP。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::DWSeparableConv`；效率收益必须以目标设备实测确认。

#### B03 LDSConv

##### 名称、位置与输入/输出尺度

**名称：** `LDSConv`。**所在阶段：** EdgeLite 骨干和 neck 的可学习 stride-2 下采样节点。  
**输入/输出尺度与通道：** 常用 `(B,Cin,H,W) -> (B,Cout,H/2,W/2)`，实现为 stride=2 的 DWSeparableConv。

##### 目标：它要解决什么问题

用较低计算量替换标准 stride-2 Conv，同时保留可学习的空间下采样和通道投影。它试图在不使用固定池化的情况下压缩网格，并让网络决定下采样时保留哪些纹理。

##### 结构图

```text
x -> DW ConvBNAct(k=3,s=2,g=Cin) -> PW ConvBNAct(1x1) -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 每通道 3x3 depthwise 核以 stride=2 采样局部邻域。
2. BN+SiLU 稳定下采样响应。
3. 1x1 pointwise 将 Cin 混合并扩展/压缩为 Cout。

##### 起作用的原理

空间降采样发生在每个通道内部，避免高成本的跨通道 k×k 运算；随后 pointwise 恢复通道交互。与池化相比，采样核可学习；与标准 Conv 相比，容量更受限。连续多次 LDSConv 的误差会累积，因此必须关注微小病斑是否在 P3 前被削弱。

##### 初始化保护策略

按完整 EdgeLite YAML 一一对应原 stride-2 节点，输出通道和 stride 不变。初次评估应只替换 backbone 或 head 中一组节点，避免无法判断哪次下采样造成召回下降。

##### YAML 插入点

`yolo26n_edgelite*.yaml` 的骨干节点 3/5/7 及 neck 下行节点 18/21；具体编号随插入桥接模块改变，以注释和 from 索引为准。

##### 与基线相比唯一改变

标准 stride-2 Conv 改为 depthwise stride-2 + pointwise 1x1，其他拓扑保持一致。

##### 预期收益及潜在代价

**预期收益：** 减少下采样节点的参数与 FLOPs，保持可学习采样。  
**潜在代价与失败模式：** 可能削弱跨通道联合边缘提取；多次串联会影响细节；实际延迟依赖后端。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：逐节点输出能量、P2/P3 细节保留率、小目标 Recall、参数/FLOPs、CPU/GPU/NPU latency。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码与 YAML 均在 EdgeLite 实验目录中；属于本仓库实验假设，不由原论文直接验证。

#### B04 TextureStreamP3

##### 名称、位置与输入/输出尺度

**名称：** `TextureStreamP3`。**所在阶段：** EdgeLGMSFBridge 的高分辨率 P3 纹理分支。  
**输入/输出尺度与通道：** `(B,C3,H3,W3) -> (B,Cmid,H3,W3)`；640 输入通常 H3=W3=80，随后桥内再下采样到 P5。

##### 目标：它要解决什么问题

从 P3 提取病斑边缘、叶脉交界和短程纹理，并以残差方式保留压缩后的 P3 基线。它解决直接把高通道 P3 下采样到 P5成本高、且细节可能被一次投影破坏的问题。

##### 结构图

```text
x(P3) -> reduce 1x1 -> base -> DWSeparable3x3 -> DWSeparable3x3 -> texture
                              base -------------------------------------- + -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 1x1 reduce 把 P3 通道降到 c_mid，控制后续高分辨率成本。
2. 两层 stride=1 的深度可分离 3x3 提取约 5x5 有效范围内的纹理。
3. 将 texture 与 base 残差相加，输出仍为 P3 尺度。

##### 起作用的原理

连续局部卷积聚合边缘与纹理，pointwise 允许跨通道重组；残差保证如果新纹理变换不可靠，压缩后的 P3 仍能直达输出。该模块不直接产生检测特征，而是为跨尺度桥提供“近距离细节角色”的输入。

##### 初始化保护策略

保持 c_mid 小且外层有融合门控。不要在这里引入 stride，空间对齐由桥接模块的 texture_down 统一处理；初始化时保留标准 BN/SiLU。

##### YAML 插入点

不直接出现，封装在 EdgeLGMSFBridge/ResidualEdgeLGMSFBridge。输入应指向骨干 P3 而不是已经多次融合的 head P3，除非专门消融。

##### 与基线相比唯一改变

相对直接投影 P3，增加两层轻量 3x3 残差纹理精炼。

##### 预期收益及潜在代价

**预期收益：** 以较低成本强调局部病斑边界，为 P3->P5 融合提供角色明确的细节流。  
**潜在代价与失败模式：** 可能同时增强叶脉和噪声；高分辨率运行仍有激活成本；过小 c_mid 会压缩掉稀有纹理。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：纹理分支/残差范数比、边缘响应、c_mid、P3 下采样前后频谱、小病斑逐类 AP。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::TextureStreamP3`；“病斑纹理增强”是机制假设，必须由消融与可视化验证。

#### B05 SemanticStreamP5

##### 名称、位置与输入/输出尺度

**名称：** `SemanticStreamP5`。**所在阶段：** EdgeLGMSFBridge 的低分辨率 P5 语义分支。  
**输入/输出尺度与通道：** `(B,C5,H5,W5) -> (B,Cmid,H5,W5)`；640 输入常为 20x20。

##### 目标：它要解决什么问题

把 P5 的整叶片、光照和大范围病斑语义压缩到与纹理流相同的 c_mid 通道，并通过较大 5x5 depthwise 卷积补充局部上下文，作为融合中的“远距离语义角色”。

##### 结构图

```text
x(P5) -> reduce 1x1 -> base -> DWSeparable5x5 -> ConvBNAct1x1 -> semantic
                              base ----------------------------------- + -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 1x1 将 C5 映射到 c_mid。
2. 5x5 depthwise 可在小 P5 网格上覆盖更宽邻域。
3. 1x1 再次混合通道。
4. 与 base 相加，得到受保护的语义流。

##### 起作用的原理

P5 网格小，因此 5x5 空间核的相对感受野很大，却保持较低成本。残差保留来自 C2PSA 的预训练语义；新分支只学习修正。压缩到与 P3 流一致的通道后，后续按通道二路 softmax 才有可比基础。

##### 初始化保护策略

输入必须来自保留的 P5/C2PSA 或受保护 P5LargeKernelContext；不应绕过原生 C2PSA。外层桥接如果已有强全局分支，应单独消融避免上下文重复。

##### YAML 插入点

不直接出现，由 EdgeLGMSFBridge 系列内部创建。

##### 与基线相比唯一改变

相对简单 P5 1x1 投影，增加 5x5 depthwise+1x1 的残差语义修正。

##### 预期收益及潜在代价

**预期收益：** 为融合提供上下文角色明确、通道匹配的 P5 流。  
**潜在代价与失败模式：** 可能与 SPPF/C2PSA/P5LargeKernelContext 功能重叠；重复上下文会增加延迟或过平滑。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：语义分支残差比、5x5 分支响应、与 C2PSA 输出相似度、全局误检率和 P5 latency。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::SemanticStreamP5`；有效性需与仅 1x1 P5 投影对照。

#### B06 FastNormFuse2

##### 名称、位置与输入/输出尺度

**名称：** `FastNormFuse2`。**所在阶段：** P5ToP3SemanticFuse 的无空间自适应双流加权器。  
**输入/输出尺度与通道：** 两路同形 `(B,C,H,W),(B,C,H,W) -> (B,C,H,W)`；仅有两个可学习标量。

##### 目标：它要解决什么问题

用极低参数成本学习两路特征的全局混合比例，避免固定 0.5 相加，也避免通道级注意力的额外卷积。

##### 结构图

```text
raw w=[w1,w2] -> ReLU -> normalize by sum+eps -> a1,a2
y = a1*x1 + a2*x2
```

##### 逐步 Flow：数据到底怎样经过它

1. 对两个标量权重做 ReLU，阻止负权重直接反相特征。
2. 用权重和加 eps 归一化，使总量接近 1。
3. 对两路同形张量加权求和。

##### 起作用的原理

归一化权重让融合尺度稳定，学习过程只决定整体更依赖哪一路。它没有样本、通道或位置条件，因此表达弱但稳定，适合做 P5 语义反馈到 P3 的保守基线。

##### 初始化保护策略

初始化两个权重均为 1，起点近似等权。保证两路已对齐且统计范围相近；eps 防止权重都被 ReLU 截为零时除零。

##### YAML 插入点

不直接插入，由 `P5ToP3SemanticFuse` 内部调用。

##### 与基线相比唯一改变

相对固定相加，唯一变化是引入两个归一化可学习标量。

##### 预期收益及潜在代价

**预期收益：** 几乎无参数和计算，可观测全局偏好，导出简单。  
**潜在代价与失败模式：** 不能按图像或通道动态选择，可能在类别/场景差异大时欠拟合。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：归一化 w1/w2、权重稳定性、两流方差、与简单相加/concat 的精度延迟对比。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::FastNormFuse2`；它是工程折中，不应等同于通道注意力。

#### B07 RoleAwareAttnFuse2

##### 名称、位置与输入/输出尺度

**名称：** `RoleAwareAttnFuse2`。**所在阶段：** P3 纹理流与 P5 语义流对齐后的通道级融合器。  
**输入/输出尺度与通道：** 两路同形 `(B,C,H,W)` -> `(B,C,H,W)`；内部描述张量为 `(B,4C,1,1)`，输出权重 `(B,2,C,1,1)`。

##### 目标：它要解决什么问题

让融合权重依据两路的内容、差异和一致性按样本与通道动态变化，而不是用一个全局标量。它要区分“这个通道更应相信细节流还是语义流”。

##### 结构图

```text
cat(x1,x2,|x1-x2|,x1*x2) -> GAP -> 1x1 reduce -> SiLU -> 1x1 -> logits(2,C)
softmax over role -> w1,w2 -> w1*x1+w2*x2
```

##### 逐步 Flow：数据到底怎样经过它

1. 构造 x1、x2、绝对差和乘积四类关系特征。
2. 全局平均池化把空间维压为 1x1，得到样本级通道摘要。
3. 两层 1x1 MLP 从 4C 压缩到 hidden，再投影到 2C logits。
4. 按二路角色维做 softmax，保证每通道 `w1+w2=1`。
5. 广播权重到 H、W 并求加权和。

##### 起作用的原理

x1/x2 表示各自强度，绝对差表示分歧，乘积表示共现；融合器据此学习每个通道的角色选择。空间池化避免生成昂贵的像素级权重，适合轻量桥接。softmax 保持凸组合，防止无界放大。输出投影零初始化时 logits=0，初始权重严格 0.5/0.5，训练起点中性。

##### 初始化保护策略

`channel_proj` 权重与偏置全零初始化是关键保护。两路必须同通道、同空间；若统计差异过大，应先各自投影/归一化。监控权重是否塌缩到单一路。

##### YAML 插入点

不直接出现，由 DistanceAwareP2P5Bridge、EdgeLGMSFBridge 和 ResidualEdgeLGMSFBridge 内部调用。

##### 与基线相比唯一改变

相对 FastNormFuse2，从两个全局标量升级为按样本、按通道的内容条件二路 softmax。

##### 预期收益及潜在代价

**预期收益：** 能根据图像与通道自适应选择纹理或语义，且不产生 HxW 注意力图。  
**潜在代价与失败模式：** 增加池化和小型 MLP；GAP 丢失空间位置，权重可能塌缩或学习到数据偏差。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：P3/P5 平均权重、每通道方差、归一化熵、类别条件权重、融合前后余弦相似度与延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::RoleAwareAttnFuse2`；其解释来自张量构造与初始化，可通过监控版模块直接验证。

#### B08 MonitoredRoleAwareAttnFuse2

##### 名称、位置与输入/输出尺度

**名称：** `MonitoredRoleAwareAttnFuse2`。**所在阶段：** ResidualEdgeLGMSFBridge 内部的可诊断 RoleAware 融合器。  
**输入/输出尺度与通道：** 前向输出与 RoleAwareAttnFuse2 相同；额外维护两个权重和、熵和、step 数三个非持久 buffer。

##### 目标：它要解决什么问题

在不改变融合结果的前提下记录模型实际怎样使用两路特征，从而判断桥接是否学习到有意义的互补，还是长期等权、完全塌缩或只依赖一路。

##### 结构图

```text
RoleAware forward -> weights
                    |-> no_grad: mean role weights + normalized binary entropy + steps
                    \-> weighted output
```

##### 逐步 Flow：数据到底怎样经过它

1. 按 RoleAwareAttnFuse2 计算二路通道权重。
2. 在 no_grad 中转为 FP32，跨 batch、通道和空间求平均。
3. 计算二元权重熵并除以 log(2)，得到 0-1 归一化熵。
4. 累加统计而不进入 state_dict；forward 输出保持原公式。
5. 每个 epoch 可调用 summary 后 reset。

##### 起作用的原理

权重均值告诉我们总体偏向哪一路；熵接近 1 表示接近均匀，接近 0 表示强选择。将监控置于 no_grad 且 buffer persistent=False，可避免它影响梯度、权重文件与推理结果。它不是新注意力算法，而是实验可解释性工具。

##### 初始化保护策略

必须在固定统计窗口 reset，否则不同 epoch 混合。分布式训练时需要明确是否跨 rank 汇总；监控值不能替代精度指标，只用于解释。

##### YAML 插入点

不直接插入，由 ResidualEdgeLGMSFBridge 固定使用。

##### 与基线相比唯一改变

相对 RoleAwareAttnFuse2 只增加 no_grad 统计，不改变训练图中的融合数学。

##### 预期收益及潜在代价

**预期收益：** 提供融合是否工作、是否塌缩的直接证据，便于筛除无效桥接。  
**潜在代价与失败模式：** 有少量统计开销；分布式汇总和窗口定义不清会产生误导。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：epoch 级 p3_weight、p5_weight、entropy、steps、跨种子一致性及其与验证 AP 的相关性。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::MonitoredRoleAwareAttnFuse2`；结论应与消融和特征可视化联合解释。

#### B09 SimAM

##### 名称、位置与输入/输出尺度

**名称：** `SimAM`。**所在阶段：** 可选地放在 EdgeLGMSFBridge 融合后、输出投影前。  
**输入/输出尺度与通道：** `(B,C,H,W) -> (B,C,H,W)`，无可训练参数，仅有超参数 `e_lambda`。

##### 目标：它要解决什么问题

根据单个神经元相对同通道空间均值的显著程度重新缩放激活，试图抑制复杂背景并突出稀有局部响应。

##### 结构图

```text
mu=mean_spatial(x); d2=(x-mu)^2
var=d2.sum/(H*W-1); energy=d2/(4*(var+lambda))+0.5
y=x*sigmoid(energy)
```

##### 逐步 Flow：数据到底怎样经过它

1. 对每个 batch、通道计算空间均值。
2. 计算每个位置到均值的平方差。
3. 用同通道方差与 e_lambda 归一化，构成能量值并加 0.5。
4. sigmoid 产生 0-1 权重，与原特征逐元素相乘；H*W<=1 时直接返回。

##### 起作用的原理

在同一通道内，偏离均值更大的位置获得不同能量响应，可视作无需参数的空间显著性度量。它不会跨通道学习，也不利用标签；因此成本低，但可能把叶脉、反光和噪声一并当作显著点。

##### 初始化保护策略

默认 `use_simam=False`，只作为独立开关消融。不要同时打开 BiBridge 或更换融合器后把结果归因于 SimAM。检查 FP16 下方差与 e_lambda 的稳定性。

##### YAML 插入点

通过 EdgeLGMSFBridge 参数 `[c_out, use_simam, reduction]` 的布尔位开启；`yolo26n_edgelite_simam*.yaml` 为示例。

##### 与基线相比唯一改变

在融合结果与 out 1x1 之间插入一个参数自由的逐元素重标定。

##### 预期收益及潜在代价

**预期收益：** 不增加训练参数，可能抑制均匀背景并突出局部异常。  
**潜在代价与失败模式：** 增加均值、平方、sigmoid 等内存算子；显著性不等同于病斑，可能放大噪声，端侧延迟也可能上升。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：e_lambda、激活稀疏度、背景误检、病斑/叶脉响应比、目标设备 elementwise latency。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::SimAM`；本仓库已有开/关 YAML，必须用相同桥接图比较。

### 10.3 本仓库上下文与桥接模块

#### C01 NearDetailConvBlock

##### 名称、位置与输入/输出尺度

**名称：** `NearDetailConvBlock`。**所在阶段：** P2/P3 近距离细节源，DistanceAwareP2P5Bridge 的 near 分支。  
**输入/输出尺度与通道：** `(B,Cin,H,W) -> (B,Cout,H,W)`；P2 常为 160x160，P3 常为 80x80（640 输入）。

##### 目标：它要解决什么问题

显式增强小病斑边界、细线状纹理和方向性结构，同时用可学习小门控限制新支路在训练初期的影响。该块解决浅层特征细节丰富但噪声也多、直接大幅改写容易损害预训练表示的问题。

##### 结构图

```text
x -> proj1x1 -> base
base -> RepConv3x3 -------------------\
base -> Conv1x3 -> Conv3x1 ---------- cat -> mix1x1(no act) -> detail
base --------------------------------/
y = base + sigmoid(gate)*detail
```

##### 逐步 Flow：数据到底怎样经过它

1. 1x1 投影把 Cin 对齐到 Cout，形成受保护 base。
2. RepConv 3x3 提取普通局部斑点与角点。
3. 1x3 后接 3x1，分解地响应横向和纵向延展结构。
4. 将 local、line、base 三路拼接并用无激活 1x1 混合为 detail。
5. 以 sigmoid(gate) 缩放 detail 后加回 base；默认 gate=-1，对应初始系数约 0.269。

##### 起作用的原理

标准 3x3 对各向同性局部结构敏感，1x3+3x1 更容易编码方向边缘；三路 concat 让网络保留原始投影并选择两类细节。门控把模块表示为 `base + alpha*detail`，alpha 有界，避免未训练的 detail 在第一步完全覆盖基线。RepConv 若支持部署重参数化，还可在推理时减少分支开销。

##### 初始化保护策略

保留 base 残差与负值 gate 初始化。若将 gate 设为更小值，应单独记录有效初始系数；P2 上使用时尤其要监控显存与噪声放大。

##### YAML 插入点

距离感知实验节点 3（P2）和 6（P3），例如 `[-1,1,NearDetailConvBlock,[256]]`；桥内还会实例化一个压缩到 c_mid 的 near block。

##### 与基线相比唯一改变

相对普通 C3k2/Conv，在指定 P2/P3 节点增加局部、方向双支路与有界残差门控；其余骨干层保持不变。

##### 预期收益及潜在代价

**预期收益：** 加强微小、线状和边界模糊病斑的局部证据，并允许从基线平滑学习。  
**潜在代价与失败模式：** 高分辨率三分支增加激活和延迟，叶脉/反光也可能被增强；多处重复插入会造成冗余。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：有效 gate、local/line/base 分支范数、边缘型类别 AP、小目标 Recall、P2 峰值显存与重参数化前后延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::NearDetailConvBlock`；病斑方向纹理作用属于针对数据域的设计假设。

#### C02 _FactorizedLargeKernelConv

##### 名称、位置与输入/输出尺度

**名称：** `_FactorizedLargeKernelConv`。**所在阶段：** FarContextConvBlock 的普通分组数为 1 的大核分支。  
**输入/输出尺度与通道：** `(B,C,H,W) -> (B,C,H,W)`；`1xK` 后接 `Kx1`，当前常用 K=7。

##### 目标：它要解决什么问题

用两个一维方向卷积近似 KxK 大感受野，降低直接 KxK 全卷积的参数和计算，同时保留跨通道混合能力。

##### 结构图

```text
x -> ConvBNAct(1xK) -> ConvBNAct(Kx1) -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 1xK 横向聚合长距离邻域并完成全通道卷积。
2. Kx1 纵向聚合，使组合后的理论感受野覆盖 KxK。
3. 两步均带 BN+SiLU，形成两次非线性而非严格线性可分解核。

##### 起作用的原理

参数量从约 K²C² 降为 2KC²；两次非线性使它不仅是矩阵分解，还能学习方向组合。它适合小 P5 网格，因为 K=7 已覆盖较大比例空间。与 depthwise 版本不同，这里每一步都混合通道，表达更强但成本更高。

##### 初始化保护策略

只作为 FarContext 的单个候选分支，不直接替代全网卷积。K 取奇数以保持对称 padding；改变 K 时锁定其他分支和 selector。

##### YAML 插入点

不直接插入，由 FarContextConvBlock 的 `large_kernel` 参数控制。

##### 与基线相比唯一改变

相对直接 KxK full Conv，改为 1xK + Kx1；相对 3x3 分支，只扩大感受野。

##### 预期收益及潜在代价

**预期收益：** 用较少成本获得大范围方向上下文，同时维持通道耦合。  
**潜在代价与失败模式：** 仍是全通道卷积，C 较大时昂贵；两次 kernel launch 可能抵消理论节省。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：K、参数/MACs、有效感受野、横纵方向响应差异和目标设备算子融合情况。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::_FactorizedLargeKernelConv`，不作为独立 YAML 模块。

#### C03 FarContextConvBlock

##### 名称、位置与输入/输出尺度

**名称：** `FarContextConvBlock`。**所在阶段：** P5 远距离上下文源，DistanceAwareP2P5Bridge 的 far 分支。  
**输入/输出尺度与通道：** `(B,Cin,H5,W5) -> (B,Cout,H5,W5)`；四分支均保持 P5 尺度。

##### 目标：它要解决什么问题

同时建模局部邻域、大核上下文、空洞上下文和全局通道语义，并按样本选择四路权重。它针对的是 P5 单一路径可能偏向某一感受野，无法适应不同病斑尺度和光照背景的问题。

##### 结构图

```text
x -> reduce -> base
base -> 3x3 local --------------------\
base -> 1xK+Kx1 large -----------------\
base -> dilated3x3(d=2)+1x1 ------------ selector(GAP+MLP)->softmax4 -> weighted sum -> out1x1 -> +base
base -> GAP+1x1+SiLU -> expand global --/
```

##### 逐步 Flow：数据到底怎样经过它

1. 1x1 将输入对齐到 Cout，形成 base。
2. 并行生成 local、factorized large、dilated 和全局池化扩展四路。
3. 将四路 concat 后做 GAP，selector 输出每个样本 4 个 logits。
4. softmax 在分支维归一化，得到总和为 1 的权重。
5. 加权和经无激活 1x1 输出，再与 base 残差相加。

##### 起作用的原理

四路对应互补感受野：3x3 保留局部，1xK/Kx1 捕获连续大范围，dilation 在不增核大小时跨步采样，全局池化提供整图通道偏置。selector 只产生每样本四个标量，成本低但能根据场景选择。残差保证多分支输出只是语义修正。

##### 初始化保护策略

保持 base 残差。先比较各单分支，再比较 selector 融合；否则无法知道收益来自大核还是自适应选择。注意该模块没有外层小 gate，直接插入 P5 时比 P5LargeKernelContext 更激进。

##### YAML 插入点

距离感知实验 P5 节点 13：`[-1,1,FarContextConvBlock,[1024,7]]`；也由 DistanceAwareP2P5Bridge 内部用压缩通道实例化。

##### 与基线相比唯一改变

相对原生 P5 C2PSA 输出，在指定位置增加一个四分支残差上下文块。

##### 预期收益及潜在代价

**预期收益：** 覆盖从局部到全局的多种病斑/叶片上下文，并按样本自适应选择。  
**潜在代价与失败模式：** 四分支带来较多计算和激活；上下文与 SPPF/C2PSA 重叠；全局池化可能使局部定位变钝。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：四路 softmax 权重、分支消融、K/dilation、P5 特征相似度、全局背景误检与 latency。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::FarContextConvBlock`；选择器是否学到有意义路由需记录权重分布验证。

#### C04 _DepthwiseFactorizedLargeKernelConv

##### 名称、位置与输入/输出尺度

**名称：** `_DepthwiseFactorizedLargeKernelConv`。**所在阶段：** P5LargeKernelContext 的高效大核分支。  
**输入/输出尺度与通道：** `(B,C,H,W) -> (B,C,H,W)`；逐通道 `1xK` 与 `Kx1`，K 必须为不小于 7 的奇数。

##### 目标：它要解决什么问题

在 P5 上提供大感受野空间过滤，同时把成本限制在逐通道级别，为 13x13 等大核实验提供可部署的近似。

##### 结构图

```text
x -> depthwise ConvBNAct(1xK,g=C) -> depthwise ConvBNAct(Kx1,g=C) -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 构造时验证 K>=7 且为奇数。
2. 每个通道独立做横向大核过滤。
3. 再独立做纵向大核过滤，空间覆盖组合为 KxK。

##### 起作用的原理

参数约为 2KC，而非 K²C²；它只改变每个通道的空间分布，不负责跨通道混合，跨通道关系由上下文块的 selector 和 out 1x1 完成。K=13 在 20x20 P5 上已经接近全图范围，因而可捕获叶片级结构。

##### 初始化保护策略

限制 K 范围并保持后续 1x1 通道混合。不要把该 depthwise 分支的理论 FLOPs 直接当作端侧收益，需检查后端对超长 depthwise kernel 的实现。

##### YAML 插入点

不直接插入；由 P5LargeKernelContext 的 kernel_size 参数控制。

##### 与基线相比唯一改变

相对普通 factorized large-kernel，把 groups 从 1 改为 C，取消大核阶段的跨通道混合。

##### 预期收益及潜在代价

**预期收益：** 极低参数成本下扩大 P5 空间感受野。  
**潜在代价与失败模式：** 表达依赖后续 pointwise；大核 depthwise 在部分设备上内存效率差。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：K、理论/实测延迟、kernel 实现、各通道响应平滑度、与 full factorized 分支的 AP 差异。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::_DepthwiseFactorizedLargeKernelConv`。

#### C05 P5LargeKernelContext

##### 名称、位置与输入/输出尺度

**名称：** `P5LargeKernelContext`。**所在阶段：** 历史最佳图中 C2PSA 后、P3->P5 桥前的受保护 P5 上下文增量。  
**输入/输出尺度与通道：** `(B,C1,H5,W5) -> (B,C2,H5,W5)`；常用 C2=256（缩放后）、K=13，输出不改 P5 尺度。

##### 目标：它要解决什么问题

在不替换原生 SPPF+C2PSA 的前提下，增加局部、超大核和全局三类上下文，并用很小的有界残差门控保护预训练 P5。它专门解决“想扩大 P5 感受野，但不能让随机初始化新分支一开始破坏强基线”的问题。

##### 结构图

```text
x -> shortcut -> base
base -> DW3x3 local -------------------\
base -> DW(1xK)+DW(Kx1) large ---------- selector(GAP+MLP)->softmax3 -> sum -> out1x1(no act) -> *sigmoid(gate) -> +base
base -> GAP+1x1 -> expand global -------/
```

##### 逐步 Flow：数据到底怎样经过它

1. shortcut 为 identity 或无激活 1x1，建立基线 base。
2. 并行计算 local、K=13 depthwise factorized large 和 global 三路。
3. concat 三路后 GAP，selector 为每个样本产生 3 个权重并 softmax。
4. 加权结果经无激活 1x1 做通道混合。
5. 乘 `sigmoid(gate)` 后加回 base；默认 gate=-3，初始系数约 0.047。

##### 起作用的原理

K=13 在 20x20 网格上覆盖大范围，global 分支提供全图统计，local 分支防止只剩平滑上下文。selector 允许图像级选择；最重要的是 gate 让整个新上下文路径初始只有约 4.7% 幅度，因此模型开始时接近预训练基线，再逐步决定是否使用。

##### 初始化保护策略

必须放在 C2PSA 后并保留 shortcut、小 gate。记录 effective_gate 曲线；若 gate 始终很小，说明分支可能无贡献。K、reduction、gate_init 应分别消融。

##### YAML 插入点

`yolo26n_edgelite_elteb_p5lk13.yaml` 节点 11：`[-1,1,P5LargeKernelContext,[1024,13,4,-3.0]]`，随后 EdgeLGMSFBridge 使用增强 P5。

##### 与基线相比唯一改变

相对历史 ELTEB+EdgeLite 图，只在 C2PSA 后新增受门控三分支上下文；保留 ELTEB 与单向 P3->P5 融合。

##### 预期收益及潜在代价

**预期收益：** 在强 P5 基线上渐进加入叶片级上下文，兼顾局部与全局。  
**潜在代价与失败模式：** 增加三分支、selector 和大核算子；可能与 SPPF/C2PSA 重复，gate 过小也可能学不到。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：effective_gate、三路权重、K/reduction、P5 余弦相似度、全局型类别 AP、峰值显存和端侧延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码与历史 YAML 均可核对；这是本仓库已实现路线，是否继续保留必须由当前统一配方的三种子实验决定。

#### C06 DistanceAwareP2P5Bridge

##### 名称、位置与输入/输出尺度

**名称：** `DistanceAwareP2P5Bridge`。**所在阶段：** P2 近细节与 P5 远上下文在 P5 尺度的跨距离融合。  
**输入/输出尺度与通道：** 输入 `[P2:(B,C2,H/4,W/4), P5:(B,C5,H/32,W/32)]`；输出 `(B,Cout,H/32,W/32)`。P2 分支连续三次 stride-2。

##### 目标：它要解决什么问题

把最浅的微小病斑细节送到最深 P5 语义层，但不增加 P2 检测头。它试图保留 P2 作为信息源的价值，同时避免在 P2 网格上直接预测所带来的候选数量和计算爆炸。

##### 结构图

```text
P2 -> NearDetail(cmid) -> stride2 x3 -> near(P5 scale) --\
P5 -> FarContext(cmid) ------------------------------- RoleAwareFuse -> out1x1 -> P5 enhanced
```

##### 逐步 Flow：数据到底怎样经过它

1. P2 先经 NearDetailConvBlock 压到 c_mid 并提取局部/方向纹理。
2. 三层 3x3 stride-2 ConvBNAct 依次 P2->P3->P4->P5，逐步而非一次性对齐。
3. P5 经 FarContextConvBlock 生成 c_mid 的多感受野语义。
4. 若由于非标准输入造成尺寸差，near 用 nearest 精确对齐 far。
5. RoleAwareAttnFuse2 按样本/通道融合 near 与 far，out 1x1 输出 Cout。

##### 起作用的原理

逐级下采样允许浅层信息在每一步学习该保留什么，最终与 P5 在同一坐标系融合。RoleAware 使用强度、差异和共现决定每通道依赖近细节还是远语义。该模块输出在 P5 尺度，因此不会增加 Detect 的位置数；但 P2 路径本身仍在高分辨率上计算。

##### 初始化保护策略

保留 Detect(P3,P4,P5)，不要重新启用 P2 Detect。桥接输出应作为单独增强源与原 P5 路径有清晰拼接/替换点；检查三次 stride-2 后的对齐。

##### YAML 插入点

距离感知 YAML 节点 14：`[[3,13],1,DistanceAwareP2P5Bridge,[256,4,7]]`，from 分别指 P2 detail 与 P5 far source。

##### 与基线相比唯一改变

相对修剪后的基线，只增加 P2->P5 信息桥；检测尺度集合仍为 P3/P4/P5。

##### 预期收益及潜在代价

**预期收益：** 利用极浅细节增强深层语义，而不支付 P2 检测头的巨大候选成本。  
**潜在代价与失败模式：** P2 高分辨率处理和三次下采样仍昂贵；长路径可能把噪声送入 P5；模块内部同时含 Near、Far、Fuse，归因复杂。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：P2 分支 FLOPs/延迟、near/far 权重熵、逐级下采样信息保留、P3小目标 Recall、P5误检和端到端速度。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码与 `yolo26n_p2p5_distanceconv_experiment` YAML 可核对；历史结果提示 P2 直接检测成本过高，因此本模块必须与无 P2 Detect 基线比较。

#### C07 EdgeLGMSFBridge

##### 名称、位置与输入/输出尺度

**名称：** `EdgeLGMSFBridge`。**所在阶段：** EdgeLite/ELTEB 的单向 P3 纹理到 P5 语义融合桥。  
**输入/输出尺度与通道：** 输入 `[P3:(B,C3,H/8,W/8),P5:(B,C5,H/32,W/32)]`；输出 `(B,Cout,H/32,W/32)`。

##### 目标：它要解决什么问题

把 P3 的病斑纹理经过轻量处理与两次可学习下采样送到 P5，与低分辨率语义按通道动态融合，为 neck 的 P5 路径提供细节补充。

##### 结构图

```text
P3 -> TextureStream(cmid) -> LDSConv s2 -> LDSConv s2 -> texture(P5) --\
P5 -> SemanticStream(cmid) ------------------------------------------ RoleAwareFuse -> optional SimAM -> out1x1 -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. P3 经 TextureStream 提取局部残差纹理。
2. 两次 LDSConv 将 P3 从 stride 8 对齐到 stride 32。
3. P5 经 SemanticStream 生成同通道语义。
4. 若尺寸仍不一致，nearest 调整 texture。
5. RoleAware 按通道融合；可选 SimAM 重标定空间；1x1 输出 Cout。

##### 起作用的原理

桥接把细节与语义先分别建模，再对齐和选择，避免直接 concat 高维特征。P3 细节可弥补 P5 下采样造成的局部丢失，P5 语义可抑制纹理流把叶脉当病斑。RoleAware 的零初始化输出投影使初始融合为等权，但此基础版本没有外层 P5 shortcut，因此新桥输出会直接替代/参与后续融合，保护弱于 Residual 版本。

##### 初始化保护策略

首次训练优先使用 ResidualEdgeLGMSFBridge；若评估本基础版，保持 use_simam=False 并与相同图的残差版对照。核对 from 索引和 P3/P5 尺度。

##### YAML 插入点

EdgeLite 多个 YAML 的节点 11/12，典型 `[[4,10],1,EdgeLGMSFBridge,[256,False,8]]`，随后在 neck P5 concat 使用其输出。

##### 与基线相比唯一改变

相对原生图新增单向 P3->P5 桥，并在后续 P5 concat 中加入增强源；不包含 P5->P3 反馈。

##### 预期收益及潜在代价

**预期收益：** 以压缩通道和轻量下采样把细节送入 P5，可选背景抑制。  
**潜在代价与失败模式：** 没有外层基线 shortcut，随机新分支影响较强；两次下采样与多模块串联增加延迟；SimAM 可能放大噪声。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：RoleAware 权重、P3/P5 分支范数、use_simam、reduction、桥输出与原 P5 相似度、P5 concat 峰值显存。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码与 `edgelite_experiment/configs` 可核对；效果需要对同一训练配方的原生、基础桥和残差桥三方比较。

#### C08 ResidualEdgeLGMSFBridge

##### 名称、位置与输入/输出尺度

**名称：** `ResidualEdgeLGMSFBridge`。**所在阶段：** 受保护的单向 P3->P5 桥，当前更适合作为基线兼容候选。  
**输入/输出尺度与通道：** 输入 P3/P5，输出 `(B,Cout,H/32,W/32)`；shortcut 将原 P5 直接投影到 Cout。

##### 目标：它要解决什么问题

保留 EdgeLGMSFBridge 的纹理-语义融合能力，同时显式保留原 P5 基线并用小门控控制新增修正，解决基础桥随机初始化时可能破坏预训练 P5 的问题。

##### 结构图

```text
P5 -> shortcut1x1(no act) -> base ----------------------------------------------\
P3 -> Texture -> LDSx2 --\
P5 -> Semantic ----------- MonitoredRoleAware -> optional SimAM -> out(no act) -> *sigmoid(gate) -> + -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. P5 通过 identity 或无激活 1x1 得到 base。
2. P3/P5 分别经 TextureStream、SemanticStream，并在 P5 尺度对齐。
3. MonitoredRoleAware 计算融合，同时累计两路均值权重和熵。
4. 可选 SimAM 后，无激活 out 投影到 Cout。
5. 以 `sigmoid(gate)` 缩放 refine，再加到 base；默认 gate=-3，约 0.047。

##### 起作用的原理

这是典型 residual adapter：强基线走短路，新结构只提供有界增量。小 gate 使初始化输出接近原 P5，预训练权重可继续发挥；训练若发现桥有用，gate 与内部权重共同增加贡献。监控权重帮助判断细节/语义是否真的互补。

##### 初始化保护策略

保持 shortcut、无激活 out 与 gate_init=-3。记录 gate 和熵；若 gate 不增长或权重长期等分，桥可能没有学到额外价值。SimAM 必须独立开关。

##### YAML 插入点

`lasttrain/configs/yolo26n_map50_v2_bounded_elteb_residual_p3p5.yaml` 节点 11：`[[4,10],1,ResidualEdgeLGMSFBridge,[256,False,8,-3.0]]`。

##### 与基线相比唯一改变

相对基础 EdgeLGMSFBridge，唯一结构增量是 P5 shortcut、小门控和监控统计；相对原生基线则是受保护的 P3->P5 残差适配器。

##### 预期收益及潜在代价

**预期收益：** 在接近基线的初始化下学习跨尺度细节修正，风险和可解释性优于直接替代。  
**潜在代价与失败模式：** 保留 shortcut 与新分支使参数/计算都存在；门控过小可能学习慢，过大则失去保护。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：effective_gate、p3/p5 weight、entropy、桥前后特征差、梯度范数、三种子精度与目标设备延迟。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码、YAML 和监控接口均可核对；它是当前仓库自定义模块，原论文仅提供单因素/基线保护的写作范式。

#### C09 P5ToP3SemanticFuse

##### 名称、位置与输入/输出尺度

**名称：** `P5ToP3SemanticFuse`。**所在阶段：** BiBridge 的反向 P5 语义反馈到 P3 支路。  
**输入/输出尺度与通道：** 输入 P3/P5，P5 上采样到 P3；输出 `(B,Cout,H/8,W/8)`，默认 Cout=C3。

##### 目标：它要解决什么问题

把整叶片与高层类别语义反馈给高分辨率 P3，使小病斑定位不只依赖局部纹理，并通过小门控避免深层语义过早覆盖 P3 边界。

##### 结构图

```text
P3 -> reduce1x1 -> p3_sem --\
P5 -> reduce1x1 -> upsample to P3 -- FastNormFuse2 -> out1x1 -> refine -> *sigmoid(gate)
P3 -> shortcut ----------------------------------------------- + -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. P3、P5 分别压到 c_mid。
2. P5 用 nearest 上采样到 P3 尺度。
3. FastNormFuse2 学习两路全局归一化比例。
4. out 1x1 投影到 Cout。
5. shortcut 保留原 P3，refine 乘 `sigmoid(init_gate)` 后相加；默认 -2 约 0.119。

##### 起作用的原理

反馈路径将高层语义作为 P3 的低幅修正，可抑制局部纹理误检；FastNormFuse2 只学两标量，避免高分辨率上的重注意力。门控与 shortcut 保护 P3 定位能力。但双向桥会形成更多跨尺度依赖，可能让 P3/P5 特征趋同并增加优化耦合。

##### 初始化保护策略

在单向 P3->P5 桥已经验证后才独立开启；保持 init_gate=-2 和 shortcut。不要与 SimAM、ELTEB 变化同时上线。记录 gate 是否增长与 P3 边界精度是否下降。

##### YAML 插入点

BiBridge YAML 节点 12：`[[4,10],1,P5ToP3SemanticFuse,[128,8]]`；随后上采样路径应使用增强 P3。

##### 与基线相比唯一改变

相对单向桥唯一新增 P5->P3 反馈支路；其他结构保持一致。

##### 预期收益及潜在代价

**预期收益：** 为小病斑特征注入全局语义，可能降低纹理型误检。  
**潜在代价与失败模式：** 高分辨率融合增加激活和延迟；可能过平滑 P3；双向依赖加大归因难度，历史图中该支路已关闭。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：effective gate、FastNorm 两权重、P3边界/小病斑 AP、P3/P5相似度、BiBridge 增量延迟与显存。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码与 BiBridge YAML 可核对；当前历史最佳图明确未包含该反馈，因此应视为退役候选，除非重新单因素验证。

### 10.4 本仓库 ELTEB 纹理模块

#### D01 _ELTEBTexture

##### 名称、位置与输入/输出尺度

**名称：** `_ELTEBTexture`。**所在阶段：** ELTEB/ELTEBLite 内部的原图纹理构造与 P3 融合器。  
**输入/输出尺度与通道：** 输入 `[image:(B,Cimg,H0,W0), P3:(B,Cp3,H3,W3)]`；先把 image 双线性缩放到 H3xW3，输出保持 `(B,Cp3,H3,W3)`。

##### 目标：它要解决什么问题

把输入图像中可解释的灰度、梯度、二阶边缘和锐化线索直接送到 P3，弥补连续 stride-2 卷积可能削弱微小病斑纹理的问题。它不是替换学习特征，而是形成一个可控的早期纹理残差。

##### 结构图

```text
image -> resize to P3 -> [RGB, gray, Sobel magnitude, |Laplacian|, unsharp] -> cat
      -> compress1x1 -> project1x1(no act) -> texture -> *fusion_alpha
P3 -------------------------------------------------------------- add / concat(identity-init) -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 原图若与 P3 尺寸不同，用 bilinear、align_corners=False 对齐。
2. 按 0.299R+0.587G+0.114B 计算灰度。
3. 固定 Sobel-x/y 求梯度幅值并 tanh；固定 Laplace 求绝对二阶边缘并 tanh。
4. full 模式另以 5x5 平均模糊构造 `unsharp=img+alpha*(img-blur)`，再与 img/gray/sobel/lap 拼接。
5. 1x1 compress 到 texture_channels，再无激活投影到 Cp3。
6. add 模式做 `P3+fusion_alpha*texture`；concat 模式拼接后 1x1，且初始化为原 P3 identity。

##### 起作用的原理

Sobel 对一阶边缘敏感，Laplacian 对快速变化和细小斑点敏感，unsharp 提高高频对比；这些固定算子给网络一个明确的纹理先验。后续 1x1 学习哪些线索与通道有用。关键保护是 fusion_alpha 初值 0：add 模式初始精确等于 P3；concat 模式的 fuse 权重也初始化为只复制原 P3，因此新增分支开始时不改变基线输出。

##### 初始化保护策略

保留 fusion_alpha=0 和 concat identity 初始化。原图归一化范围必须与训练预处理一致；固定核作为 buffer 随设备/dtype 转换。监控 fusion_alpha，防止无界增大或反向符号异常。

##### YAML 插入点

不独立出现；由 ELTEB/ELTEBLite 创建。外层 YAML 必须把原图节点和 P3 前一层同时作为输入。

##### 与基线相比唯一改变

相对标准 P3 C3k2，只增加一条原图固定纹理先验支路，并以零起点残差/identity concat 融合。

##### 预期收益及潜在代价

**预期收益：** 显式保留微小病斑边缘与高频信息，初始化精确保护基线，线索具有可解释性。  
**潜在代价与失败模式：** 固定边缘算子也会响应叶脉、反光和压缩噪声；原图旁路增加内存与数据依赖；无界 alpha 可能过度锐化。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：fusion_alpha、unsharp_alpha、各纹理图能量、病斑/叶脉响应比、P3边界 AP、原图旁路 latency。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

所有公式和初始化来自 `block.py::_ELTEBTexture`；对病斑有效仍需按 full/lite、add/concat 单因素比较。

#### D02 _BoundedELTEBTexture

##### 名称、位置与输入/输出尺度

**名称：** `_BoundedELTEBTexture`。**所在阶段：** BoundedELTEB 内部的有界纹理融合器。  
**输入/输出尺度与通道：** 与 _ELTEBTexture 相同；有效融合系数位于 `[-fusion_alpha_max,+fusion_alpha_max]`，锐化系数位于 `[0,unsharp_alpha_max]`。

##### 目标：它要解决什么问题

保留 ELTEB 的可解释纹理先验，同时通过参数化边界防止融合强度或锐化强度在小数据训练中失控。

##### 结构图

```text
fusion_alpha = max_f * tanh(raw_f)
unsharp_alpha = max_u * sigmoid(raw_u)
其余纹理构造与 _ELTEBTexture 相同
```

##### 逐步 Flow：数据到底怎样经过它

1. 构造时把用户给定初值映射到 atanh/logit 的 raw 参数。
2. 前向时 tanh 将融合系数限制在正负 max_f，允许模型抑制或增强纹理。
3. sigmoid 将锐化系数限制在 0 到 max_u，避免反向模糊或无限锐化。
4. 用有效系数构造 unsharp 并融合 P3。

##### 起作用的原理

有界重参数化改变的是优化空间而非主要结构。tanh/sigmoid 让物理含义明确：纹理残差最多占基线的一定幅度，锐化只能在非负有限区间。这样即使梯度推动 raw 参数很大，有效值也不会超出安全范围。

##### 初始化保护策略

max 值必须正；默认 fusion_alpha=0 对应零融合，unsharp 初值映射后保持用户设定。需记录 raw 与 effective 值，防止饱和导致梯度过小。

##### YAML 插入点

不独立出现，由 BoundedELTEB 参数 `fusion_alpha_max`、`unsharp_alpha_max` 控制。

##### 与基线相比唯一改变

相对 _ELTEBTexture，唯一改变是无界标量改成 tanh/sigmoid 有界参数化。

##### 预期收益及潜在代价

**预期收益：** 限制纹理支路破坏范围，提高小数据训练的稳定性与可解释性。  
**潜在代价与失败模式：** 边界过紧会压制真实收益；参数饱和后学习变慢，仍需选择 max 超参数。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：raw/effective alpha 曲线、饱和率、梯度、不同 max 的三种子 AP/Recall 与稳定性。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `block.py::_BoundedELTEBTexture`；它是保护策略，不保证精度提升。

#### D03 ELTEB

##### 名称、位置与输入/输出尺度

**名称：** `ELTEB`。**所在阶段：** P3 生成节点，组合原生 C3k2 与 full 原图纹理支路。  
**输入/输出尺度与通道：** 输入 `[image, x_before_P3]`；先由父类 C3k2 得到 `(B,Cp3,H/8,W/8)`，再纹理融合，输出同形。

##### 目标：它要解决什么问题

在保持 YOLO26n 原生 P3 特征精炼的同时，补充从原图直接计算的病斑纹理，避免用一条新支路替换原有表示。它解决的是微小、低对比病斑在多次下采样后边缘信号弱的问题。

##### 结构图

```text
x_before_P3 -> C3k2(base) -> P3_base -------------------------------\
image -> resize -> full texture cues -> compress/project -> *alpha -------- add/concat -> P3_enhanced
```

##### 逐步 Flow：数据到底怎样经过它

1. 父类 C3k2 完整执行基线 P3 生成。
2. 原图缩放到 P3，提取 RGB、gray、Sobel、Laplacian、unsharp。
3. 纹理线索压缩并投影到 P3 通道。
4. 按 add 或 identity-init concat 融合；默认 alpha=0，初始输出等于 P3_base。

##### 起作用的原理

ELTEB 是 late residual injection：学习主干仍负责语义，固定纹理算子提供高频先验，1x1 投影负责把先验翻译到特征通道。零起点使预训练 C3k2 输出完全保留，后续训练只在有监督证据支持时增加纹理。full 模式保留 RGB 与 unsharp，容量和噪声风险都高于 Lite。

##### 初始化保护策略

必须从兼容的 C3k2 权重初始化，纹理 alpha=0。原图输入的 from 索引要正确；训练日志记录 alpha。若使用 concat，核对 identity 初始化未被通用初始化器覆盖。

##### YAML 插入点

原生/EdgeLite ELTEB YAML 的 P3 节点 4 左右：`[[-2,-1],2,ELTEB,[512,False,0.25,16,'add',0.0,0.35]]`；实际 repeats 受 depth scaling。

##### 与基线相比唯一改变

相对同位置 C3k2，唯一新增 full 纹理残差；P4/P5、neck 和 Detect 保持不变。

##### 预期收益及潜在代价

**预期收益：** 在零扰动起点下补充可解释早期纹理，可能提升微小/低对比病斑。  
**潜在代价与失败模式：** 原图旁路、固定滤波与额外 1x1 增加延迟；容易放大叶脉和噪声；full 通道比 Lite 多。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：fusion/unsharp alpha、纹理分支范数、P3 小目标 AP、低对比类别 Recall、add/concat 和 full/lite latency。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码与 `elteb_experiment/configs` 可核对；当前结论必须区分 ELTEB 自身与同时存在的 EdgeLite/Bridge。

#### D04 BoundedELTEB

##### 名称、位置与输入/输出尺度

**名称：** `BoundedELTEB`。**所在阶段：** 当前 lasttrain 路线的受约束 P3 纹理增强节点。  
**输入/输出尺度与通道：** 与 ELTEB 相同，输出 P3；默认 `fusion_alpha_max=0.25`、`unsharp_alpha_max=0.75`。

##### 目标：它要解决什么问题

把 ELTEB 的两个关键强度参数限制在可解释范围，作为更稳健的 P3 纹理候选。它针对无界 alpha 在小样本、类别不均衡或长训练中可能放大噪声的问题。

##### 结构图

```text
C3k2 -> P3_base
image -> full cues(unsharp bounded) -> projection -> bounded texture scale -> residual/concat -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 先按 ELTEB 运行原生 C3k2。
2. 使用 _BoundedELTEBTexture 生成 full 纹理。
3. tanh 控制纹理残差在 ±0.25 内，sigmoid 控制锐化在 0-0.75。
4. 零融合初值保持基线，训练中逐步调整。

##### 起作用的原理

BoundedELTEB 不增加新的信息源，而是为已有纹理支路加可信域约束。它把“是否使用纹理”和“用多强”限制在安全范围，减少单个随机种子因 alpha 漂移得到偶然高分的可能。

##### 初始化保护策略

保留默认 max 与 alpha=0 作为第一基线；max 的改变单独消融。检查 raw 参数是否在边界饱和，并报告 effective 值而非只报 raw。

##### YAML 插入点

`lasttrain/configs/yolo26n_map50_v2_bounded_elteb_residual_p3p5.yaml` 的 P3 节点，参数包含两个上限。

##### 与基线相比唯一改变

相对 ELTEB，仅把 fusion/unsharp 参数改为有界重参数化；相对原生 C3k2则仍多出 full 纹理支路。

##### 预期收益及潜在代价

**预期收益：** 提高纹理增强稳定性，限制最坏破坏，适合长期训练。  
**潜在代价与失败模式：** 需要选择上限；上限过低可能把有效纹理贡献压没，过高则保护不足。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：effective alpha、边界饱和率、跨种子方差、低对比/叶脉混淆类别 AP、额外 latency。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码和 lasttrain YAML 可核对；是否优于 ELTEB 需要锁定相同图，仅改变有界参数化。

#### D05 ELTEBLite

##### 名称、位置与输入/输出尺度

**名称：** `ELTEBLite`。**所在阶段：** P3 的轻量原图纹理增强候选。  
**输入/输出尺度与通道：** 与 ELTEB 相同；纹理输入仅 3 通道：gray、Sobel、Laplacian，默认 texture_channels=8。

##### 目标：它要解决什么问题

保留最直接的灰度和边缘先验，删除 full 模式的 RGB、unsharp 与较宽纹理通道，以降低原图旁路的计算和噪声自由度。

##### 结构图

```text
image -> resize -> gray + Sobel + |Laplacian| -> cat3 -> compress(8) -> project -> *alpha
C3k2 P3 -------------------------------------------------------------- + -> y
```

##### 逐步 Flow：数据到底怎样经过它

1. 父类仍运行同一个 C3k2。
2. 原图缩放后只构造 gray/Sobel/Laplacian。
3. 3 通道压到 8 个纹理通道并投影到 P3。
4. 使用 ELTEB 的零起点 add/concat 机制融合。

##### 起作用的原理

Lite 假设病斑判别最需要亮度结构、一阶边缘和二阶斑点响应，而原始颜色已由主干卷积处理。减少输入与隐藏通道既降低成本，也限制支路学习把颜色偏差当捷径。代价是某些病害的颜色变化可能无法从纹理支路补充。

##### 初始化保护策略

保持 alpha=0；与 full 比较时除 mode、texture_channels 和随 mode 无效的 unsharp 外，其他图与训练配方一致。

##### YAML 插入点

`yolo26n_*_elteb_lite.yaml` 的 P3 节点，典型参数 texture_channels=8、fusion_alpha=0。

##### 与基线相比唯一改变

相对 full ELTEB，删除 RGB/unsharp 线索并把纹理通道 16 降到 8。

##### 预期收益及潜在代价

**预期收益：** 更低计算和更少过拟合自由度，保留关键边缘先验。  
**潜在代价与失败模式：** 颜色型病斑可能受损；固定边缘仍会响应叶脉；实测延迟未必与通道成比例。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：full/lite 参数与延迟、颜色型/纹理型类别逐类 AP、融合 alpha、边缘噪声误检。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码与 ELTEB-Lite YAML 可核对；Lite 是独立候选，不应与 full 的结果混写。

### 10.5 本仓库训练与损失模块

#### E01 BboxWIoUProgLoss

##### 名称、位置与输入/输出尺度

**名称：** `BboxWIoUProgLoss`。**所在阶段：** 训练 criterion 的边界框回归分支，替换原生 BboxLoss 的 IoU 聚焦方式。  
**输入/输出尺度与通道：** 输入前景预测框、目标框、匹配分数等；输出标量 `loss_iou` 与 `loss_dfl/L1`，不改变推理网络张量。

##### 目标：它要解决什么问题

在训练早期保持原生 CIoU，随后平滑过渡到带距离增益和动态非单调聚焦的 WIoU，使过易、正常和极难样本的梯度贡献更可控。

##### 结构图

```text
pred/target -> CIoU loss
pred/target -> raw IoU + center-distance gain -> beta(loss/running_mean) -> focus -> WIoU
box_loss=(1-blend)*CIoU + blend*WIoU -> score-weighted mean
reg_max>1: DFL; reg_max=1: normalized L1 branch
```

##### 逐步 Flow：数据到底怎样经过它

1. 按匹配分数和 fg_mask 选前景框，并计算 CIoU loss。
2. 计算 raw IoU、最小包围框对角线和中心距离，形成指数 distance_gain。
3. 以 running_mean 归一化 base loss 得 beta，再按 alpha/delta 生成非单调 focus，并夹在 0.5-3.0。
4. progress 在 10%-60% 训练区间用 smoothstep 从 0 到 1，把 CIoU 平滑混成 WIoU。
5. 对 score 权重归一化求和；定位分布分支保持原生 DFL，reg_max=1 时走归一化 L1。

##### 起作用的原理

running_mean 给每个样本难度一个相对尺度，非单调 focus 可避免极易样本占用梯度，也限制极端异常框过度主导。中心距离增益使几何位置偏差仍被惩罚。渐进混合避免一开始匹配和框预测很差时 WIoU 动态权重不稳定。

##### 初始化保护策略

保持原生 assigner、score 权重和 DFL/L1 分支；只替换 IoU 项。running_mean 是 buffer，训练时更新。先做 WIoU-only，再叠加 ProgLoss；记录 blend 曲线。

##### YAML 插入点

不进入模型 YAML；由 `patch_detection_model_loss()` 在训练进程内替换 criterion，配置在 `DEFAULT_LOSS_CONFIG`/训练脚本中。

##### 与基线相比唯一改变

相对基线只改变 box IoU loss 的聚焦与渐进混合，网络图、Detect 和匹配器保持不变。

##### 预期收益及潜在代价

**预期收益：** 可能降低异常框和极易样本对回归的无效主导，提高困难病斑定位稳定性。  
**潜在代价与失败模式：** 引入多项超参数和运行均值；错误聚焦可能压低真正困难样本；仅影响训练，不能用参数/GFLOPs解释收益。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：CIoU/WIoU 均值、blend、running_mean、focus/distance_gain 分布、box/DFL(L1) loss、定位 AP75 与训练时间。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

公式来自 `loss_extensions.py` 和 `wiou_progloss_loss.py`；属于本仓库实验实现，必须与原生 loss 分离报告。

#### E02 ProgLoss classification weighting

##### 名称、位置与输入/输出尺度

**名称：** `ProgLoss classification weighting`。**所在阶段：** 训练 criterion 的 BCE 分类正样本重加权。  
**输入/输出尺度与通道：** 输入 `(B,N,nc)` logits/target_scores，输出分类 loss 标量；推理结构不变。

##### 目标：它要解决什么问题

根据训练集类别计数为尾类正样本提供受限权重，并在训练中后期平滑开启，避免一开始用强类别权重扰乱共享特征学习。

##### 结构图

```text
class counts -> (max/count)^tail_power -> mean normalize -> clamp[min,max] = wc
progress -> smoothstep -> gain
active=1+gain*(wc-1)
positive_scale=1+(active-1)*target_score
BCE *= positive_scale -> sum
```

##### 逐步 Flow：数据到底怎样经过它

1. 从训练标签统计每类实例数；若缺失或长度不等于 nc，则回退全 1。
2. 按 `(max_count/count)^0.5` 构造权重，均值归一化并夹到 0.75-1.8。
3. 训练 10% 前 gain=0，10%-60% smoothstep 增长，之后达到 lambda_max=0.8。
4. 只用 target_scores 调整正目标项，负类位置保持基线 BCE 尺度。
5. 分类损失仍按 target_scores_sum 归一化并乘原生 cls gain。

##### 起作用的原理

尾类较少，普通 BCE 的累计正样本梯度偏小；权重提高其正样本贡献。均值归一化和上下限防止整体 loss 尺度漂移或极少类爆炸。渐进调度先让网络学习通用边缘和叶片结构，再逐步强调类别平衡。

##### 初始化保护策略

类计数必须只来自训练集，避免验证/测试泄漏；先验证计数与 class index 映射。正样本权重不应乘到全部负样本。tail_power、lambda、min/max 分别消融。

##### YAML 插入点

不进入模型 YAML；训练脚本调用 `count_train_labels` 后写入 `progloss_class_counts` 并配置 criterion。

##### 与基线相比唯一改变

相对基线只改变分类 BCE 的正样本类别权重与时间调度；box loss 若要同时改变必须另设 2x2 消融。

##### 预期收益及潜在代价

**预期收益：** 可能提升少数病害 Recall/逐类 AP，同时控制权重幅度和开启时机。  
**潜在代价与失败模式：** 头类 AP 可能下降；标签噪声尾类会被放大；数据增广或采样变化后计数权重可能失配。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：类计数/权重、gain 曲线、逐类正样本梯度、macro AP/Recall、尾类与头类差距、校准误差。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `loss_extensions.py::build_progloss_class_weights/classification_loss`；有效性必须逐类报告，不能只看总体 mAP。

#### E03 v8DetectionWIoUProgLoss / E2EWIoUProgLoss

##### 名称、位置与输入/输出尺度

**名称：** `v8DetectionWIoUProgLoss / E2EWIoUProgLoss`。**所在阶段：** 把 WIoU 与 ProgLoss 接入普通及 end-to-end 双分支训练的总 criterion。  
**输入/输出尺度与通道：** 接收 Detect 输出字典与 batch；返回总 loss 和 box/cls/dfl 三项。end2end 时分别计算 one-to-many 与 one-to-one。

##### 目标：它要解决什么问题

在不改本地 Ultralytics 原生 loss 源码的情况下，将实验 box/classification 项接入完整目标分配和 end-to-end 训练，并让 epoch progress 同步到两个分支。

##### 结构图

```text
DetectionModel.init_criterion -> end2end?
false: v8DetectionWIoUProgLoss
true : E2EWIoUProgLoss -> one2many criterion + one2one criterion
update each epoch -> loss_progress -> both bbox/progloss schedulers
```

##### 逐步 Flow：数据到底怎样经过它

1. 先调用原生 v8DetectionLoss 初始化，保留 device、stride、assigner、BCE 与 gains。
2. 仅将 bbox_loss 替换为 BboxWIoUProgLoss，并预计算类权重。
3. 复用原生 anchor、decode、assigner 流程，分类项调用自定义 weighting。
4. 每个 epoch update 计算 progress 并下发给 box loss。
5. end2end wrapper 同时更新 one2many 与 one2one，保证两个训练分支调度一致。

##### 起作用的原理

这种外置 criterion 把实验变量限制在损失函数，网络结构和权重加载不变，便于归因。end-to-end 模型若只修改一条分支会产生训练目标不一致，因此 wrapper 强制两路使用同一实现与进度。进程级 monkey patch 使原生代码仍可用于基线，但要求启动顺序严格。

##### 初始化保护策略

训练前显式调用 configure 与 patch，并打印 active config/criterion 类型；每 epoch 必须调用 update。基线进程不得残留 patch。先跑小 batch 单步，确认三项 loss 有限且两分支 progress 一致。

##### YAML 插入点

模型 YAML 不变；实验入口 `train_coffee_edgelite_wiou_progloss.py` 选择模型图并在进程中安装 criterion。

##### 与基线相比唯一改变

相对相同模型图，唯一改变是 criterion；若比较不同结构，必须同时保留各自 native-loss 对照，形成结构×loss 的析因设计。

##### 预期收益及潜在代价

**预期收益：** 隔离实验损失，兼容 end-to-end 双分支，便于回退和复现。  
**潜在代价与失败模式：** 进程级 patch 对调用顺序敏感；遗漏 update 会改变调度；同时修改 box 与 cls 会使单项归因不清。

##### 消融指标与结论门槛

除 `mAP50-95`、`Recall`、逐类 AP、参数量、GFLOPs 和目标设备端到端延迟外，本模块还应记录：criterion 类型、active config hash、one2many/one2one 分项 loss、progress 同步、NaN/Inf、每 epoch 训练耗时和最终逐类指标。所有对照必须锁定数据划分、输入尺寸、预训练权重、增强、优化器、epoch/early-stop、batch 和评估脚本。**结论门槛：至少 3 个随机种子，报告均值和波动；若收益小于种子间波动，或精度收益不能覆盖目标设备延迟与显存代价，则不得写成有效改进。**

##### 证据边界

代码事实来自 `wiou_progloss_loss.py` 与训练入口；报告必须注明这是训练期变化，推理参数/GFLOPs不变。

### 10.6 证据来源与使用边界

- 论文讲解框架与原生模块消融背景：*Toward a Deeper Understanding of YOLO26: Block-Level Architectural Analysis and Ablation Studies*，DOI `10.20944/preprints202603.2518.v1`。
- 原生 YOLO26n 图：`edgelite_experiment/local_ultralytics/ultralytics/cfg/models/26/yolo26.yaml`。
- 自定义结构实现：`edgelite_experiment/local_ultralytics/ultralytics/nn/modules/block.py`，以及对应实验目录下的 YAML。
- WIoU/ProgLoss：`wiou_progloss_experiment/loss_extensions.py`、`wiou_progloss_experiment/wiou_progloss_loss.py`。
- 本节的“预期收益”均是待检验假设；只有锁定训练配方、至少 3 个随机种子并报告均值与波动后，才能转写为实验结论。

## 11. 当前实验图与模块归属

| 实验图/目录 | 核心增量 | 明确不包含或已退役的部分 |
|---|---|---|
| `edgelite_experiment` | LDSConv、TextureStreamP3、SemanticStreamP5、EdgeLGMSFBridge | 不等同于 BiBridge 或 ELTEB |
| `edgelite_bibridge` | 在 EdgeLite 上加入 P5ToP3SemanticFuse | 不等同于 P5 大核历史最佳图 |
| `elteb_experiment` | ELTEB/ELTEBLite 的 P3 显式纹理注入 | 需按 YAML 判断是否叠加 EdgeLite/BiBridge |
| `yolo26n_p2p5_distanceconv_experiment` | NearDetail、FarContext、DistanceAwareP2P5Bridge | 直接 P2 Detect 已删除 |
| `p5_largekernel_experiment` | P5LargeKernelContext(13)，保留 ELTEB 与单向 P3-to-P5 融合 | BiBridge、P5-to-P3 feedback、SimAM 已关闭 |
| `wiou_progloss_experiment` | WIoU 盒回归与 ProgLoss 尾类调度 | 不改变网络图，只替换训练 criterion |

## 12. 后续新增模块的最低记录模板

每个新结构应至少补全以下信息，才能进入训练：

```text
名称：
输入/输出尺度与通道：
目标：
结构图：
初始化保护策略：
YAML 插入点：
与基线相比唯一改变：
预期收益及潜在代价：
消融指标：mAP50-95、Recall、逐类 AP、参数、GFLOPs、目标设备延迟
结论门槛：至少 3 个随机种子，报告均值和波动
```

这能把“结构解释”和“实验结论”分开：结构可以有合理动机，但只有经过与基线锁定配方的多种子对照，才可以认定为有效改进。
