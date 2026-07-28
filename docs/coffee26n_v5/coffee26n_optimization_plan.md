# Coffee26n 后续结构强化实施方案

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-23
- Verification Status: DESIGN-READY / IMPLEMENTATION-UNVERIFIED
- Version Label: coffee26n_plan_v3
- Working Directory: `E:\ultralytics-8.4.43`
- Reference Dataset Profile: `configs/profiles/coffee_ant6.yaml` (the current coffee evidence profile)
- Base Model: native YOLO26n

> 本文是后续代码生成和实验执行的结构设计稿。原文中的结构原理、历史结果和咖啡数据证据全部保留，但实现层必须区分“通用 YOLO detect 运行时”和“咖啡 profile”。任何调用方都必须显式提供 `--data`；`coffee_self_sum`、6 类名称、960 输入尺寸和咖啡混淆对只能来自 profile 或示例，不能写死进模型、损失或训练入口。新增模块的收益仍是假设，必须按消融流程验证后才能保留。

### 文档使用边界

1. **设计层**：第 1--23 节说明为什么选择这些模块、排序、卷积类型、激活函数、损失和背景增强，以及为什么拒绝 P2 Detect、全网 depthwise、早期 ELTEB 和无监督空间注意力。这些原则是实现必须保持的行为约束。
2. **实现层**：第 24 节以后把上述原则转换成目录、配置 schema、构造函数、`parse_model` 规则、数据/损失 API、权重迁移、输出、测试和失败行为。代码生成必须逐条满足这些契约。
3. **证据层**：第 2 节的 6 类统计、混淆对和 `YOLO26s=9,952,508` 是当前咖啡 profile 的证据，不是所有数据集的默认值。换数据集时重新计算，不得复制这些常数。

## 1. 最终决策摘要

新主线采用简单名称 **Coffee26n**。它不是另起一个模型家族，而是在原生 YOLO26n 的新式 `C3k2 + SPPF + C2PSA + P3/P4/P5 Detect` 主干上做受保护增强。

需要保留的已有优点：

1. 保留原生 YOLO26n 的混合 `C3k2(False/True)` 排列，不改成旧式 C2f 堆叠。
2. 保留 `SPPF -> C2PSA` 的 P5 语义主路径。
3. 保留 P3/P4/P5 三尺度检测，不重新加入 P2 Detect。
4. 保留 P3 细节向 P5 传递的方向，但必须改为残差接入，不能替换原生 P5。
5. 保留 `RoleAwareAttnFuse2` 的按样本、按通道注意力，并保留监控功能。
6. 保留 WIoU 处理框噪声、分类项处理类别长尾的职责分离，但修正现有 ProgLoss 的实际权重问题。

需要修改的核心问题：

1. 将 `ResidualEdgeLGMSFBridge + MonitoredRoleAwareAttnFuse2` 合并并简化命名为 `ResBridge`。
2. `ResBridge` 内的两次 LDSConv 改为小通道标准 `3x3,s=2 Conv`，因为桥内通道只有约 32，节省的 FLOPs 不值得牺牲跨通道纹理关系。
3. 删除最终方案中的早期原图 ELTEB 注入，将细节增强移动到 top-down 语义融合完成之后，命名为 `DetailBlock`。
4. 核心版 backbone 下采样保留原生标准 Conv，保证颜色与纹理联合建模以及预训练权重完整迁移；只在 neck MixDown 通过后逐节点测试 backbone MixDown。
5. neck 的两次下采样用 `MixDown`：原生标准 Conv 为主支路，抗混叠平均池化为小残差支路。
6. P5 可增加一个受门控的 `ContextBlock(k=13)`；大核只在 `30x30` 低分辨率特征上使用。
7. 融合后的 P4 可增加半通道 `MidBlock(k=9)`，用于补足前期相似病种所需的中尺度纹理组合；它与 P5 大核分别做单因素消融。
8. 分类损失改为独立的 `CoffeeLoss` 候选：保留 WIoU，用修正后的 TailBCE 处理长尾，并仅对实证混淆对加入质量门控 PairMargin。
9. 激活函数全局保留原生 SiLU；残差输出投影保持 `act=False`，不做无证据的全网激活替换。
10. 增加训练期 `BgMix` 候选，仅扰动 GT 整叶框之外的像素，用来抑制拍摄背景捷径，不增加推理参数。

最终完整候选由五个结构模块和两个训练组件组成：

| 简单名称 | 对应功能 | 是否进入核心版 | 是否已有思想基础 |
|---|---|---:|---|
| `ResBridge` | P3 细节到 P5 语义的残差注意力桥 | 是 | `ResidualEdgeLGMSFBridge + MonitoredRoleAwareAttnFuse2` |
| `DetailBlock` | 语义融合后的 P3 局部、方向和晕圈纹理精炼 | 是 | `NearDetailConvBlock`，但位置和门控重写 |
| `MixDown` | 保留原生 Conv 的 neck 抗混叠下采样 | 是 | 原生 Conv + 受门控低频残差 |
| `ContextBlock` | P5 局部、大核和全局上下文 | 条件保留 | `P5LargeKernelContext` |
| `MidBlock` | P4 半通道局部/大核组合 | 条件保留 | 新增，严格近恒等初始化 |
| `CoffeeLoss` | WIoU + TailBCE + PairMargin | 结构锁定后测试 | 修正现有 WIoU/ProgLoss |
| `BgMix` | 只扰动叶框外背景 | 结构锁定后测试 | 新增训练增强，推理时删除 |

## 2. 参考 profile 的数据事实如何约束结构

本节保留当前咖啡 6 类数据的设计证据，用于解释模块为什么存在，并形成 `coffee_ant6.yaml` 的推荐值。它不是通用训练入口的硬编码规范。对于其他数据集，代码必须从调用方传入的 dataset YAML 动态读取 `nc/names`、扫描训练标签统计类别数和框分布，并将本节所有类别相关结论视为“待重新测量”。结构本身仍可测试，但 PairMargin 默认关闭、BgMix 默认关闭，任何类别对都不得从病名猜测。

当前 6 类训练实例数为：

```text
ANT_AB=1443, ANT_CD=446, BLS_AB=356,
BLS_CD=208, CR=514, SM=347
```

最大类与最小类相差约 `6.94` 倍。训练框的归一化面积中位数约为 `0.391`，说明模型检测的是占图像很大的整片叶子，而不是小病斑框。真正困难的是：一个大框里只有很少像素提供类别证据。

当前两组 6 类原生 YOLO26n 验证混淆矩阵还给出了更直接的 ANT 吸附证据。矩阵纵轴为预测、横轴为真值：

| 真值被预测为 | 6cls bestparams | 6cls b96 map50 | 是否稳定 |
|---|---:|---:|---|
| `ANT_CD -> ANT_AB` | 0.09 | 0.10 | 是 |
| `BLS_AB -> ANT_AB` | 0.19 | 0.19 | 是，当前最明显的跨病种吸附 |
| `CR -> ANT_AB` | 0.12 | 0.17 | 是 |
| `BLS_CD -> ANT_CD` | 0.12 | 0.08 | 是 |
| unmatched/background FP 中 `ANT_AB` 占比 | 0.30 | 0.42 | 是，但需结合固定阈值 FP 数复核 |
| unmatched/background FP 中 `ANT_CD` 占比 | 0.24 | 0.25 | 是 |

这说明“相似病都往 ANT 靠”不是只由单次训练偶然造成。结构需要改善细节和上下文，损失则需要直接约束少数类正样本与 ANT logit 的相对距离。最后一列 background 不是像素级背景分类率，而是未匹配检测的归一化组成，因此只能证明 ANT 主导假阳性，不能单独证明模型关注了哪块背景。

| 数据问题 | 错误的结构反应 | Coffee26n 的结构反应 |
|---|---|---|
| 整叶框内包含大量正常叶肉、叶脉和背景 | 在骨干早期生成空间注意力或直接放大全图 Sobel | 先完成 P5->P4->P3 语义融合，再在 P3 做有界卷积精炼 |
| 病害有效区域小 | 增加 P2 Detect，把整叶当小目标 | 保留 P3 内部细节用于分类，但 Detect 仍为 P3/P4/P5 |
| 不同病害前期相似 | 全部使用 depthwise-first 下采样 | backbone 使用标准 Conv，保留颜色和纹理的跨通道联合关系 |
| 同病害前后期外观差距大 | 只强化某一个固定感受野 | P3 局部、P4 中尺度、P5 大范围上下文共同表达 |
| 类别不均衡和 ANT 吸附 | 依靠注意力模块自动解决长尾，或盲目重复尾类噪声样本 | 修正 TailBCE，并对已验证的尾类->ANT 混淆对做质量门控 PairMargin |
| 框边界和标签存在噪声 | 用更强的空间注意力拟合噪声 | 新分支全部采用有界残差和近恒等初始化 |
| 外部拍摄背景形成类别捷径 | 无掩膜地学习空间注意力 | 训练期只扰动 GT 框外像素的 BgMix；框内保持不变 |

### 2.1 为什么不把 P2 检测头加回来

在 `imgsz=960` 时，P2 是 `240x240` 网格。历史 P2/P5 路线增加到约 `5.73M` 参数和 `47.99 GFLOPs`，但相对旧最佳的 mAP50-95 提升只有约 `0.00014`。当前目标框本身又很大，因此 P2 Detect 的计算方向与数据困难不一致。

P2 可以作为独立的细节来源做未来研究，但不进入本轮 Coffee26n 主线。

### 2.2 为什么不继续直接使用早期 ELTEB

旧 V2 最佳轮附近的 ELTEB 有效融合系数约为 `-0.078`，即模型在主动减去固定原图纹理，而不是增加它。这不是“ELTEB 完全无效”的证明，但说明整图 Sobel、Laplacian、unsharp 同时响应了叶脉、反光和背景。

因此：

- `BoundedELTEB` 保留为对照组；
- Coffee26n 主方案不把原图固定滤波直接注入 backbone P3；
- 新 `DetailBlock` 只处理已经融合语义的 neck P3 特征。

## 3. Coffee26n 总体设计原则

### 3.1 近恒等起点

所有新增路径必须满足：

```math
y = x + g(\theta)\Delta(x), \qquad |g(\theta_0)| \ll 1
```

这里 `x` 是能加载预训练权重的原生输出，`Delta` 是新增分支，`g` 是有界门。训练开始时模型应接近原生 YOLO26n，而不是让随机初始化的新模块覆盖预训练表示。

### 3.2 只使用通道级跨尺度注意力

数据没有病斑像素掩膜，不能直接监督“应该看叶片的哪个位置”。因此跨尺度融合只产生每个通道的权重，不产生 `H x W` 空间权重。

### 3.3 P4 是自然过渡尺度

前期小病斑在 P3 更清楚，严重病害和叶片整体状态在 P5 更稳定。原生 PAN 已经通过 `P3 -> P4 -> P5` 传播信息，所以不再增加第二套复杂 BiBridge。P4 的原生 `C3k2(True)` 保持不变；只有在其输出后才条件加入近恒等 `MidBlock`，增强中尺度组合而不再增加跨尺度边。

### 3.4 不同时替换 backbone 和 neck 的全部卷积

backbone 标准 Conv 可以完整继承 `yolo26n.pt` 的空间与通道联合核。Coffee26n 第一版只修改 neck 下采样；若 `MixDown` 有效，再单独测试 backbone 节点 3/5/7，不能一次全替换。

### 3.5 激活函数遵循原生，而不是全网替换

当前 YOLO26 的 `Conv.default_act` 是 `nn.SiLU()`。低对比、早期病斑依赖微弱的正负响应，SiLU 的平滑性和负半轴信息比硬截断 ReLU 更适合当前任务；同时原生预训练权重也是在 SiLU 下形成的。因此本轮没有足够证据把全网改成 ReLU、ReLU6、HardSwish、Mish 或 GELU。

具体约束如下：

- backbone、neck 主分支和新增卷积分支内部继续使用 SiLU；
- `DetailBlock/MidBlock/ResBridge/ContextBlock` 的残差输出投影必须 `act=False`，让模块能产生正负修正；
- attention/selector 的隐藏层使用 SiLU，最终 logit 投影 `act=False`；
- `sigmoid/tanh/softmax` 只用于门和权重，不替代特征激活；
- 不设置第一轮激活函数消融。只有结构和损失均锁定、且出现明确的激活饱和或分支死亡证据，才允许在单个 `DetailBlock` 内做 SiLU/GELU 局部对照。

### 3.6 加强联通的边界

Coffee26n 不是通过堆更多任意连接来“加强联通”，而是保护并利用原生 YOLO26 的有效路径：

```text
B4 -> top-down T4
B3 -> top-down T3
T3 -> bottom-up O4
T4 -> bottom-up O4
O4/O4' -> bottom-up O5
E5 -> bottom-up O5
B3 -> ResBridge -> E5       # 唯一新增跨两级连接
```

这样每个 backbone 层既进入相邻 FPN/PAN，也有 P3 细节到 P5 的受门控长连接。额外恢复 P5->P3 BiBridge、给每一级都做 dense concat，或在 backbone 内再加 B3->B4/B4->B5 旁路，会重复现有路径并让高层背景语义反向污染细节层，当前没有必要。P4 `MidBlock` 强化的是融合后的表示，不新增跨尺度边。

## 4. 核心模块一：ResBridge

### 4.1 修改目的和原因

`ResBridge` 是本方案的核心。它完整结合用户要求的两个部分：

1. **Residual connection**：保留 C2PSA 后的 P5，桥接只作为修正量。
2. **Role-aware attention**：根据每张图和每个通道决定更信任 P3 细节还是 P5 语义。

它替代旧 `EdgeLGMSFBridge` 的“直接生成新 P5”行为，也替代旧桥内连续 LDSConv 的过度轻量下采样。

### 4.2 插入位置

```text
backbone P3 (node 4) -------------------\
                                         ResBridge -> enhanced P5
C2PSA 或 ContextBlock 后的 P5 ----------/
```

输出作为 neck 自顶向下路径的起点，并在最后 P5 head 的 Concat 中再次作为 P5 skip。方向仅为 P3->P5，不加入 P5->P3 反馈。

### 4.3 内部结构

以 `imgsz=960`、YOLO26n 为例：

```text
P3:(B,128,120,120)
 -> 1x1 reduce to 32
 -> DetailBlock-lite
 -> standard Conv3x3,s2                 (60x60)
 -> standard Conv3x3,s2                 (30x30)
 -> detail:(B,32,30,30) -----------------------------\
                                                           RoleAttention
P5:(B,256,30,30)                                           -> fuse
 -> 1x1 reduce to 32                                      /
 -> {DW5x5, DW3x3 dilation=2} -> cat -> 1x1 residual ----/
 -> semantic:(B,32,30,30)

delta = Conv1x1(fuse, 256, act=False)
output = P5 + sigmoid(bridge_gate) * delta
```

建议参数：

```text
reduction=8
c_mid=max(16, c5//8)=32
bridge_gate_init=-3.0
initial bridge scale=sigmoid(-3)=0.0474
```

### 4.4 RoleAttention 数学公式

设对齐后的 P3 细节为 `d`，P5 语义为 `s`：

```math
z = [d,\ s,\ |d-s|,\ d\odot s]
```

```math
[\ell_d,\ell_s] = W_2\,\mathrm{SiLU}(W_1\,\mathrm{GAP}(z))
```

```math
[a_d,a_s] = \mathrm{softmax}([\ell_d,\ell_s],\mathrm{role}),
\qquad a_d+a_s=1
```

```math
f = a_d\odot d + a_s\odot s
```

最终桥接：

```math
P5' = P5 + \sigma(g_b)\,W_o(f)
```

`W2` 必须零初始化，使初始 `a_d=a_s=0.5`。这能防止注意力在第一轮就塌缩到某一路。

### 4.5 代码骨架

实现文件建议为：

```text
coffee26n_experiment/local_ultralytics/ultralytics/nn/modules/block.py
```

RoleAttention 不必重复发明，直接复用已有监控版：

```python
RoleAttention = MonitoredRoleAwareAttnFuse2
```

`ResBridge` 骨架：

```python
class ResBridge(nn.Module):
    """Residual P3-to-P5 bridge with channel-only role attention."""

    def __init__(self, c3, c5, reduction=8, gate_init=-3.0):
        super().__init__()
        c = max(16, c5 // reduction)

        self.p3_reduce = Conv(c3, c, 1, 1)
        self.p3_detail = DetailBlock(c, c, reduction=2, line_kernel=5, dilation=2, gate_max=0.25)
        self.p3_down = nn.Sequential(
            Conv(c, c, 3, 2),
            Conv(c, c, 3, 2),
        )

        self.p5_reduce = Conv(c5, c, 1, 1)
        self.p5_local = Conv(c, c, 5, 1, g=c)
        self.p5_dilated = Conv(c, c, 3, 1, d=2, g=c)
        self.p5_mix = Conv(c * 2, c, 1, 1, act=False)

        self.attn = MonitoredRoleAwareAttnFuse2(c)
        self.out = Conv(c, c5, 1, 1, act=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def effective_gate(self):
        return torch.sigmoid(self.gate)

    def forward(self, xs):
        p3, p5 = xs
        detail = self.p3_down(self.p3_detail(self.p3_reduce(p3)))

        base = self.p5_reduce(p5)
        semantic_delta = self.p5_mix(torch.cat((self.p5_local(base), self.p5_dilated(base)), 1))
        semantic = base + semantic_delta

        if detail.shape[-2:] != semantic.shape[-2:]:
            detail = F.interpolate(detail, size=semantic.shape[-2:], mode="nearest")

        fused = self.attn(detail, semantic)
        return p5 + self.effective_gate().to(fused.dtype) * self.out(fused)
```

### 4.6 必须记录的诊断量

- `bridge_gate`；
- P3/P5 平均注意力权重；
- 注意力二元熵；
- 按类别统计的 P3/P5 权重；
- `P5'` 与原 P5 的余弦相似度；
- 关闭桥接门后重新验证的 mAP 差值。

若训练后半段 `bridge_gate < 0.03`，并且关闭桥接后 mAP50 变化小于 `0.002`，应删除该桥，而不是因为它“理论合理”而保留。

## 5. 核心模块二：DetailBlock

### 5.1 修改目的和原因

有效病斑面积小，但标签只告诉模型整片叶子的类别。`DetailBlock` 不尝试产生病斑掩膜，而是用三个固定感受野的可学习卷积分支保留可区分的局部结构：

- `3x3`：点状斑块和局部边界；
- `1x5 -> 5x1`：方向性边缘、条带和叶脉相邻病斑；
- `3x3,dilation=2`：低对比晕圈和更宽的扩散区域。

它放在 top-down P3 C3k2 之后，此时 P3 已经接收 P5/P4 语义，能比早期 ELTEB 更好地区分病斑和无关叶脉。

### 5.2 插入位置

```text
P5 up -> cat(P4) -> C3k2 = T4
T4 up -> cat(backbone P3) -> C3k2 = T3_base
T3_base -> DetailBlock = T3
T3 -> MixDown -> bottom-up P4
```

### 5.3 数学形式

```math
u=W_r(x)
```

```math
r_1=DWConv_{3\times3}(u),\quad
r_2=DWConv_{5\times1}(DWConv_{1\times5}(u)),\quad
r_3=DWConv_{3\times3,d=2}(u)
```

```math
\Delta=W_m([r_1,r_2,r_3])
```

```math
y=shortcut(x)+0.25\tanh(g_d)\Delta
```

`g_d=0` 时初始增强强度严格为 0，输出等于原生 P3 head。门允许正向增加或负向抑制纹理，这一点比只能正向放大的 sigmoid 门更适合存在叶脉噪声的数据。

### 5.4 代码骨架

```python
class DetailBlock(nn.Module):
    """Late P3 detail refinement with bounded signed residual."""

    def __init__(self, c1, c2, reduction=2, line_kernel=5, dilation=2, gate_max=0.25):
        super().__init__()
        c = max(16, c2 // reduction)
        self.shortcut = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=False)
        self.reduce = Conv(c1, c, 1, 1)
        self.spot = Conv(c, c, 3, 1, g=c)
        self.line = nn.Sequential(
            Conv(c, c, (1, line_kernel), 1, g=c),
            Conv(c, c, (line_kernel, 1), 1, g=c),
        )
        self.halo = Conv(c, c, 3, 1, d=dilation, g=c)
        self.mix = Conv(c * 3, c2, 1, 1, act=False)
        self.gate_raw = nn.Parameter(torch.zeros(1))
        self.gate_max = float(gate_max)

    def effective_gate(self):
        return self.gate_max * torch.tanh(self.gate_raw)

    def forward(self, x):
        u = self.reduce(x)
        delta = self.mix(torch.cat((self.spot(u), self.line(u), self.halo(u)), 1))
        return self.shortcut(x) + self.effective_gate().to(delta.dtype) * delta
```

### 5.5 为什么不用空间注意力替代

空间注意力会生成 `A(x) in R^(H x W)`，但当前损失只有整叶框和类别，不知道病斑像素在哪里。它可能稳定地关注叶缘、叶脉或拍摄背景并获得训练集捷径。`DetailBlock` 只改变局部卷积响应，不直接宣称某个位置就是病斑，风险更小。

### 5.6 消融要求

依次测试：

1. 仅 `3x3`；
2. `3x3 + asymmetric`；
3. 三分支完整版本；
4. line kernel `3` 对 `5`；
5. dilation `1` 对 `2`。

第一轮不要同时改变 kernel 和 dilation。

## 6. 核心模块三：MixDown

### 6.1 修改目的和原因

旧 LDSConv 的顺序是：

```text
depthwise 3x3,s2 -> pointwise 1x1
```

它在通道混合之前就进行空间抽样。病害前期的颜色差、斑点边界和叶脉关系往往需要跨通道联合判断，YOLO26n-n 本身又很窄，因此不应在 backbone 全面采用 depthwise-first 下采样。

`MixDown` 以原生标准 Conv 为主，额外加入低频抗混叠残差：

```math
D(x)=Conv_{3\times3,s=2}(x)
```

```math
L(x)=Conv_{1\times1}(AvgPool_{2\times2,s=2}(x))
```

```math
y=D(x)+\sigma(g_m)L(x)
```

标准 Conv 分支保留颜色与纹理的跨通道空间卷积；平均池化分支降低 stride-2 抽样对弱病斑的混叠。`g_m=-4` 时低频分支初始权重约 `0.018`。

### 6.2 使用位置

第一版只替换 neck 两个原生下采样节点：

```text
T3 120x120 -> MixDown -> 60x60
O4 60x60   -> MixDown -> 30x30
```

backbone 节点 0/1/3/5/7 仍使用原生 `Conv3x3,s2`。只有 neck 结果明确为正，才单独测试 backbone P3、P4、P5 的 MixDown。

### 6.3 代码骨架

```python
class MixDown(nn.Module):
    """Native stride-2 Conv plus a gated anti-alias residual."""

    def __init__(self, c1, c2, gate_init=-4.0):
        super().__init__()
        self.base = Conv(c1, c2, 3, 2)
        self.low = Conv(c1, c2, 1, 1, act=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def effective_gate(self):
        return torch.sigmoid(self.gate)

    def forward(self, x):
        base = self.base(x)
        low = self.low(F.avg_pool2d(x, kernel_size=2, stride=2))
        return base + self.effective_gate().to(low.dtype) * low
```

### 6.4 预训练保护

`MixDown.base` 的结构和原生 neck Conv 完全相同，应显式把原生 node 17/20 权重映射到新图的 `MixDown.base`。低频支路和 gate 是唯一新增参数。

### 6.5 对照卷积

卷积类型必须单独比较：

| 变体 | 顺序 | 用途 |
|---|---|---|
| native Conv | full `3x3,s2` | 准确率控制组和 backbone 默认 |
| LDSConv | DW `3x3,s2` -> PW `1x1` | 历史轻量对照 |
| SCDown | PW `1x1` -> DW `3x3,s2` | 速度候选，不作为准确率默认 |
| MixDown | native Conv + gated avg branch | Coffee26n neck 候选 |

任何结论都必须同时报告真实设备延迟；depthwise 的理论 FLOPs 下降不等于实际推理更快。

## 7. 条件模块四：ContextBlock

### 7.1 目的

同一种病从早期到晚期，病斑大小、密度和叶片整体颜色变化很大。`ContextBlock` 在 P5 的 `30x30` 网格上同时保留：

- depthwise `3x3` 局部上下文；
- depthwise `1x13 -> 13x1` 大范围上下文；
- GAP 全局通道上下文。

它解决“不同病程需要不同感受野”的问题，但可能与 SPPF/C2PSA 重复，所以只能在核心结构验证成功后加入。

### 7.2 数学形式

```math
[q_l,q_k,q_g]=softmax(MLP(GAP([L(x),K(x),G(x)])))
```

```math
C(x)=q_lL(x)+q_kK(x)+q_gG(x)
```

```math
y=x+\sigma(g_c)W_o(C(x))
```

建议 `kernel=13, reduction=4, gate_init=-3.0`。必须记录三路 selector 权重和外部门；若 selector 长期固定为一路，应退化成单分支模块重新测试。

### 7.3 实现方式

优先复用已验证能构图的 `P5LargeKernelContext(c1,c2,kernel_size,reduction,gate_init)`，但必须提供参数名适配器，使本方案和 parser 统一使用 `kernel`，不能只写 `pass` 后再让关键字参数不匹配：

```python
class ContextBlock(P5LargeKernelContext):
    """Simple Coffee26n name for the gated P5 context block."""

    def __init__(self, c1, c2, kernel=13, reduction=4, gate_init=-3.0):
        super().__init__(
            c1=c1,
            c2=c2,
            kernel_size=kernel,
            reduction=reduction,
            gate_init=gate_init,
        )
```

插在 `C2PSA` 后、`ResBridge` 前：

```text
C3k2 -> SPPF -> C2PSA -> ContextBlock -> ResBridge
```

## 8. 条件模块五：MidBlock

### 8.1 为什么 P4 需要一个小型中尺度模块

当前混淆矩阵中 `BLS_AB -> ANT_AB` 和 `CR -> ANT_AB` 稳定偏高。它们不是纯粹的 P3 小斑点问题：P3 能看到局部纹理，P5 能看到整叶状态，但区分“局部斑点如何分布在叶脉、叶缘和叶片区域中”需要 `60x60` 的中尺度组合。直接把 P4 换成全大核或再加一条 P5->P3 反馈都会增加平滑或反馈噪声，因此只在已完成 top-down 与 bottom-up 拼接的 `O4` 上做局部修正。

### 8.2 结构和大核范围

`MidBlock` 先保持通道数不变，再把通道分成两半：

```text
O4 (128,60,60)
  -> A: depthwise 3x3
  -> B: depthwise 1x9 -> 9x1
  -> concat(A,B) -> 1x1 mix
  -> O4' = O4 + 0.20*tanh(g_m)*delta
```

只在半通道上使用因式分解大核，既扩大了中尺度感受野，又避免对全部 P4 特征做一次强平滑。这里的大核是 `1x9 -> 9x1`，不是密集 `9x9`：后者在 `c=128` 时仅卷积权重就约 `1.33M`，没有必要用这么高的参数换取同样的空间覆盖。

建议实现：

```python
class MidBlock(nn.Module):
    """Partial-channel P4 refinement with a bounded signed residual."""

    def __init__(self, c1, c2, kernel=9, gate_max=0.20):
        super().__init__()
        self.base = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=False)
        ca = c2 // 2
        cb = c2 - ca
        self.local = Conv(ca, ca, 3, 1, g=ca)
        self.large = nn.Sequential(
            Conv(cb, cb, (1, kernel), 1, g=cb),
            Conv(cb, cb, (kernel, 1), 1, g=cb),
        )
        self.mix = Conv(c2, c2, 1, 1, act=False)
        self.ca, self.cb = ca, cb
        self.gate_raw = nn.Parameter(torch.zeros(1))
        self.gate_max = float(gate_max)

    def effective_gate(self):
        return self.gate_max * torch.tanh(self.gate_raw)

    def forward(self, x):
        base = self.base(x)
        xa, xb = base.split((self.ca, self.cb), dim=1)
        delta = self.mix(torch.cat((self.local(xa), self.large(xb)), dim=1))
        return base + self.effective_gate().to(delta.dtype) * delta
```

### 8.3 插入位置和消融

```text
T3 -> MixDown -> cat(T4) -> C3k2 = O4
O4 -> MidBlock = O4'
O4' -> MixDown -> cat(E5) -> C3k2 = O5
Detect(T3, O4', O5)
```

`MidBlock` 不是核心版的默认必选项。先比较 `K0=core`、`K1=core+MidBlock(k=9)`；只有 `K1` 对 `BLS_AB/CR -> ANT_AB` 的混淆和宏平均指标同时达到门槛，才与 P5 `ContextBlock` 做组合。若门长期接近 0，删除模块。

## 9. 完整 Coffee26n 数据流

### 9.1 全结构总图

```text
Image
  -> Conv s2
  -> Conv s2 -> C3k2(False)                              P2
  -> Conv s2 -> C3k2(False) ----------------------------- B3/P3 skip
  -> Conv s2 -> C3k2(True) ------------------------------ B4/P4 skip
  -> Conv s2 -> C3k2(True) -> SPPF -> C2PSA
  -> [ContextBlock] ------------------------------------- B5 context

B3 ------------------------------\
                                    ResBridge(RoleAttention + residual) -> E5
B5 context -----------------------/

E5 -> Upsample -> cat(B4) -> C3k2(True)                  T4
T4 -> Upsample -> cat(B3) -> C3k2(True)                  T3_base
T3_base -> DetailBlock                                  T3/P3 detect

T3 -> MixDown -> cat(T4) -> C3k2(True)                  O4
O4 -> [MidBlock]                                        O4'/P4 detect
O4' -> MixDown -> cat(E5) -> C3k2(True, attn=True)       O5/P5 detect

Detect(T3, O4', O5)
```

方括号中的 `ContextBlock` 和 `MidBlock` 都表示条件模块。核心版从 `C2PSA` 直接进入 `ResBridge`，并从 O4 直接进入第二个 `MixDown`。两个条件模块必须分别证明有效后才允许组合。

### 9.2 张量表（`imgsz=960` 仅为参考 profile 示例）

下表用 `960 -> P3=120, P4=60, P5=30` 展示节点关系；实现必须按输入 `H,W` 和模型 stride 动态检查，不能把 `120/60/30` 写成固定 shape。若输入为 `H x W`，三路应分别为约 `ceil(H/8) x ceil(W/8)`、`ceil(H/16) x ceil(W/16)`、`ceil(H/32) x ceil(W/32)`，并以实际 Ultralytics stride/padding 输出为准。

| Node | Module | From | 输出 `(C,H,W)` | 作用 |
|---:|---|---|---:|---|
| 0 | Conv | -1 | 16,480,480 | 原生 stem |
| 1 | Conv | -1 | 32,240,240 | 原生 P2 下采样 |
| 2 | C3k2(False) | -1 | 64,240,240 | 浅层局部特征 |
| 3 | Conv | -1 | 64,120,120 | 原生 P3 下采样 |
| 4 | C3k2(False) | -1 | 128,120,120 | B3 细节源 |
| 5 | Conv | -1 | 128,60,60 | 原生 P4 下采样 |
| 6 | C3k2(True) | -1 | 128,60,60 | B4 中尺度源 |
| 7 | Conv | -1 | 256,30,30 | 原生 P5 下采样 |
| 8 | C3k2(True) | -1 | 256,30,30 | 深层语义 |
| 9 | SPPF | -1 | 256,30,30 | 多池化感受野 |
| 10 | C2PSA | -1 | 256,30,30 | 全局关系 |
| 11 | ContextBlock | -1 | 256,30,30 | 条件 P5 上下文 |
| 12 | ResBridge | [4,11] | 256,30,30 | 残差注意力增强 E5 |
| 13 | Upsample | -1 | 256,60,60 | top-down |
| 14 | Concat | [13,6] | 384,60,60 | 合并 B4 |
| 15 | C3k2(True) | -1 | 128,60,60 | T4 |
| 16 | Upsample | -1 | 128,120,120 | top-down |
| 17 | Concat | [16,4] | 256,120,120 | 合并 B3 |
| 18 | C3k2(True) | -1 | 64,120,120 | T3_base |
| 19 | DetailBlock | -1 | 64,120,120 | T3 |
| 20 | MixDown | -1 | 64,60,60 | P3->P4 |
| 21 | Concat | [20,15] | 192,60,60 | PAN P4 |
| 22 | C3k2(True) | -1 | 128,60,60 | O4 |
| 23 | MidBlock | -1 | 128,60,60 | 条件 P4 大核修正 O4' |
| 24 | MixDown | -1 | 128,30,30 | P4->P5 |
| 25 | Concat | [24,12] | 384,30,30 | PAN P5 |
| 26 | C3k2(True,attn=True) | -1 | 256,30,30 | O5 |
| 27 | Detect | [19,23,26] | P3/P4/P5 | end-to-end detect |

### 9.3 完整候选 YAML

目标文件：

```text
coffee26n_experiment/configs/coffee26n_full.yaml
```

```yaml
nc: 1  # parseable placeholder only; the generator must overwrite from dataset YAML before build
end2end: True
reg_max: 1
scale: n
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]             # 4 B3
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]                    # 6 B4
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]                        # 10 native B5
  - [-1, 1, ContextBlock, [1024, 13, 4, -3.0]]   # 11 optional in core ablation
  - [[4, 11], 1, ResBridge, [8, -3.0]]            # 12 output channels follow P5

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]                    # 15 T4

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]                    # 18 T3_base
  - [-1, 1, DetailBlock, [256, 2, 5, 2, 0.25]]   # 19 T3

  - [-1, 1, MixDown, [256, -4.0]]                 # 20
  - [[-1, 15], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]                    # 22 O4
  - [-1, 1, MidBlock, [512, 9, 0.20]]             # 23 O4'

  - [-1, 1, MixDown, [512, -4.0]]                 # 24
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]]        # 26 O5

  - [[19, 23, 26], 1, Detect, [nc]]
```

变体关系必须明确：

- `coffee26n_core.yaml`：删除 node 11 `ContextBlock` 和 node 23 `MidBlock`；`ResBridge` 改为 `[4,10]`，Detect 来源为 `[18,21,24]`；`nc` 由 dataset YAML 注入，不在文件中固定；
- `coffee26n_p5lk.yaml`：core 只加 `ContextBlock(k=13)`；
- `coffee26n_p4mid.yaml`：core 只加 `MidBlock(k=9)`；
- `coffee26n_full.yaml`：仅在两个单模块都通过后才使用上面的组合图，Detect 来源为 `[19,23,26]`。

## 10. 代码注册和 parse_model 修改

### 10.1 文件归属

新实验必须自包含：

```text
coffee26n_experiment/
  README.md
  configs/
    models/
      yolo26n_native_control.yaml
      coffee26n_core.yaml
      coffee26n_p4mid.yaml
      coffee26n_p5lk.yaml
      coffee26n_full.yaml
    profiles/
      generic.yaml
      coffee_ant6.yaml
  local_ultralytics/
    ultralytics/nn/modules/block.py
    ultralytics/nn/modules/__init__.py
    ultralytics/nn/tasks.py
    ultralytics/cfg/models/26/
      yolo26n.yaml
  loss/
    __init__.py
    coffee_loss.py
    loss_extensions.py
    pair_config.yaml
    test_coffee_loss.py
  augmentations/
    bgmix.py
    test_bgmix.py
  tests/
    test_config_and_dataset.py
    test_structure.py
    test_weight_transfer.py
  train_coffee26n.py
  verify_structure.py
  submit_coffee26n.slurm
  source_manifest.json
```

禁止直接修改仓库根目录的全局 `ultralytics/` 后再与旧结果比较，否则无法确定服务器实际导入的是哪一份代码。
`local_ultralytics/` 必须是运行时优先导入的完整、可追溯副本；不得通过当前工作目录碰巧导入根目录同名模块。

### 10.2 modules/__init__.py

导入并加入 `__all__`：

```python
from .block import ContextBlock, DetailBlock, MidBlock, MixDown, ResBridge
```

### 10.3 tasks.py import

在模块导入列表增加：

```python
ContextBlock,
DetailBlock,
MidBlock,
MixDown,
ResBridge,
```

### 10.4 parse_model

`MixDown`、`DetailBlock`、`MidBlock`、`ContextBlock` 都是普通 `(c1,c2,...)` 模块，加入 `base_modules`：

```python
base_modules = frozenset({
    # existing modules...
    MixDown,
    DetailBlock,
    MidBlock,
    ContextBlock,
})
```

`ResBridge` 输出通道始终等于输入 P5 通道，使用专用解析分支：

```python
elif m is ResBridge:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("ResBridge expects from=[P3_INDEX, P5_INDEX].")
    c3, c5 = ch[f[0]], ch[f[1]]
    c2 = c5
    args = [c3, c5, *args]
```

这样不会重复旧 EdgeBridge YAML 中手写 `c_out` 而遗漏 width scaling 的风险。

## 11. 预训练权重迁移

### 11.1 原则

必须从同一份 `yolo26n.pt` 初始化。所有未改变的模块逐层迁移，新增模块保持其保护性初始化。不能使用 `strict=False` 后只打印“加载成功”而不检查覆盖率。

### 11.2 full 图层映射

```text
target 0..10 <- source 0..10
target 15    <- source 13
target 18    <- source 16
target 22    <- source 19
target 26    <- source 22
target 27    <- source 23（仅迁移 shape-compatible 的回归/共享参数）

target 20.base <- source 17
target 24.base <- source 20
```

以下模块是新增参数：

```text
target 11 ContextBlock
target 12 ResBridge
target 19 DetailBlock
target 20.low + gate
target 23 MidBlock
target 24.low + gate
```

### 11.3 初始化验收

构图后必须验证：

1. 未修改且 shape-compatible 的原生层权重逐元素匹配 source；COCO 80 类分类输出到当前 6 类分类输出不匹配是预期行为；
2. `RoleAttention` 初始 P3/P5 权重为 `0.5/0.5`；
3. ResBridge gate 为 `0.0474 +/- 1e-4`；
4. DetailBlock gate 为 `0`；
5. MixDown gate 为 `0.0180 +/- 1e-4`；
6. ContextBlock gate 为 `0.0474 +/- 1e-4`；
7. MidBlock gate 为 `0`；
8. Detect 来源为 `[19,23,26]`，stride 为 `[8,16,32]`；
9. 一次 `1x3x960x960` 前向输出有限值且无 NaN。

## 12. 参数预算和大卷积边界

### 12.1 当前硬预算已经现场构图核验（咖啡 profile 证据）

使用当前 `ultralytics/cfg/models/26/yolo26.yaml`、`nc=6` 分别构建 n/s，实测参数为：

| 模型 | 当前 6 类参数量 | 相对 YOLO26s |
|---|---:|---:|
| 原生 YOLO26n | 2,506,140 | 25.18% |
| 原生 YOLO26s | 9,952,508 | 100% |

因此当前咖啡 profile 的参数验收采用两层门槛：

1. **硬门槛**：最终构建后的精确参数必须小于“同一 `nc`、同一 scales、同一 Ultralytics 版本构建的原生 YOLO26s 参数量”。当前 6 类的 `9,952,508` 只是已测例子；实现必须在 `verify_structure.py` 中动态构建并记录该上限，等于或超过立即淘汰；
2. **推荐门槛**：优先保持原生 YOLO26n 同 `nc` 参数的 `2.0x` 以内（咖啡 6 类约为 `5,012,280`，仅作估算）。超过推荐值虽不违反硬约束，但必须证明精度增益明显优于较小版本，且延迟可接受。

参数比较不能跨 `nc`、scales、输入尺寸或代码版本直接复用。`verify_structure.py` 必须输出 `model_params`、`native_n_params`、`native_s_params`、`hard_limit`、`headroom` 和比较所用的 `nc/scales/source_revision`。

按当前模块骨架和 n 尺度通道估算，增量约为：

| 新增部分 | 估算增量参数 | 备注 |
|---|---:|---|
| ResBridge | 约 0.047M | 包括 c=32 的 bridge、RoleAttention 和输出投影 |
| DetailBlock | 约 0.010M | P3 c=64，深度可分空间分支 |
| 两个 MixDown 低频支路 | 约 0.021M | 原生 3x3 主支路不算增量 |
| MidBlock(k=9) | 约 0.019M | P4 c=128，仅半通道大核 |
| ContextBlock(k=13) | 约 0.19M | P5 c=256，depthwise factorized 大核 |
| 完整组合 | 约 2.79M 总参数 | 只是实现前估算，不是最终验收值 |

完整组合预计远低于 YOLO26s，但 `verify_structure.py` 仍必须同时输出 `sum(p.numel())`、GFLOPs、峰值显存和目标设备 latency；估算不能替代真实构图。

### 12.2 哪些位置允许大核

- **P3 120x120**：不上 `9/13` 大核。保留 `3x3`、`1x5->5x1` 和 `3x3,d=2`，因为大范围平滑容易覆盖早期小病斑。
- **P4 60x60**：允许半通道 `1x9->9x1`，用于病斑与叶脉/叶缘的中尺度关系。
- **P5 30x30**：允许 `1x13->13x1`，用于病程范围、密度和全叶颜色上下文。

参数许可不等于应该使用密集大卷积。P5 上一个完整 `256x256x13x13` 卷积仅权重就约 `11.08M`，单层已经使总模型超过 YOLO26s，明确禁止。全通道标准因式分解 `1x13->13x1` 约增加 `1.70M`，虽然参数可通过硬门槛，但 C2PSA 和前后 `1x1` 已承担跨通道混合，第一轮没有必要。只有 depthwise 版本已经显示 large selector 权重稳定较高、且有正收益，才允许把分组数从 depthwise 单独放宽做后续对照。

## 13. CoffeeLoss：框噪声、长尾和 ANT 吸附

### 13.1 先纠正文档对当前 ProgLoss 的描述

现有 `loss_extensions.py` 实际执行的是：

```math
r_c=(n_{max}/n_c)^{0.5},\qquad
\tilde w_c=clip\left(r_c/mean(r),0.75,1.8\right)
```

随后渐进倍率最大只有 `0.8`：

```math
w_c^{eff}=1+0.8(\tilde w_c-1)
```

在当前 6 类计数下，真实数值为：

| 类别 | 计数 | `r_c` | 均值归一化并截断后 | 当前最终有效倍率 |
|---|---:|---:|---:|---:|
| ANT_AB | 1443 | 1.000 | 0.750 | 0.800 |
| ANT_CD | 446 | 1.799 | 0.967 | 0.974 |
| BLS_AB | 356 | 2.013 | 1.082 | 1.066 |
| BLS_CD | 208 | 2.634 | 1.416 | 1.333 |
| CR | 514 | 1.676 | 0.901 | 0.921 |
| SM | 347 | 2.039 | 1.096 | 1.077 |

表中最后一列按高质量 `q=1` 正锚点计算；soft target `q<1` 时实际调整更弱。所以真实问题不是“多个尾类都被截在 1.8”，而是**均值归一化把 BLS_AB、CR、SM 又拉回了 1 附近，CR 甚至被降权**。这与 `CR -> ANT_AB=0.12~0.17` 的现象不一致，必须修正。

### 13.2 YOLO26n 的第三个框损失不是 DFL

当前 YOLO26 配置为 `reg_max=1`。因此 `BboxLoss.dfl_loss=None`，代码中的第三个 `loss[2]` 虽然仍乘 `hyp.dfl` 并沿用 dfl 名称，实际计算的是归一化 L1 距离，不是分布焦点损失。CoffeeLoss 不改变 `reg_max`，总损失应准确写成：

```math
L_{branch}=\lambda_{box}L_{WIoU}
+\lambda_{cls}\left(L_{TailBCE}+\lambda_{pair}(t)L_{Pair}\right)
+\lambda_{l1}L_{normL1}
```

YOLO26 end-to-end 的总损失仍按原生进度组合 one-to-many 与 one-to-one：

```math
L=o_{2m}(t)L_{o2m}+o_{2o}(t)L_{o2o}
```

`CoffeeLoss` 必须作为 `E2ELoss` 的 `loss_fn` 同时接入两条分支，不能只改 one-to-many 后让推理使用的 one-to-one 分支保持旧损失。

### 13.3 框回归继续使用 WIoU

整叶框边界存在人为松紧差异，继续保留现有的 CIoU->WIoU 渐进混合：前 10% 主要依赖原生 CIoU，10%-60% 平滑转向非单调 WIoU，避免训练初期让随机预测主导 running mean。WIoU、TailBCE、PairMargin 必须分别消融，不再叠加新的 Shape-IoU、Inner-IoU 或额外框损失。

现有 `progress_blend()` 把 WIoU ramp 错误地绑定在 `progloss_enabled` 上：关闭 ProgLoss 时反而会从第一轮 100% 启用 WIoU。CoffeeLoss 必须拆成互不依赖的 `wiou_ramp()`、`tail_ramp()`、`pair_ramp()`；否则 L0-L4 的单因素对照不成立。

由于 `normL1` 仍会直接响应边界偏差，必须监控 `box/WIoU` 与 `normL1` 的量级。如果 WIoU 改善 mAP50、但 mAP50-95 和 L1 明显恶化，应先调整 WIoU blend/focus，而不是继续加结构。

### 13.4 修正后的 TailBCE

去掉跨类别均值归一化，让最大类保持 1，其余类只上调不下调：

```math
w_c=clip\left((n_{max}/n_c)^{0.5},1.0,2.4\right)
```

```math
w_c(t)=1+ramp(t;0.10,0.60)(w_c-1)
```

候选最终倍率为：

```text
ANT_AB 1.000, ANT_CD 1.799, BLS_AB 2.013,
BLS_CD 2.400, CR 1.676, SM 2.039
```

保持当前 soft target 的保护方式：对正类 BCE 的额外倍率再乘任务对齐质量 `q=target_scores`，而不是把低质量正锚点无条件放大到完整类别权重。负类 BCE 不按类别全局放大，避免海量背景负样本改变整个分类损失尺度。

首轮只比较 cap `1.8` 与 `2.4`，其余超参数固定。repeat sampler 不作为默认项，因为重复抽取也会重复尾类错标；只有 TailBCE 对人工复核子集仍表现为尾类欠拟合时才单独测试 sampler。

### 13.5 质量门控 PairMargin

TailBCE 只增强真实类的绝对 BCE 梯度，不能直接要求它的 logit 高于 ANT。对当前两组混淆矩阵都确认的方向，增加相对间隔：

| 真实类 y | 竞争类 c | 启用依据 |
|---|---|---|
| ANT_CD | ANT_AB | 两次为 0.09/0.10 |
| BLS_AB | ANT_AB | 两次均为 0.19 |
| CR | ANT_AB | 两次为 0.12/0.17 |
| BLS_CD | ANT_CD | 两次为 0.12/0.08 |

`BLS_CD -> ANT_AB` 只在一组结果明显，不进入初始硬编码。后续 pair 列表只能由当前 6 类、固定阈值混淆矩阵更新，不能凭病名主观添加。

对正锚点 `i`，令 `y_i=argmax(target_scores_i)`、`q_i=max(target_scores_i)`。只保留 `fg_mask` 且 `q_i>=0.30` 的锚点：

```math
m_{y,c}=clip\left(0.15+0.05\log\frac{n_c+\epsilon}{n_y+\epsilon},0.15,0.30\right)
```

```math
L_{Pair}=\frac{\sum_i q_i\sum_{c\in C(y_i)}
softplus\left(10[m_{y_i,c}-(z_{i,y_i}-z_{i,c})]\right)/10}
{\sum_i q_i+\epsilon}
```

```math
\lambda_{pair}(t)=0.20\cdot ramp(t;0.10,0.60)
```

这种设计只在可信正锚点上提高真实类相对竞争 ANT 的距离，不会对所有 ANT 样本做全局惩罚。若直接降低所有 ANT logit，会伤害数量最多但本身有效的 ANT_AB 真阳性，因此禁止使用固定 ANT logit penalty。

### 13.6 不采用的分类损失改动

- 不在第一轮把 BCE 全面换成 Focal/Varifocal/ASL；它们同时改变全部正负锚点，无法判断 ANT 改善来自类别去偏还是 hard-negative 重加权。
- 不在 noisy label 上使用大幅 label smoothing；它会进一步缩小前期相似病种的 logit 间隔。
- 不同时使用 TailBCE、repeat sampler、PairMargin 和 BgMix；每项先做单因素实验。

CoffeeLoss 第一轮默认值：

```yaml
tail_power: 0.5
tail_weight_min: 1.0
tail_weight_max: 2.4
tail_ramp_start: 0.10
tail_ramp_end: 0.60
pair_lambda_max: 0.20
pair_quality_min: 0.30
pair_margin_base: 0.15
pair_margin_kappa: 0.05
pair_margin_min: 0.15
pair_margin_max: 0.30
pair_softplus_beta: 10.0
wiou_ramp_start: 0.10
wiou_ramp_end: 0.60
```

### 13.7 实现和单元测试契约

损失代码保持在实验目录，不修改仓库根的全局 loss：

```text
coffee26n_experiment/loss/
  coffee_loss.py
  loss_extensions.py
  pair_config.yaml
  test_coffee_loss.py
```

`coffee_loss.py` 继承当前 `v8DetectionLoss/BboxLoss/E2ELoss`，只替换三处：WIoU box 项、TailBCE 分类项、PairMargin 分类附加项。每个 epoch 的 `update()` 同时推进原生 o2m/o2o 比例和三个独立 ramp。

必须通过以下测试后才能训练：

1. 当前计数生成的六个 TailBCE 权重与文档数值误差 `<1e-4`；
2. `q<0.30` 或非前景锚点的 PairMargin 严格为 0；
3. 对 `BLS_AB -> ANT_AB` 违例样本，梯度满足 `dL/dz_BLS_AB<0`、`dL/dz_ANT_AB>0`；
4. 达到 margin 后 PairMargin 接近 0，且无配置 pair 的类别不产生该项；
5. `reg_max=1` 时第三项走 normL1，`reg_max>1` 的通用测试仍走 DFL；
6. one-to-many 与 one-to-one 都能记录非零 TailBCE/PairMargin，且总损失有限、反向无 NaN；
7. 分别关闭 WIoU、TailBCE、PairMargin 时，其他项的 ramp 和数值不发生联动。

## 14. BgMix：只抑制框外背景捷径

### 14.1 必要性和边界

两组原生基线的 unmatched/background 假阳性中，ANT_AB 占 `0.30/0.42`、ANT_CD 占 `0.24/0.25`。这不足以证明具体像素注意区域，但足以要求检查模型是否利用了拍摄台、盆、土壤、光照等框外背景。当前没有叶片/病斑分割 mask，因此不允许构造声称能定位病斑的无监督空间注意力。

`BgMix` 利用现有整叶框做最低风险的背景干预：框内像素与标签完全不变，只改变所有 GT 框之外的区域。它不增加推理参数，也不修改验证集。

### 14.2 精确工作流

每张训练图在所有几何增强和 box 坐标更新之后、归一化之前执行：

1. 将该图所有 GT `xyxy` 框向外扩张图像宽高的 `2%`，取并集作为 preserve mask；
2. 在 preserve 边界使用 `16 px` 余弦羽化，避免矩形接缝；
3. 以 `p=0.25` 只处理 mask 外部；
4. 外部变换在高斯模糊、轻度降饱和、亮度/对比度扰动中随机选择，禁止粘贴含未知病叶的其他图像；
5. mask 内保持逐像素不变，boxes/classes 不变；Mosaic 图保留所有叶框并集；
6. 每个 run 记录应用次数、变换类型和 preserve 面积比例。

首轮建议范围：

```yaml
bgmix_p: 0.25
bgmix_box_expand: 0.02
bgmix_feather_px: 16
bgmix_blur_kernel: 21
bgmix_blur_sigma: [3.0, 6.0]
bgmix_saturation_gain: [0.6, 0.9]
bgmix_value_gain: [0.8, 1.2]
```

BgMix 只能削弱框外背景相关性，不能分离整叶框内部的大量正常叶肉。框内问题由晚置 DetailBlock 的有界有符号残差、P4/P5 上下文和 CoffeeLoss 共同处理。

### 14.3 更重的一致性损失暂不进入主线

若 BgMix 单因素明确降低 ANT background FP，且主指标不退化，才可测试 `BgConsistency`：同一图生成两个仅框外不同的视图，对匹配正锚点的类别 logit 加一致性约束。它需要双前向并改变训练目标，首轮不采用。若未来测试，必须独立记录训练吞吐和显存，且不能与 PairMargin 第一次同时加入。

## 15. 实验假设和变量

### 15.1 研究问题

在参考咖啡 profile 上，受保护的残差注意力桥、晚置细节卷积、P4/P5 受限大核、抗混叠下采样和定向 CoffeeLoss，是否能在不引入 P2 Detect 和无监督空间注意力的情况下，提高前期相似病种、尾类和跨病程样本的识别能力，并减少 ANT 吸附和 ANT 假阳性？对其他数据集，研究问题应自动替换为该 profile 中实际存在的类别不均衡、混淆对和背景假阳性证据；若没有这些证据，对应损失/增强保持关闭。

### 15.2 假设

- H1：`Residual + RoleAttention` 同时存在时，优于直接覆盖 P5 或固定平均融合。
- H2：标准 Conv 主路径优于全网 LDSConv，尤其改善颜色相近类别的混淆。
- H3：晚置 DetailBlock 优于早期原图 ELTEB，因为它先获得语义再精炼纹理。
- H4：MixDown 能减少下采样混叠，同时不破坏原生 Conv 的预训练能力。
- H5：ContextBlock 只对跨病程变化提供增益；若与 C2PSA 重复，其门和 selector 会显示低贡献。
- H6：MidBlock 的 P4 半通道大核能降低 `BLS_AB/CR -> ANT_AB`，而不会像 P3 全大核一样抹平局部证据。
- H7：修正后的 TailBCE 优于现有均值归一化 ProgLoss，尤其改善 CR、BLS_AB 和 SM。
- H8：PairMargin 能进一步降低已确认的尾类->ANT 混淆，但不会显著伤害 ANT_AB/ANT_CD AP。
- H9：BgMix 能降低 ANT 主导的 background FP；若无改善，说明主要问题位于框内或检测匹配，而不是外部背景。

### 15.3 变量

| 类型 | 内容 |
|---|---|
| 自变量 | bridge residual、RoleAttention、DetailBlock、MixDown、MidBlock、ContextBlock、TailBCE、PairMargin、BgMix |
| 主要因变量 | 三种子平均 mAP50、mAP50-95、macro Recall、逐类 AP |
| 特定因变量 | `ANT_CD/BLS_AB/CR -> ANT_AB`、`BLS_CD -> ANT_CD`、逐类 background FP、pair violation rate |
| 成本因变量 | 参数、GFLOPs、显存、训练吞吐、目标设备 latency |
| 控制变量 | 数据快照、split、imgsz、预训练权重、增强、优化器、batch、epoch、评估脚本 |
| 混杂因素 | 数据 YAML 被覆盖、不同类别口径、不同输入尺寸、自动 batch、不同 fitness、混淆矩阵阈值不同、同时改变 loss 和 sampler |

## 16. 严格消融顺序

### 16.1 第一阶段：重新建立当前 profile 的控制组

| ID | 结构 | 目的 |
|---|---|---|
| B0 | 原生 YOLO26n + 当前代码的 WIoU/ProgLoss | 当前结构消融基线，必须保存真实 loss 配置 |
| B1 | 旧 V2：BoundedELTEB + ResidualEdgeLGMSFBridge + LDSConv | 旧最佳思想在新 6 类上的迁移基线 |

B0、B1 都运行 seeds `0,1,2`。没有这一步，后续不能把其他 `nc`、其他输入尺寸或旧数据快照的结果当作当前 profile 阈值。通用实现必须先生成一个与输入 dataset YAML 同 `nc` 的原生 YOLO26n control。

### 16.2 第二阶段：证明 residual 和 attention 各自有效

使用原生 backbone、原生 neck，只有 bridge 不同。先 seed 0 筛选：

| ID | P5 接入 | 融合器 |
|---|---|---|
| R0 | 直接替换 | 固定 0.5/0.5 |
| R1 | 残差 gate | 固定 0.5/0.5 |
| R2 | 直接替换 | RoleAttention |
| R3 | 残差 gate | RoleAttention，即 ResBridge |

目标不是默认宣布 R3 最好，而是用 2x2 设计分别估计 residual 和 attention 的主效应。R3 只有达到筛选门槛才进入三种子验证。

### 16.3 第三阶段：卷积和排序

以获胜 bridge 为唯一固定基线：

| ID | 唯一改变 |
|---|---|
| C0 | neck native Conv |
| C1 | neck LDSConv |
| C2 | neck SCDown |
| C3 | neck MixDown |
| D0 | 无 P3 额外模块 |
| D1 | 早期 BoundedELTEB |
| D2 | 晚置 DetailBlock |

先完成 C0-C3，再完成 D0-D2。不能把 C3 和 D2 同时作为第一次测试。

### 16.4 第四阶段：backbone 下采样条件消融

只有 neck 的 C3 `MixDown` 已通过后，才保持其余结构不变，逐个测试 backbone 的 stride-2 节点：

| ID | 唯一改变 | 目的 |
|---|---|---|
| X0 | backbone 全部原生 Conv | 控制组 |
| X3 | node 3 改 MixDown | 检查进入 P3 时的弱纹理抗混叠 |
| X5 | node 5 改 MixDown | 检查 P3->P4 中尺度过渡 |
| X7 | node 7 改 MixDown | 检查 P4->P5 全叶语义过渡 |

三者首轮不能组合。每个 `MixDown.base` 必须迁移对应原生 Conv 权重；若 node 3 造成前期病斑 AP 下降，立即停止向更浅层扩展。这里是在不破坏原生标准卷积主支路的前提下加强联通，不测试全 backbone LDSConv/SCDown。

### 16.5 第五阶段：P4/P5 大核上下文

| ID | 唯一改变 |
|---|---|
| K0 | Coffee26n core，无 MidBlock/ContextBlock |
| K1 | core + MidBlock(k=9) |
| K2 | core + ContextBlock(k=7) |
| K3 | core + ContextBlock(k=13) |
| K4 | core + K1 + 获胜的 P5 Context，仅在各自单独通过后 |

K1-K3 先 seed 0；仅获胜项进入 seeds 1/2。K4 不是默认结果，必须证明两个模块组合后仍有附加增益。若门接近 0 或混淆对不降，删除对应模块。

### 16.6 第六阶段：类别不均衡和 ANT 去偏

结构完全锁定后，按下表逐级测试：

| ID | 唯一损失配置 | 目的 |
|---|---|---|
| L0 | 原生 CIoU + BCE + normL1 | 原生 loss 控制 |
| L1 | 现有 WIoU + 现有均值归一化 ProgLoss | 旧 loss 控制 |
| L2 | WIoU + 修正 TailBCE，cap=1.8 | 验证去掉均值归一化 |
| L3 | WIoU + 修正 TailBCE，cap=2.4 | 验证更强尾类权重 |
| L4 | L3 + PairMargin | 验证对 ANT 的相对间隔 |

只有 L3 优于 L2，才采用 `2.4`。只有 L4 在固定阈值下降低指定混淆对、同时 ANT 类 AP 不下降超过 `0.02`，才保留 PairMargin。repeat sampler 仅作为 L2/L3 无效后的独立备选，不与 L4 同时首次测试。

### 16.7 第七阶段：背景抑制

| ID | 唯一改变 |
|---|---|
| G0 | 获胜结构 + 获胜 loss，无 BgMix |
| G1 | G0 + BgMix(p=0.25) |
| G2 | G0 + BgMix 获胜参数 + BgConsistency，仅后续可选 |

G1 的首要指标是固定置信度/固定 recall 下逐类 false positives，而不是只看归一化 confusion matrix。G2 双前向成本高，不进入第一轮主线。

## 17. 固定训练工作流

### 17.1 数据冻结

训练前由 `--data` 指定的数据集生成并保存。目录名可以保持 `coffee26n_experiment` 以便延续本方案，但其中的 data lock 不能依赖咖啡路径：

```text
coffee26n_experiment/data_lock/
  dataset_yaml_copy.yaml
  train_files.txt
  val_files.txt
  test_files.txt
  label_sha256.csv
  class_counts.json
  box_statistics.json
  confusion_audit_pairs.json
```

`dataset_yaml_copy.yaml` 必须记录解析后的绝对/规范化路径、`nc`、`names`、split 文件列表和源 YAML SHA256；`class_counts.json`、`box_statistics.json` 和 `confusion_audit_pairs.json` 都由当前数据重新生成。若 profile 没有混淆对，`confusion_audit_pairs.json` 必须是空列表，不能自动套用 ANT 对。

每个 run 的输出中复制同一份 `data_lock_id`。如果数据变化，必须建立新实验批次，不能继续和旧批次排序。

### 17.2 固定配方

下面是当前 `coffee_ant6` 参考 profile 的第一轮结构消融配方，用于保证咖啡历史结果可比；它不是通用训练入口的强制输入尺寸或 batch。generic profile 应使用调用方的 `--imgsz`，未提供时采用实现明确记录的通用默认值，并在所有对照中保持一致。

第一轮咖啡结构消融建议统一使用：

```text
epochs=300
imgsz=960
batch=64
nbs=64
optimizer=MuSGD
lr0=0.01
lrf=0.01
momentum=0.9
weight_decay=0.0005
warmup_epochs=4
patience=80
mosaic=0.7
close_mosaic=150
hsv_h=0.005
hsv_s=0.20
hsv_v=0.10
scale=0.35
erasing=0.0
amp=True
seeds=0,1,2
```

原因：前期病种颜色差异细微，过强 HSV 会抹去信号；有效病斑区域小，随机擦除可能制造“病斑被擦掉但标签不变”的额外噪声；Mosaic 在前半程保留泛化作用，后半程回到自然整叶图。

此配方本身也属于待验证控制条件。所有结构必须使用同一配方，不能只给新模型使用更好的增强。

### 17.3 预期训练入口（实现后的 CLI 契约）

```powershell
python coffee26n_experiment/train_coffee26n.py --data <dataset.yaml> --variant native --profile generic --loss native --bgmix off --seed 0 --dry-run
python coffee26n_experiment/train_coffee26n.py --data <dataset.yaml> --variant core --profile generic --loss native --bgmix off --seed 0 --dry-run
python coffee26n_experiment/train_coffee26n.py --data <dataset.yaml> --variant p4mid --profile generic --loss native --bgmix off --seed 0
python coffee26n_experiment/train_coffee26n.py --data <dataset.yaml> --variant p5lk13 --profile generic --loss native --bgmix off --seed 0
python coffee26n_experiment/train_coffee26n.py --data <dataset.yaml> --variant full --profile generic --loss coffee_l4 --bgmix 0.25 --seed 0

# 当前咖啡证据复现示例；路径和类别只来自 profile，不得写入代码默认值。
python coffee26n_experiment/train_coffee26n.py --data coffee_self_sum/coffee_self_sum.yaml --variant core --profile coffee_ant6 --loss native --bgmix off --seed 0 --dry-run
```

训练脚本尚未生成，因此这些命令是后续实现契约，不是当前可运行命令。`<dataset.yaml>` 必须由调用方替换；缺少 `--data` 时脚本应立即报错，而不是回退到咖啡数据集。

## 18. 输出和监控契约

每个 run 必须生成：

```text
args.yaml
model.yaml
model_summary.txt
param_budget.json
pretrain_transfer.json
data_lock_id.txt
results.csv
per_class_metrics.csv
confusion_matrix.csv
confusion_pairs.csv
class_false_positives.csv
module_monitor.csv
loss_monitor.csv
class_weight_monitor.csv
pair_monitor.csv
latency.json
resolved_config.yaml
run_manifest.json
weights/best.pt
weights/last.pt
```

`module_monitor.csv` 至少包含：

```text
epoch
bridge_gate
role_p3_weight
role_p5_weight
role_entropy
detail_gate
mixdown1_gate
mixdown2_gate
mid_gate
context_gate
context_local_weight
context_large_weight
context_global_weight
pair_loss
bgmix_applied_images
precision
recall
mAP50
mAP50-95
```

逐类别、逐混淆对字段不得做成固定列名：`class_weight_monitor.csv` 使用长表 `epoch,class_id,class_name,count,base_weight,effective_weight`；`pair_monitor.csv` 使用长表 `epoch,true_id,true_name,rival_id,rival_name,margin,eligible_count,violation_rate,pair_loss`。这样 `nc=1/6/80` 都能复用。上面出现的 ANT 字段只属于旧咖啡报告，不进入通用 schema。

实时监控：

- 每个 epoch 检查 `results.csv` 是否增长；
- GPU 利用率和显存每 60 秒记录一次；
- 连续 20 个 epoch 无 mAP50 改善只告警，不自动终止；
- NaN、loss 爆炸或模型图不匹配时立即停止该 run；
- 不自动重试失败实验，先记录真实错误。

## 19. 评价指标和晋级门槛

### 19.1 指标优先级

1. 当前固定 6 类 validation 的 macro mAP50；若另建人工复核 hard subset，必须单独报告，不能把它称为整个 clean val；
2. mAP50-95；
3. macro Recall；
4. BLS_AB、BLS_CD、SM 尾类平均 AP；
5. 指定尾类->ANT 混淆对下降比例；
6. 固定阈值/固定 recall 下 ANT_AB、ANT_CD 假阳性数；
7. 参数、GFLOPs、显存和真实 latency。

因为框覆盖整叶且边界存在噪声，mAP50-95 不能单独代表病害分类能力；但也不能只看 mAP50 而忽略定位退化。

### 19.2 seed 0 筛选门槛

相对同阶段控制组同时满足：

- `Delta mAP50 >= +0.005`，或尾类平均 AP `>= +0.010`；
- mAP50-95 不下降超过 `0.003`；
- macro Recall 不下降超过 `0.005`；
- 没有单类 AP 下降超过 `0.02`。
- 对 loss/BgMix 实验，至少一个预注册 ANT 混淆对相对下降 `>=10%`，且其他对不能明显恶化。

### 19.3 三种子保留门槛

- 平均 `Delta mAP50 >= +0.010`；
- 平均 `Delta mAP50-95 >= +0.008`；
- 尾类平均 AP `>= +0.015`；
- 至少 2/3 seeds 高于对应控制；
- 任一类别平均 AP 不下降超过 `0.02`；
- 参数必须小于运行时同 `nc` 构建的原生 YOLO26s；推荐不超过同 `nc` 原生 YOLO26n 的 `2.0x`，超过时必须单列精度/成本收益；
- GFLOPs 必须报告，不再与参数共用 `1.30x` 门槛；
- 目标设备端到端延迟不超过原生的 `1.25x`，除非 mAP50 增益超过 `0.02`。

三种子只用于判断工程稳定性，不声称完成严格统计显著性检验。应报告均值、标准差和逐 seed 原始值。

## 20. 针对六类数据问题的最终作用链

### 20.1 整叶框噪声

```text
原生 backbone 语义
 -> WIoU 降低异常框梯度主导
 -> C2PSA 全局关系
 -> ResBridge 小门残差
 -> neck 先语义融合
 -> DetailBlock 有界局部修正
```

核心不是生成未经监督的空间 mask，而是保持原生语义，再允许小幅局部修正。

### 20.2 有效区域少

```text
P3 120x120 保留局部证据
 -> DetailBlock 多形状卷积
 -> P3 Detect 使用细节
 -> MixDown 把细节逐级送到 P4/P5
```

### 20.3 前期病种相似

```text
backbone standard Conv 保留 RGB 跨通道联合关系
 -> P3 spot/line/halo 三分支
 -> P4 MidBlock 半通道 1x9->9x1 中尺度组合
 -> RoleAttention 按样本选择细节或语义通道
 -> PairMargin 约束已验证相似类与 ANT 的相对 logit
```

### 20.4 同病种前后期差距大

```text
早期局部证据: P3
中等扩散范围: P4
严重期和全叶状态: P5 + optional ContextBlock
 -> Detect 对三个尺度共同监督
```

### 20.5 类别不均衡和 ANT 吸附

```text
结构先固定
 -> TailBCE 去掉错误的跨类别均值归一化
 -> PairMargin 只作用于高质量正锚点和已确认混淆对
 -> 以 macro/逐类 AP、pair confusion 和 ANT AP 共同验收
```

### 20.6 外部背景捷径

```text
所有几何增强完成
 -> GT 框扩大 2% 并做 16px 羽化
 -> BgMix 只随机化框外背景
 -> 固定阈值统计逐类 background FP
 -> 推理阶段完全移除 BgMix
```

## 21. 禁止组合和停止条件

以下内容不进入第一版：

1. 不加入 P2 Detect；
2. 不恢复 P5->P3 BiBridge；
3. 不加入 SimAM；
4. 不同时使用早期 ELTEB 和晚置 DetailBlock；
5. 不同时把 ContextBlock、MidBlock、MixDown、DetailBlock 第一次叠加训练；
6. 不把 backbone 全部替换为 depthwise 卷积；
7. 不使用无门控的新分支直接覆盖 C2PSA P5；
8. 不把旧 9 类、5 类或不同 imgsz 结果放进当前 6 类排行榜。
9. 不全网替换 SiLU，不在残差输出投影后加 ReLU/SiLU；
10. 不在 P3 使用 9/13 大核，不使用会使总参数达到 YOLO26s 的密集大卷积；
11. 不对所有 ANT logit 施加固定惩罚，不凭病名主观扩充 PairMargin 类别对；
12. 不把 TailBCE、PairMargin、repeat sampler、BgMix 第一次同时启用。

停止或回退条件：

- attention entropy 长期 `<0.2`：融合塌缩，检查归一化或退回简单残差；
- bridge/context/mid gate 长期接近 0：模块无贡献，删除；
- DetailBlock gate 持续达到 `+/-0.25` 边界：检查噪声放大和上限；
- MixDown low gate 快速接近 1：检查低频分支是否覆盖细节；
- 训练集 AP 上升、固定 val 或人工复核 hard subset 的尾类 AP 下降：判定为长尾过拟合；
- PairMargin 使目标混淆下降、但 ANT_AB 或 ANT_CD AP 下降超过 `0.02`：降低 lambda/margin 或删除；
- BgMix 使 background FP 下降、但 mAP50-95 下降超过 `0.003`：检查框扩张和边缘伪影，不直接保留；
- 参数达到或超过运行时同 `nc` 原生 YOLO26s：无条件淘汰，不以精度增益豁免；
- 单 seed 增益小于种子间标准差：不得认定有效。

## 22. 后续代码生成顺序

后续实现必须按以下顺序推进：

1. 建立 `coffee26n_experiment` 自包含目录和 data lock；
2. 复制并锁定当前本地 Ultralytics 实现；
3. 实现 `DetailBlock`、`MixDown`、`ResBridge`、`MidBlock` 和 `ContextBlock` 别名；
4. 导出模块并修改 parse_model；
5. 生成 `native/control/core/p4mid/p5lk/full` YAML；
6. 实现精确预训练层映射；
7. 写 `verify_structure.py`，先通过 shape、gate、Detect、参数硬门槛和 forward 验证；
8. 实现独立 `CoffeeLoss`，补齐 reg_max=1 的 normL1 命名、TailBCE、PairMargin 单元测试；
9. 实现 BgMix，并验证 preserve mask 内像素逐元素不变；
10. 实现带 variant/loss/bgmix 参数的统一训练入口；
11. 先跑 B0/B1，再跑 R0-R3；
12. bridge 通过后才跑 C0-C3、D0-D2，再逐节点跑 X3/X5/X7；
13. core 通过后分别测试 MidBlock 与 ContextBlock，最后才允许组合；
14. 结构锁定后按 L0-L4 调整 loss，再跑 G0-G1；
15. 最终模型运行 seeds 0/1/2 和独立 test；
16. 生成完整对照表、混淆对、假阳性和模块监控解释。

## 23. 最终候选的一句话定义

**Coffee26n = 原生 YOLO26n C3k2/SPPF/C2PSA backbone + 带 RoleAttention 的 P3->P5 ResBridge + 原生 top-down C3k2 + 晚置 P3 DetailBlock + neck MixDown + 条件 P4 MidBlock(k=9) + 条件 P5 ContextBlock(k=13) + 原生 P3/P4/P5 end-to-end Detect；训练期再按顺序验证 CoffeeLoss 与 BgMix。**

这条路线保留历史结构真正有价值的部分：P3/P5 角色分工、通道注意力、受保护残差、原生 SiLU、C2PSA/PAN 和 YOLO26 end-to-end loss 双分支；同时去掉与整叶框噪声问题不匹配的直接 P5 替换、全网 LDSConv、早期无语义纹理注入、P2 Detect、反向空间反馈和无证据的全网激活替换。最终参数必须硬性低于运行时以相同 `nc` 构建的原生 YOLO26s，但所有条件模块仍以实测收益决定去留。

## 24. 通用实现契约

本节是代码生成的强制接口。实现者可以调整内部变量名，但不能改变以下外部行为、张量关系、默认值和失败策略。

### 24.1 不变量清单

| 编号 | 必须保持的不变量 | 验证位置 |
|---|---|---|
| I1 | 模型家族仍是原生 YOLO26n，backbone 的 `C3k2 -> SPPF -> C2PSA` 语义主路径不被替换 | `test_structure.py`、模型摘要 |
| I2 | 检测输出始终是 P3/P4/P5 三路，stride 为 `8/16/32`；不加入 P2 Detect | 结构测试、dry-run |
| I3 | 所有新增特征分支是有界近恒等残差，初始输出接近原生控制组 | gate 初始化测试、前向差异测试 |
| I4 | 跨尺度融合只生成通道级权重，不生成无监督 `H x W` 空间 mask | 静态审查、模块单测 |
| I5 | 默认激活保持 SiLU；残差投影和线性混合投影使用 `act=False` | 模块构造测试 |
| I6 | 参数硬上限由同 `nc`、同 scales 的原生 YOLO26s 动态计算 | `verify_structure.py` |
| I7 | `--data` 是必填输入；模型、loss、增强不包含固定数据集路径、类别名或 `nc=6` | CLI 测试、静态扫描 |
| I8 | generic profile 默认关闭 PairMargin 和 BgMix；请求开启但配置无效时必须报错 | profile/配置测试 |
| I9 | CoffeeLoss 同时包住 YOLO26 end-to-end 的 one-to-many 与 one-to-one 分支 | loss 集成测试 |
| I10 | 代码只写入实验包，不修改仓库根 `ultralytics/` | 路径检查、最终报告 |

### 24.2 配置优先级和解析结果

配置合并顺序固定为：

```text
CLI 参数 > profile YAML > generic.yaml 默认值 > 由 dataset YAML 计算的事实
```

规则如下：

1. `--data` 没有默认值，缺失或路径不存在时在构建模型前退出，退出码非零。
2. 先解析 dataset YAML，得到 `nc`、`names`、split 路径、标签统计和 `data_lock_id`；再读取 profile。
3. profile 中声明的 `expected_nc`、`expected_names` 或 `dataset_yaml` 若与实际不一致，必须报出字段级差异并退出；不能静默降级为 generic。
4. CLI 显式给出的 `--loss`、`--bgmix`、`--imgsz`、`--weights` 等值覆盖 profile，但覆盖后仍要重新校验范围。
5. 每次运行保存合并后的 `resolved_config.yaml`，并额外保存 `args.yaml`（原始 CLI）和 `profile_snapshot.yaml`（未合并 profile）。后续复现实验只读取 `resolved_config.yaml`，不能依赖脚本当前默认值。
6. 任何未识别的配置键、variant、loss 名或 profile 键都应立即报错，禁止“忽略未知字段”。

### 24.3 CLI 契约

训练入口为 `coffee26n_experiment/train_coffee26n.py`。以下参数是最小稳定接口；实现可以增加参数，但不得删除或改变这些语义。

| 参数 | 类型 | 必填 | 默认 | 约束 |
|---|---|---:|---|---|
| `--data` | path | 是 | 无 | 必须是可解析的 YOLO detect dataset YAML |
| `--variant` | enum | 否 | `native` | `native,b0,r3,core,p4mid,p5lk13,full` |
| `--profile` | name/path | 否 | `generic` | generic 或存在的 profile YAML |
| `--loss` | enum | 否 | `native` | `native,coffee_l1,coffee_l2,coffee_l3,coffee_l4` |
| `--bgmix` | `off` 或 float | 否 | `off` | `0 <= p <= 1`；generic 默认 off |
| `--weights` | path or `none` | 否 | `yolo26n.pt` | 本地文件优先；dry-run 不联网下载 |
| `--imgsz` | int or `H,W` | 否 | profile/default | 每个维度为正；结构检查不得假定 960 |
| `--seed` | int | 否 | `0` | 保存到 resolved config |
| `--dry-run` | flag | 否 | false | 只构图、迁移、单批前向和验收，不训练 |

建议同时保留 `--device`、`--batch`、`--workers`、`--epochs`、`--project`、`--name` 等原生训练参数，并将其写入 resolved config。`--bgmix 0` 与 `--bgmix off` 等价；非零 `--bgmix` 必须要求 `augmentations/bgmix.py` 已注册。

variant 的唯一改变必须可追溯到下表：

| variant | 结构 | 用途 |
|---|---|---|
| `native`/`b0` | 原生 YOLO26n，使用同 `nc` 动态生成 control | 基线和参数上限 |
| `r3` | 原生 neck + ResBridge | residual/attention 2x2 后的候选 |
| `core` | ResBridge + DetailBlock + MixDown，无 Mid/Context | 主核心结构 |
| `p4mid` | core + MidBlock | P4 单因素 |
| `p5lk13` | core + ContextBlock(k=13) | P5 单因素 |
| `full` | core + 已分别通过的 MidBlock/ContextBlock | 最终候选，不得默认跳过单因素 |

loss 名称与唯一行为固定为：

| `--loss` | box 分类/回归行为 | PairMargin | 用途 |
|---|---|---:|---|
| `native` | 原生 CIoU + BCE + `reg_max=1` normL1 | off | 通用默认和 L0 |
| `coffee_l1` | 现有 WIoU + 旧均值归一化 ProgLoss + normL1 | off | 只用于复现旧 L1 控制，不作为新默认 |
| `coffee_l2` | WIoU + 新 TailBCE(cap=1.8) + normL1 | off | 去掉均值归一化的第一档 |
| `coffee_l3` | WIoU + 新 TailBCE(cap=2.4) + normL1 | off | 更强尾类上调 |
| `coffee_l4` | 与 L3 相同 | on | 仅 profile 有已验证非空 pairs 时允许 |

选择 `coffee_l2/l3` 时即使 profile 的 `tail_bce.enabled=false`，CLI 也会按该 loss 明确启用 TailBCE 并使用当前标签计数；选择 `native` 时 profile 中的 TailBCE/PairMargin 配置不应偷偷生效。选择 `coffee_l4` 但没有有效 pair 必须失败。BgMix 始终是正交的独立 CLI/profile 变量，不由 loss 名称隐式开启。

### 24.4 Profile schema

`configs/profiles/generic.yaml` 必须是无咖啡词、无本地数据路径的可复用默认配置：

```yaml
schema_version: 1
name: generic
dataset_yaml: null
expected_nc: null
expected_names: null
tail_bce:
  enabled: false
  counts: null
  cap: 1.8
  exponent: 0.5
pair_margin:
  enabled: false
  pairs: []
  quality_threshold: 0.30
  margin_min: 0.15
  margin_max: 0.30
  softplus_beta: 10.0
bgmix:
  enabled: false
  p: 0.0
  box_expand: 0.02
  feather_px: 16
  blur_kernel: 21
structure:
  context: false
  context_kernel: 13
  mid: false
  mid_kernel: 9
  mixdown: true
budget:
  compare_family: yolo26s
  recommended_n_multiplier: 2.0
```

当前咖啡证据另存为 `configs/profiles/coffee_ant6.yaml`，示例结构如下。这里的路径、名称、计数和 pair 只属于该 profile：

```yaml
schema_version: 1
name: coffee_ant6
dataset_yaml: coffee_self_sum/coffee_self_sum.yaml
expected_nc: 6
expected_names: [ANT_AB, ANT_CD, BLS_AB, BLS_CD, CR, SM]
tail_bce:
  enabled: true
  counts: {ANT_AB: 1443, ANT_CD: 446, BLS_AB: 356, BLS_CD: 208, CR: 514, SM: 347}
  cap: 2.4
  exponent: 0.5
pair_margin:
  enabled: true
  pairs:
    - {true_name: ANT_CD, rival_name: ANT_AB}
    - {true_name: BLS_AB, rival_name: ANT_AB}
    - {true_name: CR, rival_name: ANT_AB}
    - {true_name: BLS_CD, rival_name: ANT_CD}
  quality_threshold: 0.30
  margin_min: 0.15
  margin_max: 0.30
  softplus_beta: 10.0
bgmix:
  enabled: true
  p: 0.25
  box_expand: 0.02
  feather_px: 16
  blur_kernel: 21
```

实现必须把 `true_name/rival_name` 解析成当前 dataset 的整数 ID，并在 profile 与数据集名称不一致时失败；不能按数组位置猜测。

### 24.5 Dataset YAML 和标签验证

在任何模型构建前执行以下顺序：

1. 用 YAML 解析器读取 `path/train/val/test/names/nc`；支持 `names` 为 list 或整数键字典，统一成有序 `list[str]`。
2. 若 `nc` 存在，要求 `nc == len(names)`；若不存在，从 names 推导并把解析结果写入 `resolved_config.yaml`。
3. 解析每个 split 的图片和同名 label；对于本方案的 axis-aligned detect 任务，非空标签行必须严格为 5 列 `class cx cy w h`，class id 为整数且落在 `[0,nc)`，xywh 为有限值且归一化合法。空标签文件可以存在，但必须统计。需要 OBB、segment 或 pose 时应建立独立任务规格，不能让 6 列标签被本 detect 实现含糊接受。
4. 统计 `class_counts`、每类图片数、框面积/宽高分布，并记录空标签图片。统计必须来自当前 data lock，不能读取文档常数。
5. 检查 train/val/test 图片 hash 或显式 split manifest；发现跨 split 完全重复时警告并按项目策略失败，不要继续混合结果。
6. 验证 profile：`expected_nc`、`expected_names`、每个 pair 的名字和开关。generic profile 的 `pairs` 必须为空；请求 PairMargin 但 pair 为空时直接报错。
7. 将规范化后的文件列表、标签 SHA256、dataset YAML SHA256 写入 `data_lock/`，生成稳定 `data_lock_id`。

### 24.6 动态尺寸和类别规则

- 模型 YAML 可以在生成阶段使用占位 `nc`，但真正传给 `YOLO/Model` 的配置必须是已经注入整数 `nc` 的临时副本。
- `names` 只用于报告和 profile 映射，不进入模型结构参数；检测 head 的类别维度来自解析后的 `nc`。
- 结构测试至少覆盖 `nc=1,6,80` 和 `imgsz=640,960`；`80` 类测试可用合成 dataset YAML，不需要真实训练数据。
- 所有空间尺寸通过 forward 输出、stride 或 `torch.fx` 形状推断获得，不能在 Python 中写死 `120/60/30`。

## 25. 模块 API、排序和 `parse_model` 契约

### 25.1 构造函数和张量契约

下列签名是生成代码的基准。所有 `c1/c2` 必须为正整数，kernel 必须为奇数，stride/dilation 必须为正整数；非法值用 `ValueError` 失败。

```python
class ResBridge(nn.Module):
    def __init__(self, c3: int, c5: int, reduction: int = 8, gate_init: float = -3.0): ...
    def forward(self, xs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: ...

class DetailBlock(nn.Module):
    def __init__(self, c1: int, c2: int, reduction: int = 2,
                 line_kernel: int = 5, dilation: int = 2,
                 gate_max: float = 0.25): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MixDown(nn.Module):
    def __init__(self, c1: int, c2: int, gate_init: float = -4.0): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MidBlock(nn.Module):
    def __init__(self, c1: int, c2: int, kernel: int = 9,
                 gate_max: float = 0.20): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ContextBlock(nn.Module):
    def __init__(self, c1: int, c2: int, kernel: int = 13,
                 reduction: int = 4, gate_init: float = -3.0): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

Shape rules:

- `ResBridge` receives `[P3, P5]`, where spatial sizes differ by exactly a factor of four after its two stride-2 reductions; it returns P5 shape `(B,c5,h5,w5)`.
- `DetailBlock`, `MidBlock` and `ContextBlock` preserve spatial dimensions and return `c2` channels.
- `MixDown` halves each spatial dimension according to the same padding convention as native `Conv(k=3,s=2)` and returns `c2` channels.
- All modules accept AMP dtypes and devices inherited from inputs; gates are cast to the delta dtype without creating CPU tensors.
- Runtime shape mismatches must include module name, received shapes and expected relationship in the exception.

The implementation must retain the internal choices already specified in Sections 4--8: standard Conv in the narrow ResBridge down path, depthwise local/line branches only where stated, `act=False` on residual output projections, and bounded gates (`sigmoid` for nonnegative gates, `gate_max*tanh` for signed gates). Do not replace these with a generic attention or activation block.

### 25.2 YAML 排序和来源索引

The canonical full graph is the order in Section 9.1. YAML files must use explicit `from` indices and comments containing semantic names (`B3`, `B4`, `E5`, `T3`, `O4`, `O5`). The implementation must not infer a different graph from comments. Variant generation may remove optional nodes, but it must regenerate source indices and Detect sources rather than reusing stale integers.

After build, `verify_structure.py` must discover the three Detect input nodes by their runtime strides and report both semantic names and actual indices. A hard-coded assertion that only `[19,23,26]` is valid is forbidden because core variants have different indices.

### 25.3 `parse_model` behavior

1. Export all five classes from `local_ultralytics/ultralytics/nn/modules/__init__.py` and import them in the copied `nn/tasks.py` namespace used by the package.
2. Add ordinary `(c1,c2,...)` modules to the parser's `base_modules`: `MixDown`, `DetailBlock`, `MidBlock`, `ContextBlock`.
3. Keep custom modules out of `repeat_modules` unless the implementation explicitly proves repeated construction semantics. Canonical YAML uses `n=1`; if a custom module is requested with `n != 1`, fail loudly instead of silently changing depth.
4. Use a dedicated branch for ResBridge:

```python
elif m is ResBridge:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("ResBridge expects from=[P3_INDEX, P5_INDEX].")
    c3, c5 = ch[f[0]], ch[f[1]]
    c2 = c5
    args = [c3, c5, *args]
```

5. Apply the same width/depth scaling rules as native YOLO26 to declared `c2` for ordinary modules. ResBridge output channel is derived from `c5`, not scaled a second time from a YAML `c_out`.
6. Preserve native handling for `Concat`, `Detect`, `SPPF`, `C2PSA`, `C3k2` and `nn.Upsample`. Do not fork parser behavior outside the copied package.
7. After parsing, assert all channel counts are positive, Detect receives exactly three tensors, and no Detect source has stride 4.

## 26. 源码隔离和预训练迁移

### 26.1 复制和导入隔离

The experiment package is portable and must not rely on import order from the checkout root. The entrypoint should resolve its own root and insert only the package-local implementation before importing Ultralytics:

```python
PACKAGE_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = PACKAGE_ROOT / "local_ultralytics"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(LOCAL_ULTRALYTICS))
from ultralytics import YOLO  # resolved from LOCAL_ULTRALYTICS
```

入口必须先插入 `PACKAGE_ROOT`，再把 `LOCAL_ULTRALYTICS` 放到更高优先级；这样 package-local `tasks.py` 可以在 `DetectionModel.init_criterion()` 内延迟导入 `loss.coffee_loss`，同时 Ultralytics 仍从 local 副本解析。延迟导入可以避免 `tasks.py <-> loss.py` 的模块初始化环；不要在 `tasks.py` 顶层先导入 CoffeeLoss。

启动时打印并记录 `ultralytics.__file__`、`ultralytics.nn.tasks.__file__`、`ultralytics.nn.modules.block.__file__` 和 `loss.coffee_loss.__file__`。前三者若不在 `local_ultralytics/`，或最后一个不在实验包 `loss/`，必须在训练前失败。禁止修改或 monkey-patch 根目录 `ultralytics/`。

### 26.2 `source_manifest.json`

Because this checkout is not a Git repository, every copied source file must be auditable by hash. Generate:

```json
{
  "schema_version": 1,
  "generated_at": "ISO-8601",
  "package_root": "...",
  "native_source_root": "...",
  "native_revision": "ultralytics-8.4.43",
  "files": [
    {
      "logical_path": "ultralytics/nn/tasks.py",
      "source_path": "...",
      "package_path": "local_ultralytics/ultralytics/nn/tasks.py",
      "sha256": "64-hex",
      "size_bytes": 12345
    }
  ]
}
```

The manifest must include every `.py` and model YAML copied into the package, not only custom files. A verification command must recompute hashes and exit nonzero on mismatch.

### 26.3 预训练迁移报告

Load the local `--weights` checkpoint through an explicit migration function. The report `pretrain_transfer.json` must contain:

```json
{
  "source_weights": "...",
  "source_model": "yolo26n",
  "source_nc": 80,
  "target_nc": 6,
  "exact_matched_keys": [],
  "remapped_keys": [{"source": "...", "target": "...", "numel": 0}],
  "skipped_shape_keys": [],
  "missing_target_keys": [],
  "new_parameter_keys": [],
  "expected_head_mismatches": [],
  "source_numel_considered": 0,
  "target_numel_loaded": 0,
  "coverage_percent": 0.0,
  "gate_initialization": {}
}
```

For the full graph, use the semantic mapping in Section 11.2 (`target 0..10`, `target 15/18/22/26`, and the native branches inside MixDown). For core/p4mid/p5lk variants, generate a variant-specific mapping from semantic node names; never apply full-node integers blindly. Same-shape native layers must be copied exactly. Classification heads with different `nc` are expected mismatches and must be listed, not hidden by `strict=False`.

`coverage_percent = 100 * target_numel_loaded / eligible_native_target_numel`；分母只包含目标图中应当从原生 YOLO26n 继承的参数，不包含新增模块，也不包含因 `nc` 不同而预期重建的分类参数。报告还要给出包含全部目标参数的第二个比例 `loaded_over_total_target_percent`，避免通过排除过多参数制造虚高覆盖率。

接受迁移前，验证第 11.3 节的全部 gate，逐元素抽检已复制 tensor；若 coverage 低于配置阈值，或某个应保持原生的层无明确理由缺失，必须失败。

## 27. CoffeeLoss 和 BgMix 实现契约

### 27.1 Loss adapter 和 end-to-end 双分支

损失代码保存在 `coffee26n_experiment/loss/`，并把该目录建成含 `__init__.py` 的显式 Python package。当前本地 YOLO26 的真实接口是：`E2ELoss.__init__()` 用 `loss_fn(model,tal_topk,tal_topk2)` 分别创建 one-to-many 和 one-to-one，前向时调用两个实例的 `.loss(preds,batch)`；trainer 每个 epoch 调用一次顶层 criterion 的无参数 `.update()`。因此实现必须使用下面的继承/factory 关系，不能只写一个独立 `CoffeeLoss.__call__()` 后假设 E2ELoss 会自动使用它：

```python
import functools

class CoffeeLoss(v8DetectionLoss):
    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        *,
        profile: Mapping[str, Any],
        config: Mapping[str, Any],
    ): ...

    def update_progress(self, progress: float) -> None: ...

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple: ...

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class CoffeeE2ELoss(E2ELoss):
    def __init__(self, model, *, profile, config):
        factory = functools.partial(CoffeeLoss, profile=profile, config=config)
        super().__init__(model, loss_fn=factory)
        self.total_epochs = max(int(model.args.epochs), 1)

    def update(self) -> None:
        super().update()
        progress = min(self.updates / max(self.total_epochs - 1, 1), 1.0)
        self.one2many.update_progress(progress)
        self.one2one.update_progress(progress)
```

package-local `DetectionModel.init_criterion()` 必须在函数内部延迟执行 `from loss.coffee_loss import CoffeeE2ELoss, CoffeeLoss`，并根据 resolved loss 配置返回：Coffee loss 开启且 `end2end=True` 时返回 `CoffeeE2ELoss`；Coffee loss 开启且 `end2end=False` 时返回 `CoffeeLoss`；`loss=native` 时完全返回原生 criterion。导入后立即校验模块 `__file__` 位于实验包。profile/config 可以在构图后挂在 model 的明确命名属性上，但必须保存到 resolved config，且不能依赖模块级全局变量。

`CoffeeLoss.loss()` 保持原生返回 `(loss * batch_size, loss_detach)` 的形状和顺序。内部在 `reg_max=1` 时计算 `WIoU + TailBCE + PairMargin + normL1`；通用合成测试设置 `reg_max>1` 时，第三框项委托原生 DFL。不能因为原生 `loss[2]` 变量名仍含 `dfl`，就把 `reg_max=1` 的第三项错误记录为 DFL。

同一个 factory 会建立 one-to-many (`tal_topk=10`) 与 one-to-one (`tal_topk=7,tal_topk2=1`) 两个 CoffeeLoss 实例；两者使用相同 profile/config，但保留独立 assigner 和诊断计数。测试必须为两条分支构造非零 target，确认 TailBCE/PairMargin 非零、梯度有限，并确认顶层加权仍使用原生 `o2m/o2o` 更新逻辑。

### 27.2 独立 ramp 和动态 TailBCE

The three schedules are independent functions:

```python
wiou_ramp(progress, start=0.10, end=0.60)
tail_ramp(progress, start=0.10, end=0.60)
pair_ramp(progress, start=0.10, end=0.60)
```

Disabling one component must not change the other two schedules. TailBCE counts come from current `data_lock/class_counts.json` unless profile explicitly supplies counts and passes equality validation. Compute:

```text
w_c = clip((n_max / n_c) ** exponent, 1.0, cap)
w_c(progress) = 1 + tail_ramp(progress) * (w_c - 1)
```

不得再做算术均值归一化或最大值归一化。正确语义是：实例数最多的类别权重恰好为 `1.0`，其他类别权重在 `[1.0, cap]` 内只上调、不下调；这样才不会重复现有 ProgLoss 的“尾类看似加权、头类和部分尾类实际被压低”问题。若启用 TailBCE 的某个声明类别计数为 0，应要求修复数据/profile 或显式关闭该类加权，禁止在除零后静默设权重。额外权重只作用于使用现有质量目标 `q` 的正 soft targets；不要把每类权重全局乘到海量背景负样本。权重写入长表 `class_weight_monitor.csv`。

### 27.3 质量门控 PairMargin

PairMargin is valid only when all of the following hold: profile enables it, pair list is nonempty, the anchor is foreground, `q >= quality_threshold`, and the true/rival IDs are valid. For each configured pair `(y,c)` use the signed margin described in Section 13.5 and softplus beta from config. A nonconfigured pair contributes exactly zero. The implementation must reject duplicate, self-pairs, out-of-range IDs, and unknown names.

Required gradient test for a violating `(true=BLS_AB, rival=ANT_AB)` example: `dL/dz_true < 0` and `dL/dz_rival > 0`. Once the margin is satisfied, the pair term should be numerically near zero. These assertions are generic over integer IDs; the names in this paragraph are only the coffee profile fixture.

### 27.4 BgMix 接入位置和适用性保护

`BgMix` is a training-only transform inserted after all geometric transforms and box-coordinate updates, before image normalization and model forward. It must never run in validation, test, inference, or export paths.

Exact behavior:

1. 对每张图取全部 GT 框的并集。每个框沿 x 方向向两侧扩张 `round(box_expand * W)`，沿 y 方向向两侧扩张 `round(box_expand * H)`，裁剪到图像边界，再只对 preserve mask 边界做羽化；不能改成按短边统一扩张。
2. With probability `p`, apply one allowed external transform (blur, mild saturation, or value/contrast change) only outside the preserve mask. No external image paste is allowed.
3. If the image has no GT boxes, either skip and record `no_box_skip` or fail according to config; the default is skip. Never transform the whole image while pretending a preserve mask exists.
4. Assert that every pixel inside the binary preserve mask is bitwise unchanged before normalization. Boxes and class IDs are unchanged.
5. Record `applied`, `skipped_no_boxes`, transform type and preserve-area ratio. A unit test must test one box, multiple overlapping boxes, boundary clipping and an empty-label image.

## 28. 运行输出、错误和可复现性

### 28.1 必需运行目录

Each run writes under `<project>/<name>/` without overwriting an existing run unless an explicit resume path is supplied:

```text
resolved_config.yaml
args.yaml
profile_snapshot.yaml
dataset_yaml_copy.yaml
source_manifest.json
model.yaml
model_summary.txt
param_budget.json
pretrain_transfer.json
data_lock_id.txt
results.csv
per_class_metrics.csv
confusion_matrix.csv
confusion_pairs.csv
class_false_positives.csv
module_monitor.csv
class_weight_monitor.csv
pair_monitor.csv
loss_monitor.csv
latency.json
weights/best.pt
weights/last.pt
```

`run_manifest.json` 必须记录命令行、解析后的路径、源码包 hash、Python/PyTorch/CUDA 版本、seed、device、输入尺寸、可用时的 Git 状态（否则写 `not-a-git-repo`）以及开始/结束时间。训练产物不能反向复制到源码包。

最小 `run_manifest.json` schema：

```json
{
  "schema_version": 1,
  "run_id": "stable-unique-id",
  "status": "dry_run_passed|running|completed|failed",
  "command": "...",
  "argv": [],
  "variant": "core",
  "profile": "generic",
  "data_yaml": "resolved-absolute-path",
  "data_lock_id": "sha256",
  "resolved_config_sha256": "sha256",
  "source_manifest_sha256": "sha256",
  "nc": 1,
  "names": ["class0"],
  "imgsz": [640, 640],
  "seed": 0,
  "device": "cpu",
  "environment": {"python": "...", "torch": "...", "cuda": "..."},
  "imports": {"ultralytics": "...", "tasks": "...", "block": "...", "coffee_loss": "..."},
  "started_at": "ISO-8601",
  "ended_at": "ISO-8601-or-null",
  "failure": null
}
```

最小 `param_budget.json` schema：

```json
{
  "schema_version": 1,
  "nc": 1,
  "scale": "n",
  "variant": "core",
  "model_params": 0,
  "native_n_params": 0,
  "native_s_params": 0,
  "hard_limit": 0,
  "headroom_params": 0,
  "within_hard_limit": true,
  "recommended_limit": 0,
  "within_recommended_limit": true,
  "gflops": null,
  "input_shape": [1, 3, 640, 640]
}
```

`data_lock_id` 建议定义为 SHA256：对规范化 dataset YAML 内容、排序后的 split 相对文件列表和排序后的 label SHA256 记录做确定性序列化后计算。不得把绝对机器路径本身作为唯一 hash 输入，否则同一数据搬到服务器后无法识别为同一快照。

### 28.2 显式失败规则

The run must stop before training for: missing `--data`; invalid YAML or labels; profile/name mismatch; invalid pair; unknown config key; import isolation failure; missing requested custom module; Detect not exactly P3/P4/P5; nonfinite dry-run output; parameter limit violation; pretrained coverage below threshold; or requested `--bgmix`/PairMargin silently unavailable. Error messages must include the offending path, field, value and expected constraint.

It is acceptable to warn (and continue) for a high worker count, unavailable optional GPU telemetry, or an absent test split when test evaluation was not requested. Warnings must not be used to hide a requested feature being disabled.

### 28.3 Dry-run 不产生在线副作用

`--dry-run` must not download weights, datasets or packages. If the requested `--weights` file is absent, report the exact path and exit nonzero. This prevents an AMP check or auto-download from masking a local implementation error.

## 29. 验证矩阵

The implementation is not code-generation-complete until the following checks pass. Tests should run on CPU unless a GPU is explicitly available.

| Area | Cases | Required assertion |
|---|---|---|
| Dataset parser | synthetic `nc=1,6,80`; list/dict names; empty/invalid labels | correct `nc/names`, explicit failures |
| Genericity scan | generic package excluding coffee profile | no `coffee_self_sum`, coffee names, `nc=6`, fixed 960 or fixed ANT pair in code |
| Model variants | native,b0,r3,core,p4mid,p5lk13,full | build succeeds and variant graph differs only as specified |
| Input sizes | 640 and 960, rectangular if supported | Detect strides `[8,16,32]`, no fixed spatial assertion |
| Forward | CPU batch 1, random finite tensor | finite output, no NaN/Inf, three Detect branches |
| Gates | every custom module | initial values match Section 11.3; residual off is near native |
| Parameters | same `nc` custom vs native n/s | hard limit computed dynamically; report headroom |
| Weight migration | same-shape native layers and changed head | exact copies, explicit expected mismatches, coverage report |
| Conv choices | native/LDSConv/SCDown/MixDown controls | only requested branch changes; latency measured when available |
| Loss math | `reg_max=1` and synthetic `reg_max>1` | normL1 vs DFL routing is correct |
| Loss ramps | disable each of WIoU/TailBCE/PairMargin | remaining ramps and values do not change |
| E2E loss | one-to-many and one-to-one | both branches receive CoffeeLoss and backprop finite |
| TailBCE | counts from data, cap 1.8/2.4 | exact expected weights, no mean normalization |
| PairMargin | valid/invalid/quality-low/satisfied pairs | zero gating, gradient direction, validation failures |
| BgMix | one/multiple/no GT boxes | preserve mask bitwise, only train path changes pixels |
| Import isolation | package launch from repo root and another cwd | resolved imports all inside local package |
| Dry-run | each variant with `--dry-run` | artifacts emitted, no training/download, exit 0 only on all gates |

Static hardcode scanning may ignore `docs/` and `configs/profiles/coffee_ant6.yaml`, but must scan `train.py`, `loss/`, `augmentations/`, `tests/`, generic profile and generated model configs. A string found in an error message is allowed only if it is not used as a default path, class ID or model dimension.

## 30. Dry-run 验收门和完成定义

### 30.1 必需命令顺序

From `E:\ultralytics-8.4.43`, after the package is generated:

```powershell
python coffee26n_experiment/verify_structure.py --data <dataset.yaml> --variant native --profile generic --imgsz 640 --weights none
python coffee26n_experiment/verify_structure.py --data <dataset.yaml> --variant core --profile generic --imgsz 640 --weights yolo26n.pt
python coffee26n_experiment/train_coffee26n.py --data <dataset.yaml> --variant core --profile generic --loss native --bgmix off --imgsz 640 --seed 0 --dry-run
python -m pytest coffee26n_experiment/tests coffee26n_experiment/loss coffee26n_experiment/augmentations -q
```

The implementer must replace `<dataset.yaml>` with the caller's path. For the coffee evidence profile, run the same commands with `--profile coffee_ant6` and the profile's dataset YAML. If a dependency or weight is absent, report the exact blocked check; do not claim the dry-run passed.

### 30.2 完成定义

The implementation task is complete only when:

1. The self-contained package exists with the tree in Section 10/28 and a valid source manifest.
2. Generic and coffee profiles parse; generic contains no coffee assumptions.
3. All five custom modules construct, export and parse with dynamic channels.
4. Native control and every requested variant pass shape, stride, finite-forward, gate and parameter checks.
5. Pretrained transfer produces a detailed report; no silent `strict=False` success message is used.
6. CoffeeLoss unit/integration tests cover both end-to-end branches, `reg_max=1`, independent ramps, TailBCE and PairMargin.
7. BgMix tests prove the preserve mask and training-only insertion.
8. The dry-run creates resolved config and verification artifacts without network access or full training.
9. Final report lists changed files, commands run, pass/fail results, remaining blockers and exact next experiment IDs.

### 30.3 明确非目标

- Do not run the full 300-epoch experiment as part of code generation.
- Do not claim an accuracy improvement before controlled B0/B1 and the staged ablations are trained.
- Do not modify root `ultralytics/`, delete unrelated user files, or package the dataset into the experiment folder.
- Do not add P2 Detect, reverse spatial feedback, global activation replacement, dense full-channel large kernels, or unvalidated class penalties merely to make the graph look more complex.

## 31. 代码生成交接

The standalone prompt in `docs/coffee26n_codegen_multiagent_prompt.md` is the recommended execution instruction. It repeats the boundaries in this document and assigns read-only exploration, single-writer implementation and independent review to separate agents. The main agent must wait for every requested subagent, apply only reviewed changes, run the dry-run gate, and report file/symbol-level evidence. The prompt is intentionally dataset-agnostic; the caller supplies `--data` and selects `generic` or an explicitly validated profile.
