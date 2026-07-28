# Coffee26n 通用代码生成多智能体提示词

下面整段可直接作为一个新的 Codex 主任务提示词使用。它要求主智能体实际创建并等待子智能体，不是让主智能体在文字里模拟分工。

---

你是本任务的主智能体和唯一最终负责人。请在当前机器上实际完成 Coffee26n 自包含实验包的代码生成、验证和复核，不要只给方案或伪代码。

## 一、固定输入

- 工作目录：`E:\ultralytics-8.4.43`
- 唯一设计规格：`E:\ultralytics-8.4.43\docs\coffee26n_optimization_plan.md`
- 目标实验包：`E:\ultralytics-8.4.43\coffee26n_experiment`
- 多智能体提示词来源：`E:\ultralytics-8.4.43\docs\coffee26n_codegen_multiagent_prompt.md`
- 原生 YOLO26 模型：`E:\ultralytics-8.4.43\ultralytics\cfg\models\26\yolo26.yaml`
- 原生 loss：`E:\ultralytics-8.4.43\ultralytics\utils\loss.py`
- 可复用模块实现：`E:\ultralytics-8.4.43\edgelite_experiment\local_ultralytics\ultralytics\nn\modules\block.py`
- 可参考 parser：`E:\ultralytics-8.4.43\edgelite_experiment\local_ultralytics\ultralytics\nn\tasks.py`
- 可参考实验 loss：`E:\ultralytics-8.4.43\wiou_progloss_experiment\wiou_progloss_loss.py`
- 可参考 loss helpers：`E:\ultralytics-8.4.43\wiou_progloss_experiment\loss_extensions.py`
- 本地预训练权重候选：`E:\ultralytics-8.4.43\yolo26n.pt`

数据输入不是代码默认值。实现的生产 CLI 必须要求调用方显式传入 `--data`。为了实现阶段的自动化测试，可以在 `coffee26n_experiment/tests/fixtures/` 生成最小合成 YOLO detect 数据集；只有在调用者明确给出真实 dataset YAML 时才对其执行 dry-run。不得把 `coffee_self_sum`、`nc=6`、咖啡类别名、ANT 类别 ID 或 `imgsz=960` 写入通用代码默认值。

## 二、任务目标

根据完整设计规格生成可移动、可 dry-run、可测试的自包含实验包。必须实现：

1. 原生 YOLO26n control 和 `r3/core/p4mid/p5lk13/full` 结构变体。
2. `ResBridge`、`DetailBlock`、`MixDown`、`MidBlock`、`ContextBlock`，保持规格中的排序、卷积类型、激活函数、近恒等门控和 P3/P4/P5 Detect。
3. 数据集无关的配置/profile 系统，优先级为 `CLI > profile > generic defaults > dataset facts`。
4. 动态解析 `nc/names`、标签统计和 data lock；generic profile 无咖啡路径或类别。
5. 只在实验包副本中修改 Ultralytics；根目录 `ultralytics/` 保持不变。
6. 精确、可审计的预训练权重迁移和 `pretrain_transfer.json`。
7. CoffeeLoss：WIoU、TailBCE、PairMargin、`reg_max=1` 的 normL1，并同时接入 YOLO26 end-to-end one-to-many/one-to-one 两条分支。
8. 训练期 BgMix，严格只改 GT 框联合 preserve mask 外的像素。
9. `verify_structure.py`、统一训练入口、完整测试、source manifest、README 和 Slurm 入口。
10. 不跑完整训练，不声称精度已经提高；只完成代码、dry-run 和最小验证。

## 三、硬边界

1. 不修改、覆盖或删除 `E:\ultralytics-8.4.43\ultralytics\` 下任何文件。
2. 不修改无关实验目录，不清理用户文件，不重置工作区。
3. 不加入 P2 Detect、P5->P3 反向空间反馈、SimAM、全网激活替换、全网 depthwise 下采样或密集全通道大核。
4. Detect 必须是 P3/P4/P5，实际 stride 必须是 `[8,16,32]`。
5. 自定义模型参数必须小于以相同运行时 `nc` 构建的原生 YOLO26s；上限动态计算，不能写死 `9,952,508`。
6. generic profile 默认 `PairMargin=false`、`pairs=[]`、`BgMix=false`、TailBCE 可关闭；请求功能但配置无效时失败，禁止静默关闭。
7. 不以 `strict=False` 加一条“加载成功”日志代替权重覆盖率和映射报告。
8. dry-run 禁止在线下载权重、数据或依赖。
9. 多智能体阶段禁止多个智能体同时写同一工作树。所有探索和复核子智能体只读；只有主智能体是写入者。
10. 当前目录不是 Git 仓库，必须用 SHA256 `source_manifest.json` 和明确路径证明来源与未改动边界。

## 四、执行协议

### Phase 0：主智能体预检

在创建子智能体前，主智能体必须：

1. 完整阅读 `coffee26n_optimization_plan.md`，不能只读标题或结论。
2. 检查目标目录是否已存在；若存在，先阅读并与规格对比，不得直接覆盖未知用户改动。
3. 记录根目录 `ultralytics/nn/tasks.py`、`ultralytics/nn/modules/block.py`、`ultralytics/utils/loss.py` 和原生模型 YAML 的 SHA256，作为结束时未改动证明。
4. 列出将创建的目标文件和验证步骤，建立任务计划。
5. 当前阶段只读，不编辑文件。

### Phase 1：并行只读探索

主智能体必须使用真实的子智能体/协作工具并行创建以下三个子智能体。若当前环境提供 `spawn_agent`、`wait_agent` 和 `list_agents`，应分别调用三次 `spawn_agent(task_name=...)`，再用 `wait_agent`/`list_agents` 等待状态全部为 completed；不要只把这些名字写进计划。主智能体加三个子智能体正好占用四个并发槽，不得再创建第四个并行探索智能体。子智能体不得编辑、创建或删除文件。主智能体必须等待三个都返回后，整理统一实现决定；不能收到第一个结果就开始写。

#### 子智能体 1：`architecture_explorer`

发送以下任务：

> 只读检查 `E:\ultralytics-8.4.43` 中原生 YOLO26、edgelite 本地副本、模块导出、`parse_model`、Detect、scales 和权重结构。重点核对 Coffee26n 规格中的五个模块、YAML 节点顺序、动态通道、P3/P4/P5 stride、变体节点索引、预训练映射和参数预算。禁止编辑文件。输出必须包含：读取的文件；关键类/函数和行号；每个 variant 的建议节点/Detect 来源；`parse_model` 所需分支；权重迁移风险；与规格冲突的事实；建议测试。所有结论给文件和 symbol 证据。

#### 子智能体 2：`loss_explorer`

发送以下任务：

> 只读检查 `ultralytics/utils/loss.py`、`wiou_progloss_experiment/wiou_progloss_loss.py`、`loss_extensions.py` 以及 YOLO26 end-to-end loss 调用链。确认 `reg_max=1` 时第三框项的真实含义、one-to-many/one-to-one 接入点、WIoU ramp、当前 ProgLoss 权重归一化、TailBCE 正负样本语义、PairMargin 可获得的 logits/target_scores/fg_mask 和 trainer epoch 更新方式。禁止编辑文件。输出必须包含：调用链；精确签名；两分支挂接方案；独立 ramp 方案；数值/梯度风险；必须测试的断言；每项的文件/symbol 证据。

#### 子智能体 3：`test_explorer`

发送以下任务：

> 只读检查当前仓库测试方式、Python/依赖可用性、模型构图入口、已有 dry-run/自包含实验模式。设计最小但完整的测试矩阵，覆盖 `nc=1,6,80`、`imgsz=640,960`、全部 variant、CPU forward、AMP/梯度、Detect stride、动态 YOLO26s 参数上限、import isolation、source manifest、权重迁移、两条 E2E loss、BgMix preserve mask 和通用性静态扫描。禁止编辑文件。输出必须包含：可执行命令；必要 fixtures；可能缺失依赖；最快验收路径；每项的文件/symbol 证据。

每个探索子智能体的统一返回格式：

```text
STATUS: complete | blocked
FILES_READ:
KEY_FACTS:
SPEC_CONFLICTS:
IMPLEMENTATION_RECOMMENDATIONS:
TESTS_REQUIRED:
BLOCKERS:
```

主智能体等待全部三个结果后，必须在 commentary 中简要说明吸收了哪些事实、有哪些规格需要按源码适配；只有此时才能开始编辑。

### Phase 2：唯一写入者实现

只有主智能体可以编辑。使用补丁式编辑，保留目标目录里任何已有且不冲突的用户内容。按以下顺序实现，每完成一项就运行最小相关检查：

1. 建立 `coffee26n_experiment/` 目录、README、配置目录、tests、loss、augmentations 和脚本入口。
2. 复制最小但完整的 package-local Ultralytics 运行时；创建并验证 `source_manifest.json`。
3. 实现五个模块，导出模块并修改 package-local parser。构造签名、门控、卷积和激活严格服从规格第 4--8、25 节。
4. 生成 `native/core/p4mid/p5lk13/full` 模型 YAML；如 `r3` 需要独立 YAML 一并生成。所有 YAML 的 `nc` 在运行时从 dataset 注入。
5. 实现 generic 与 coffee profile schema、配置合并、unknown-key 检查、dataset YAML/label 验证、动态 class counts 和 data lock。
6. 实现权重迁移：按 variant 的语义节点映射，迁移 MixDown.base 等原生分支，生成完整 `pretrain_transfer.json`。
7. 实现 CoffeeLoss。按规格第 27.1 节让 `CoffeeLoss` 继承当前 `v8DetectionLoss`，通过 factory 交给 `CoffeeE2ELoss`，覆盖当前实际调用的 `.loss()`/assignment 路径，并在顶层无参数 `.update()` 中同步两分支 progress；package-local `DetectionModel.init_criterion()` 只在函数内延迟导入实验包 `loss.coffee_loss`，避免循环导入，并核对模块 `__file__`。不能只实现一个没有被 E2ELoss 调用的独立 `__call__()`。必须保留原生 trainer 返回接口，同时在 one-to-many/one-to-one 两分支使用同一配置；拆开三个 ramp；正确处理 `reg_max=1`。
8. 实现 BgMix，并接入训练增强正确位置。验证 train-only、no-box guard 和 preserve mask 逐像素不变。
9. 实现 `verify_structure.py` 和 `train_coffee26n.py` 的完整 CLI；保存 `resolved_config.yaml` 和 run manifest。
10. 编写 pytest/最小集成测试、README、验证说明和 Slurm 入口。Slurm 脚本必须要求或接收 dataset YAML，不能隐藏咖啡默认路径。

实现期间的强制行为：

- 每次编辑前说明正在修改哪些文件和原因。
- 不用根目录模块做 monkey patch；入口启动后验证实际 import 路径位于 `coffee26n_experiment/local_ultralytics/`。
- 遇到规格与当前源码签名不一致时，以保持规格行为和当前原生调用接口为准，并在 README/最终报告记录适配点。
- 请求的功能无法接入时必须暴露错误并修复，不得自动退回 native loss 或关闭增强。
- 不启动 300 epoch 或任何长训练。

### Phase 3：并行独立复核

主实现和首轮测试完成后，再真实创建并等待以下三个只读复核子智能体。确认 Phase 1 的三个探索智能体已经结束、并发槽已释放，再调用三次 `spawn_agent(task_name=...)` 创建 reviewer，并用 `wait_agent`/`list_agents` 等待全部完成。复核智能体可以运行只读测试和检查命令，但不得编辑文件；运行 pytest 时设置 `PYTHONDONTWRITEBYTECODE=1` 并禁用 pytest cache，临时输出放到系统临时目录，避免把测试缓存写入源码包。主智能体必须等待全部返回并逐项处置 findings。

#### 子智能体 4：`correctness_reviewer`

发送以下任务：

> 对 `E:\ultralytics-8.4.43\coffee26n_experiment` 做只读 correctness review。重点检查 parser 通道推导、variant YAML 索引、ResBridge 输入顺序、gate/activation、Detect stride、YOLO26 end-to-end 两分支、`reg_max=1`、TailBCE/PairMargin 梯度、BgMix 插入点、预训练语义映射和 coverage。运行最小相关测试但不编辑。按严重度输出 findings，每项必须有绝对文件路径、行号/symbol、影响、复现方式和修复建议；无问题也要明确说明残余风险。

#### 子智能体 5：`generality_reviewer`

发送以下任务：

> 对 `E:\ultralytics-8.4.43\coffee26n_experiment` 做只读 dataset-independence review。扫描通用代码中的 `coffee_self_sum`、`nc=6`、咖啡类名、ANT 固定 ID、固定 `960/120/60/30`、固定 `9,952,508` 和默认 PairMargin/BgMix。检查 `nc=1,6,80`、names list/dict、CLI/profile 优先级、unknown key、动态 YOLO26s 参数上限、通用长表输出和 Slurm 参数。coffee 专用常数只允许出现在 `configs/profiles/coffee_ant6.yaml`、文档和专用 fixture。按严重度输出带路径/行号的 findings，不编辑。

#### 子智能体 6：`test_reviewer`

发送以下任务：

> 只读运行或审查最小验收测试，覆盖全部 model variants、`nc=1,6,80`、`imgsz=640,960`、CPU forward finite、Detect `[8,16,32]`、gate 初始化、动态参数门槛、pretrain report、CoffeeLoss 两分支、独立 ramps、PairMargin 梯度、BgMix preserve mask、import isolation、manifest hashes 和 dry-run 无网络。记录逐条 pass/fail、命令、耗时和缺失依赖；不编辑文件，不跑完整训练。

复核返回格式：

```text
STATUS: complete | blocked
FINDINGS:
TEST_COMMANDS:
PASS_FAIL_MATRIX:
RESIDUAL_RISKS:
```

主智能体收到所有复核后：

1. 先按 P0/P1/P2 严重度汇总 findings。
2. 自己修复所有 P0/P1 和任务范围内的 P2；不要让多个 reviewer 写入。
3. 对修复涉及的区域重新运行最小测试和完整 dry-run gate。
4. 若某 finding 不采纳，必须用源码/测试证据解释，不能只说“认为没问题”。

## 五、强制测试和验收

优先使用 package-local test fixtures。命令可按当前 Python 入口适配，但必须覆盖等价行为并报告实际命令。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider coffee26n_experiment/tests coffee26n_experiment/loss coffee26n_experiment/augmentations -q
python coffee26n_experiment/verify_structure.py --matrix --profile generic --weights none
python coffee26n_experiment/verify_structure.py --data coffee26n_experiment/tests/fixtures/nc1/data.yaml --variant native --profile generic --imgsz 640 --weights none
python coffee26n_experiment/verify_structure.py --data coffee26n_experiment/tests/fixtures/nc6/data.yaml --variant core --profile generic --imgsz 960 --weights none
python coffee26n_experiment/verify_structure.py --data coffee26n_experiment/tests/fixtures/nc80/data.yaml --variant full --profile generic --imgsz 640 --weights none
python coffee26n_experiment/train_coffee26n.py --data coffee26n_experiment/tests/fixtures/nc6/data.yaml --variant core --profile generic --loss native --bgmix off --imgsz 640 --seed 0 --weights none --dry-run
```

如果 `--weights none` 的 native forward 接口需要适配，允许随机初始化，但参数门槛比较必须在相同 `nc` 下同时构建 n/s。若本地 `yolo26n.pt` 存在，再增加一次带权重 core dry-run；不存在则明确记录，不联网下载。

静态扫描至少检查：

```powershell
rg -n "coffee_self_sum|ANT_AB|ANT_CD|BLS_AB|BLS_CD|nc\s*[:=]\s*6|9,952,508|imgsz\s*[:=]\s*960" coffee26n_experiment -g "*.py" -g "*.yaml" -g "*.slurm"
```

对扫描结果逐条分类。`configs/profiles/coffee_ant6.yaml`、coffee 专用测试 fixture 和说明文档可以包含 profile 证据；generic profile、训练入口、loss、增强、parser、通用 tests 和模型生成逻辑不得依赖这些常数。

结束前重新计算 Phase 0 记录的根目录 SHA256。任何变化都视为阻断，除非能证明是用户在任务期间做的外部改动；不得自行回退用户改动，必须报告并停止相关覆盖操作。

## 六、验收标准

只有同时满足以下条件才能宣布代码生成完成：

1. 目标目录结构完整，source manifest 可重新验证。
2. 实际 import 全部来自 package-local Ultralytics。
3. 五个模块和全部 variant 可构图，Detect 为 P3/P4/P5，forward finite。
4. `nc=1,6,80` 和 `imgsz=640,960` 的最小矩阵通过。
5. 参数上限按相同 `nc` 动态计算，Coffee26n 严格小于 YOLO26s。
6. 预训练迁移报告包含 exact/remapped/skipped/new/coverage，native shape-compatible 层抽检逐元素一致。
7. CoffeeLoss 同时作用于 one-to-many/one-to-one；normL1、TailBCE、PairMargin、独立 ramp 的单元测试通过。
8. BgMix 的 preserve mask、train-only 和 no-box 测试通过。
9. generic profile/通用代码无咖啡硬编码；请求错误配置会非零退出。
10. dry-run 不下载、不训练，输出 resolved config、参数报告、迁移报告、model summary 和 run manifest。
11. 所有复核智能体已结束，findings 已修复或用证据说明不采纳。
12. 根 `ultralytics/` 的基线文件 hash 未被本任务改变。

## 七、最终报告格式

最终答复必须自包含，按以下顺序报告：

1. **结果**：是否完成；没有完成时准确说明阻断点。
2. **生成内容**：目标目录和关键文件，使用绝对可点击路径。
3. **结构实现**：五个模块、排序、Detect 来源、参数 n/s 比较和预训练覆盖率。
4. **通用性**：说明 `--data`、动态 `nc/names`、generic/coffee profile 边界，以及硬编码扫描结果。
5. **损失和增强**：说明 E2E 两分支、`reg_max=1`、TailBCE、PairMargin、BgMix 接入和默认关闭行为。
6. **子智能体结果**：列出六个子智能体是否完成、最重要 findings 和处置。
7. **验证表**：实际命令、pass/fail、未运行项及原因。
8. **未验证内容**：明确没有跑完整训练、不能声称准确率提升。
9. **下一步命令**：给一条 generic dry-run 和一条用户选定 profile 的真实训练命令；不要擅自启动长训练。

现在开始执行。先完成 Phase 0，然后真实创建 Phase 1 的三个只读子智能体并等待全部返回。

---

## 使用说明

这份提示词的默认目标是“生成可靠代码”，不是“立即训练某个数据集”。启动任务时可以额外在提示词末尾补充真实的 dataset YAML、profile 和权重路径；不补充时，主智能体应使用合成 fixtures 完成通用构图和 dry-run，不能把咖啡数据集偷偷变成生产默认值。
