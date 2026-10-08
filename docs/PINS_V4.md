# 引脚 V4 与评测口径说明

本轮对应两个任务：先把评分口径拆清楚，再改进引脚位置与编号/名称关联。器件检测、器件 OCR 和网络算法不是本轮配对实验的变量。最终采用的配置、完整结果与退步案例见 `reports/pins_v4_comparison.md`；本说明不预填尚未完成的最终分数。

## 模块职责

| 文件或目录 | 职责 |
|---|---|
| `evaluate_v2.py` | 保持原有严格评测。没有为了提高分数修改它；它仍是本地非官方评测器。 |
| `evaluate_diagnostic.py` | 新增身份规范化与几何诊断。给出一对一匹配表及器件、引脚、连通错误分项，不读取数据集。 |
| `pcb/pin/localization/` | 引脚定位模块：根据图片和器件框产生端点、器件边界上的基点、方向等候选；版本由配置中的 `pin_localization` 选择。 |
| `pcb/pin_detection_v4.py` | 在 V3 候选基础上，只修正有连续像素证据的短引线端点或单一 L 形转角。双端器件、晶振与 GND 保持既有几何；长导线、分叉、断点、文字遮挡与图像边界均保守回退。 |
| `pcb/pin_semantics_v4.py` | 引脚语义模块：局部距离、方向和行间距约束；编号一对一分配；区分框内信号名称和物理编号；为未识别编号保留内部标识。 |
| `pcb/pin/semantics/v4.py` | 接入模块化流程。先为 OCR 文字确定全场景唯一器件归属，再逐器件分配编号/名称，汇总诊断记录。 |
| `tools/pin_frontend_cache.py` | 建立 `0001～0150` 白名单、生成不接触答案的器件/OCR/V3 引脚快照，并保存图片、源代码、权重与缓存签名。 |
| `tools/benchmark_pins_v4.py` | 在相同前端快照上配对运行 V3 与候选引脚模块，核验器件/OCR 未变，再使用四种口径评测。 |
| `configs/current.json` | 当前分支的 V3 引脚、V3 导线与 V3 拓扑基线。 |
| `configs/examples/pins_v4.json` | 引脚 V4 候选配置；导线与拓扑保持 V3。以最终对比报告确认的配置为准。 |
| `configs/examples/pins_semantics_v4.json` | 仅替换引脚语义的分项实验配置，便于区分定位与语义改动的影响。 |

### 编号与名称如何处理

- 纯数字文字即使被前面的分类器归为 `VALUE`，也只能在符合局部引脚距离、方向和行间距约束时作为编号候选，不能在整张图上随便寻找数字。
- 芯片框内的 `A1`、`A2`、`P12` 等“字母＋数字”文字优先作为信号名称；芯片框外的 `A1` 等仍可作为 BGA 物理编号。框内纯数字仍可在局部证据成立时作为编号。
- 同一实际 OCR 文字只允许归属于一个器件。不同位置的两个 `1` 可以分别用于两个芯片；不是在整张图上禁止编号重复。
- 每个器件内，一个编号最多对应一个引脚。出现竞争时进行整体候选匹配，而不是按遍历顺序将后面的引脚直接删除。
- 未识别编号保留为内部 `UNK_...`；它表示“端点候选仍在，但不知道物理编号”，不是新识别出的编号。
- GND 的不可观察内部编号不猜测。几何诊断可以检查接地点位置与连通，但不能声称识别出了标注中的内部编号。

上述规则仍可能受到端点误检、OCR 字符识错、纯数字元件值与引脚号相邻等问题影响，不保证所有候选正确。诊断中的 `promoted_numeric_values` 表示候选被采用，不等于已与答案核对正确。

定位修正沿原始轴的固定窄走廊追踪，并验证从器件边缘起的像素连通，不允许逐步漂移到相邻线。可见短线端点只是局部坐标证据，不保证等同于标注中的物理引脚端点。`pin_localization_events` 保存原始 `v3_tip`、候选坐标、是否移动和回退原因；不会改变器件框或自动推断整条长导线的引脚边界。

## 四种评测口径不能混用

| 报告中的口径 | 使用哪些预测 | 允许怎样匹配 | 能说明什么 |
|---|---|---|---|
| `strict` | 正式输出的 `result.json` | 原 `evaluate_v2.py` 的规则；普通标识、引脚编号/名称等按原口径比较，GND 序号按原有规则处理。 | 相同严格规则下，算法改动是否改善了结果。不是官方成绩。 |
| `identity_normalized` | 同一份 `result.json` | 标识、引脚编号和名称忽略大小写；GT 明确为 `∅数字` 或 `__unresolved_...` 的器件，允许类型相同且框误差不超过 20 像素的一对一自动编号匹配。真实可见编号写错不豁免。 | 落实用户要求后的身份识别表现，不把合理的自动编号差异算成错误。 |
| `geometry_connectivity` | 同一份 `result.json`，只包含可导出的引脚 | 器件按类型＋框误差 20 像素一对一匹配；引脚按对应器件内的位置误差 5 像素一对一匹配，不要求编号/名称相同。 | 将身份识别问题暂时移开，检查已经输出的端点位置和连通关系。不能当作编号识别准确率。 |
| `retained_geometry_connectivity` | 单独的 `internal_connectivity.json`，包括未识别编号的内部候选 | 未知引脚先获得唯一临时标识，再重新构建网络，只按几何口径评测。 | 检查“保留端点但不知道编号”后可恢复多少物理连接，以及增加了多少错误连接。不是正式提交结果。 |

身份规范化还有以下边界：

- 电气 `value` 不做无差别大小写折叠，`m` 与 `M` 不能混为一谈。
- 未标注芯片若两边 `Name` 都只是各自自动编号的副本、空串或 `null`，不会仅因这两个占位名不同扣分；真实芯片型号不同仍算错误。
- 大小写碰撞、重复候选仍采用一对一匹配，不能把多个预测都匹配成同一个答案。
- `UNK_...`、`__TMP_...` 等未知标识不能在身份口径中冒充正确编号，哪怕碰巧与某个字符串相同。
- 连线几何 `NetLine` 沿用线段端点 5 像素容差。它与“电气上是否相连”不是同一个指标。

`PinPair` 检查引脚两两是否属于同一网络；`NetHypergraph` 检查网络中的引脚集合匹配；`Component` 和 `Pin` 分别检查器件与引脚。在几何口径中，器件与引脚指标刻意不考查型号、值、编号、名称，不能与严格口径直接比较并称为算法提升。

报告同时保存 `precision`、`recall`、合并计数的 `f1` 和逐例平均 `macro_f1`。汇报前后差值时必须固定口径与指标，例如“同一严格口径下的逐例平均引脚 F1 提高多少个百分点”。同一个版本从 `strict` 换到 `identity_normalized` 的变化属于评分口径变化，不属于算法改进。

错误分项允许重叠。例如类型错误也可能同时导致“没有类型＋位置匹配”。这些分项是可观测错误，不自动断言是 OCR、YOLO 或其他某个模块造成的。

## 为什么重新建立配对基线

本轮使用恢复后的 `feature/modular-baseline` 当前文件建立新基线：两边固定相同器件/OCR，固定 **V3 wire 与 V3 topology**，只允许引脚定位和语义版本不同。

此前未提交的 nets V4 文件和对应运行证据不在本轮恢复的工作目录中，因此不能把本轮分数直接与旧 nets V4 历史分数相减。完整报告只报告当前这次同条件配对结果。

运行器会检查源代码签名、缓存内容校验和、图片签名和配置，并检查候选运行没有修改器件框、类型、值、名称与 OCR 文本。重跑的 V3 输出还必须与准备阶段保存的 V3 输出完全一致。核验失败会报错，不会静默继续并声称实验可比。

## 数据隔离

唯一允许的开发数据目录为：

```text
C:\Users\pps\Desktop\EDA0.5\赛题六公开数据集\赛题六公开数据集\200_train_cases
```

仅使用自然编号 `0001～0150`。`development_manifest` 逐个检查这些固定目录，不遍历保留案例；每例必须恰有一张 PNG 和一个 `*_target*.json`，重定向到白名单外的目录或文件、缺失与歧义均停止运行，不自动扩大范围。`--case-ids` 也只接受白名单中的不重复编号。

`0151～0200` 与整个 `10_GTcase` 不用于本轮。未得到用户明确验证命令，不得读取、抽样或统计它们。即便后来获准验证，也不能据此继续调参，除非用户另外授权。

准备阶段只用图片做推理，不读取答案内容。配对阶段先完成全部图片推理，再读取允许范围内的答案评分。实际使用的编号写入日志和报告。这里是开发集配对评测，不代表独立保留集泛化成绩。

## PowerShell 复现

下面命令在同一个 PowerShell 窗口顺序执行。使用已有 Python 环境，不安装或下载新模型；代码与输出均位于 `E:\EDA\EDA`。每次使用新的带时间戳目录，避免覆盖原有证据。

### 1. 准备前 150 例冻结快照及 V3 基线

```powershell
Set-Location -LiteralPath 'E:\EDA\EDA'
$pinPython = 'C:\Users\pps\Desktop\EDA0.5\.venv\Scripts\python.exe'
$pinDataRoot = 'C:\Users\pps\Desktop\EDA0.5\赛题六公开数据集\赛题六公开数据集\200_train_cases'
$env:PYTHONPATH = 'E:\EDA\EDA\.runtime\py312'
$env:PYTHONIOENCODING = 'utf-8'
$pinRunTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$pinPrepOutput = "runs/pins_prepare_$pinRunTag"
$pinPairOutput = "runs/pins_compare_$pinRunTag"

& $pinPython -B tools/benchmark_pins_v4.py --data-root $pinDataRoot --prepare-only --output-dir $pinPrepOutput --device 0 --ocr-device cuda
if ($LASTEXITCODE -ne 0) { throw '前端准备失败，请先查看输出目录中的 progress.json。' }
```

准备成功后生成 `$pinPrepOutput/report.json`。正常完整对比要求该参考报告包含 150 例成功记录；不应把少量校准案例的准备报告当作完整参考报告。

### 2. 使用该报告进行 V3/V4 配对评测

```powershell
$pinReference = Join-Path $pinPrepOutput 'report.json'
& $pinPython -B tools/benchmark_pins_v4.py --data-root $pinDataRoot --reference-report $pinReference --candidate-config configs/examples/pins_v4.json --output-dir $pinPairOutput --device 0 --ocr-device cuda
if ($LASTEXITCODE -ne 0) { throw '配对评测存在失败案例，请检查 report.json，不能只汇报成功案例。' }
```

如只做语义分项实验，将候选配置改为 `configs/examples/pins_semantics_v4.json`，并另设一个全新输出目录；不能覆盖完整候选实验。已完成的准备报告可复用，但若冻结前端或网络源代码发生变化，签名校验会拒绝复用，需要重新准备。

运行器拒绝使用已有非空输出目录。中断后的目录留作证据，再次运行请更换目录名；不要为了重跑直接清空旧结果。

### 3. 只看 0001 的新版实际输出

这条命令只读 `0001` 图片，不计算正确率，也不读取答案。它使用模块化入口；仅传 `--pipeline v4_component` 而不指定模块化配置，并不等于启用了引脚 V4。

```powershell
$pinSingleOutput = "runs/pins_single_0001_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $pinPython -B main.py --input_dir $pinDataRoot --case_start 1 --case_end 1 --orchestrator modular --pipeline_config configs/examples/pins_v4.json --output_dir $pinSingleOutput --device 0 --ocr_device cuda --save_debug_images
```

输出位于 `$pinSingleOutput/0001/`。这里的 `result.json` 是可导出结果，不应把配对实验中的 `internal_connectivity.json` 替换进去作为正式结果。

### 4. 运行不依赖真实数据集的回归测试

```powershell
& $pinPython -B -m unittest discover -s tests -p test_evaluate_diagnostic.py -v
& $pinPython -B -m unittest discover -s tests -p test_pin_semantics_v4.py -v
& $pinPython -B -m unittest discover -s tests -p test_pin_detection_v4.py -v
& $pinPython -B -m unittest discover -s tests -p test_benchmark_pins_v4.py -v
& $pinPython -B -m unittest discover -s tests -v
```

## 输出文件怎么看

- `manifest.json`：本次请求与实际案例编号、配置、源文件签名、数据隔离声明。
- `progress.json`：逐例完成状态，用于查看运行进度和失败原因。
- `report.json`：前后两组四种口径的汇总、逐例结果、以百分点计的差值、匹配表及错误分项。
- `baseline/案例号/result.json`、`candidate/案例号/result.json`：同条件下的正式预测结构。
- 每例 `scene.json`：包含内部引脚候选的场景记录，便于检查未导出候选。
- 每例 `diagnostics.json`：引脚候选、编号与名称关联、OCR 文字归属及重分类事件等诊断。
- 每例 `internal_connectivity.json`：为几何诊断单独重建的网络，可能含 `__TMP_...`，不能视为识别成功的编号。
- 每例 `pin_audit.json`：标明内部编号原值、临时值及 `recognized_number`，避免把“保留了候选”误说成“识别正确”。

内部场景坐标以原图左上角为原点；提交格式的坐标通过原有导出层统一转换，不能直接把两种坐标数值混比。此次引脚语义修改不会自行翻转坐标。
