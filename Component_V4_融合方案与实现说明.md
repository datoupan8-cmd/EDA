# Component V4.1 融合方案与实现说明

## 一、为什么融合

原 V3/V4 已具备官方 schema、OCR 文字角色、Pin/Wire/Topology、诊断图和严格 JSON 校验，但旧 YOLO 权重在元件本体检测上仍有明显漏检与误报。同组 V2 提供继续训练后的 YOLO11n、1280 像素切片推理、坐标恢复、NMS，以及 EasyOCR 分块思路。V4.1 将同组方案中可泛化的视觉能力接入原有工程，同时拒绝自动编造位号、0057 单案例模板和宽松主评测。

两套方案不适合整体二选一。同组方案更强的是 `bbox + type`；原工程更强的是官方语义、全局文字关联、不可观测字段处理和完整下游。因此最终边界是：YOLO 只回答“元件在哪里、是什么类型”，OCR 与关联模块回答“位号、Name、Value 是什么”。

## 二、两套原始架构

### 项目 A：原 V4

```text
Image → tiled YOLO(old weight) → bbox/type
      → RapidOCR full image → text roles
      → proposal fusion + global designator/value assignment
      → V3 Pin → Wire → Topology → validated JSON
```

优势是结构完整、可回滚、每个 Component 有来源诊断；问题是旧权重 detection recall 较低，单次全图 OCR 对小字和局部文字召回不足。

### 项目 B：同组 V2

```text
Large image → overlapping tiles → continued YOLO → global bbox/type → NMS
Tiles → EasyOCR
Detector + OCR → component semantics
```

优势是继续训练后的 detector 和分块处理；风险是自动生成 U1/Q1、0057 专用 MOSFET 模板、大小写宽松 evaluator，以及部分流程无法区分可观察位号与内部 ID。

## 三、融合选择

| 模块 | 来源 | V4.1 处理 | 结论 |
|---|---|---|---|
| 继续训练 YOLO 权重 | 同组 V2 | 设为默认，旧权重保留回滚 | 直接采用 |
| 1280/192 tiled YOLO | 两边已有 | 统一 adapter、全图坐标、class-aware NMS | 保留 |
| RapidOCR 全图 | 原工程 | 使用隔离的 1.4.4 运行时 | 保留 |
| EasyOCR 每块 | 同组思路 | 640/96 分块，每块识别并恢复全图坐标 | 融合 |
| OCR token 合并 | 新增 | 按文本、IoU、置信度去重 | 融合 |
| 选择性局部二次 OCR | 用户建议 | 扩框、放大、CLAHE、锐化，只处理缺失/低置信元件 | 保留可选，非默认 |
| 位号全局匹配 | 原 V4 | type-prefix、距离、方向、置信度与一对一约束 | 保留并扩展 |
| Value 兼容 | 两边 | 扩展官方前缀和物理量类型，仍独立分配 | 保留 |
| Name association | 原 V4 | 代码保留，BEST 关闭 | 当前 A/B 无稳定收益 |
| Geometry fallback | 原 V4 | 代码保留，BEST 关闭 | 原组合误报过多 |
| 自动补 U1/Q1 | 同组 V2 | 不接入 | 无视觉证据会伪造 key |
| 0057 MOSFET 模板 | 同组 V2 | 不接入 | 单案例过拟合风险 |
| Pin/Wire/Topology | 原 V3 | 本轮冻结 | 便于判断 Component 前端贡献 |

## 四、最终 V4.1 数据流

```text
                         ┌─ 1280 tiled YOLO → bbox/type/confidence ─┐
PNG ─────────────────────┤                                           ├─ proposals/NMS
                         ├─ RapidOCR 1.4.4 full image ───────────────┤
                         └─ EasyOCR every 640 tile ──────────────────┘
                                                ↓
                                 global coordinates + token dedup
                                                ↓
                                token role classification
                 DESIGNATOR / VALUE / PIN_NUMBER / PIN_NAME / MODEL / OTHER
                                                ↓
                         constrained global designator/value assignment
                                                ↓
                       Component(key, Name, type, value, bbox, provenance)
                                                ↓
                          frozen Pin → Wire → Topology → strict validator
```

`selective_local` 是另一条可切换路径：RapidOCR 初识别和初步关联后，只对缺位号、缺 Value、box 缺型号或关联分数低的元件扩框，进行放大增强和 EasyOCR 二次识别，再重做关联。

## 五、主要代码

- `pcb/component_detector_yolo.py`：权重加载、切片、全图坐标恢复、NMS、包含权重 SHA256 的缓存键。
- `pcb/ocr_backends.py`：全分块 EasyOCR、Hybrid token 合并、选择性局部二次 OCR 和缓存。
- `pcb/text_detection.py`：文字角色、官方位号前缀与 Value 物理量规则。
- `pcb/component_text_assignment.py`：位号、Name、Value 的候选评分和全局一对一分配。
- `pcb/component_detection_v4.py`：B～F/BEST 阶段和 V4/V4.1 文字规则开关。
- `pcb/vision_v4.py`：Component V4.1 接入冻结的 V3 下游，并选择局部二次 OCR 区域。
- `pcb/data_policy.py`：硬限制开发范围为 0001～0150。
- `evaluate_component_v4.py`：Component 专项指标和下游回归指标。
- `run_ocr_ablation.py`、`run_component_ablation.py`：可重复的消融入口。
- `tools/prepare_component_dataset.py`：直接读取最新版官方 target，完成坐标转换和 120/30 划分。
- `tools/train_component_detector.py`：确定性训练、early stopping、multi-scale 与类别顺序校验；本轮没有重新训练。

## 六、关键算法说明

### Tiled YOLO 与 NMS

大图按 1280×1280、重叠 192 像素切片。每个预测框加回 tile 偏移恢复整图坐标，再对同类别高重叠框执行 NMS。缓存键使用权重完整 SHA256，避免旧权重和新权重因文件大小相近而错误共用缓存。

### 全分块 OCR

EasyOCR 对每个 640×640、重叠 96 像素的分块执行识别。文字框恢复到整图坐标后，与 RapidOCR 全图 token 合并；同文本且空间高度重叠时保留高置信结果。该方式比 EasyOCR-only 更稳定，因为 RapidOCR 负责整体文字，EasyOCR 补充局部小字。

### Hungarian 位号分配

每个元件候选与每个 DESIGNATOR token 建立分数，特征包括距离、相对方向、文字尺度、OCR 置信度和 type-prefix 兼容性。全局一对一分配保证同一 R1 不会同时绑定两个电阻。

### 选择性局部 OCR

元件初步关联后，程序筛出缺字段或低置信元件；按 bbox 尺寸动态扩展 ROI，放大两倍、增强对比度和锐化，再调用 EasyOCR。实验显示它更快，Component 和 Value 略高，但网络相关指标下降，因此没有替换默认 Hybrid。

## 七、A/B 结果

所有分数都是本地非官方诊断结果。主表使用最新版 0001～0150；新权重训练过其中 120 例，所以泛化判断另看未参与训练的 dev30。0151～0200 和 10 Golden 未读取。

### 完整 150 例

| 版本 | ComponentF1 | Symbol F1 | BBox F1 | Type Acc | PinF1 | NetF1 | LineF1 | PairF1 | 总分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 V4，旧权重 + RapidOCR | 0.3110 | 0.6667 | 0.7702 | 0.8496 | 0.1025 | 0.1112 | 0.0193 | 0.0092 | 15.98 |
| 只换同组 V2 新权重 | 0.3762 | 0.8137 | 0.8768 | 0.9249 | 0.1719 | 0.1506 | 0.0263 | 0.0171 | 21.12 |
| V4.1 新权重 + Hybrid OCR + 新规则 | **0.3810** | **0.8137** | **0.8768** | **0.9249** | **0.1751** | **0.1579** | **0.0299** | **0.0181** | **21.63** |

新权重贡献总分 +5.14；在此基础上 Hybrid OCR 与 V4.1 规则再贡献 +0.51。最终相对原 V4 提高 +5.66。

### 无训练重叠 dev30 的 OCR 消融

| OCR | ComponentF1 | 位号 OCR micro F1 | PinF1 | NetF1 | LineF1 | PairF1 | 总分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RapidOCR 1.4.4 全图 | 0.3556 | 0.5101 | 0.1645 | 0.1399 | 0.0163 | 0.0097 | 19.84 |
| EasyOCR 全分块 | 0.2815 | 0.3553 | 0.1203 | 0.1092 | 0.0173 | 0.0126 | 15.45 |
| Hybrid：Rapid 全图 + Easy 每块 | 0.3639 | **0.5532** | 0.1710 | **0.1580** | **0.0280** | **0.0163** | **21.00** |
| Rapid + 低置信元件局部 Easy | **0.3646** | 0.5505 | **0.1733** | 0.1533 | 0.0179 | 0.0082 | 20.82 |

这证明“每块均 OCR”对总分有正收益；选择性局部路径速度更好，但当前连接质量不及 Hybrid。

## 八、错误与回退

最终 150 例仍有 661 个 Detection Miss，位号 OCR micro F1 只有 0.5292，说明约一半可观察位号没有正确进入 token 集。Value accuracy 从原 V4 的 0.5979 降到 0.5431；新 token 会参与 Value 竞争，后续需要更严格的局部关系和置信度校准。BEST 中 Name 阶段关闭，因此 `NameAccuracy=1.0` 只是没有错误覆盖默认字段，不能解释为已经识别芯片型号。

新元件框使下游同步改善，但绝对值仍低：PinF1 0.1751、PinPairF1 0.0181。预测 singleton net 比例约 79.29%，说明 terminal 定位、pin number/pinname 和真实连接恢复仍是主要瓶颈。

0002 是局部反例：Symbol/BBox、Pin、Net、Line 均提高，但严格 ComponentF1 从 0.6957 降到 0.4167，总分从 40.83 降到 39.11，原因是文字字段变更导致严格语义匹配减少。该反例保留在报告中，没有为单张图加入特判。

## 九、没有采用的方法

- 没有重新训练或更换 YOLO 架构；第一轮先验证现有同组权重。
- 没有采用自动位号，因为正式评分要求可观察位号，猜测 U1/U2 可能刚好相反。
- 没有采用 0057 专用 MOSFET 模板。
- 没有将 EasyOCR 单独作为主 OCR。
- 没有启用旧 Geometry Fallback 和 Name 阶段，因为已有 A/B 显示负收益。
- 没有接入 HAWP、Netlistify Transformer，也没有修改 Pin/Wire/Topology。

## 十、当前限制与下一步

V4.1 已显著改善 Component 前端，但不是完整比赛解法。下一阶段的优先级为：

1. 基于新 bbox 重做 terminal candidate 与 pin-to-component attachment。
2. 对 terminal 周围做方向化、多旋转 pin number/pinname OCR，并明确区分 internal pin key。
3. 收紧 Value 与 Name 的局部关联，消除本轮 Value 回退。
4. 在 Pin 稳定后再重做 component/text suppression、wire CC 和 topology。
5. Detector 定稿前做 5-fold OOF；只有当规则式 junction 仍是明确瓶颈时，才评估 HAWP keypoint。

`OFFICIAL_SCORE = FALSE`

`SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE`
