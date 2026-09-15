# Component 前端融合源码审计

> 审计对象：`PCB_Competition_Solution_V3` 与 `EDA_components_handoff_20260914`  
> 数据边界：只允许最新版官方公开集 `0001～0150`；`0151～0200` 与 `10_GTcase` 未用于开发。  
> 分数性质：本文引用的成绩均为本地非官方诊断结果，`OFFICIAL_SCORE = FALSE`。

## 一、结论先行

两套方案不应二选一。V3 保留完整的 `Scene → Pin → Wire → Topology → Submission` 下游接口、RapidOCR、文字角色、官方类型转换、特殊几何检测和调试链路；同组方案提供更强的 YOLO symbol detector、1280 像素切片推理、全图坐标恢复和跨切片 NMS。融合后的主线应是：YOLO 只预测 `bbox + type + confidence`，V3 OCR 独立识别文字，再通过受约束的一对一匹配产生 designator、Name 和 value，最后把结果装回 V3 `Component`。

同组代码中以下逻辑不得直接进入 V4 主线：

- `assign_named_fallback_designators()` 会按阅读顺序自动生成 `U1/U2/Q1...`，图片没有可观察位号时会制造高置信错误。
- `detect_mosfet_symbols()` 的二值模板在源码注释和交接文档中明确针对训练案例 0057 的 29×20 Datasheet 风格，存在明显风格过拟合风险。
- EasyOCR 与 YOLO、关联逻辑同时启用会让第一轮无法区分提升来源；第一轮保留 V3 RapidOCR。
- V3 的 `source_from_name()` 根据文件名中的 KiCad、Datasheet 等选择先验，违反“仅根据图像自适应”的约束，V4 不使用该分支。

## 二、项目 A：PCB_Competition_Solution_V3

### 2.1 原始数据流

```text
PNG
  ↓ RapidOCR
OCR Text
  ↓ classify_tokens()
DESIGNATOR / VALUE / PIN_NUMBER / PIN_NAME / MODEL_TEXT / OTHER
  ↓
detect_components_v3()
  ├─ detect_large_box_components()
  ├─ detect_capacitor_centers()
  ├─ detect_ground_components()
  ├─ OCR anchor + v3_priors.json 推测 bbox
  └─ legacy detect_scene() 按 key 补漏
  ↓
Scene(Component)
  ↓（本轮冻结）
Pin V3 → Wire V3 → Topology V3 → Submission
```

### 2.2 十五项审计回答

| 问题 | 实际实现 |
|---|---|
| Component bbox | 大矩形、平行极板、接地三角形；多数普通器件由 designator 附近的统计先验框产生，最后还有紧邻文字的方框回退。 |
| Component type | 主要由 designator 前缀映射，特殊几何检测固定为 box/c/gnd。 |
| designator | RapidOCR 后由 `DESIGNATOR_RE` 分类并直接作为 key。 |
| value | `VALUE` token 与 r/c/l 做类型兼容的一对一 Hungarian 分配。 |
| Name | box 在 legacy 分支从附近“字母+数字”文字取型号；V3 主组件检测没有完整独立 Name 分配。 |
| OCR 时机 | OCR 先于 Component；OCR 同时影响 proposal、type 和 key。 |
| 文字驱动 proposal | 是，普通元件的主要路径。 |
| 独立 Symbol Detector | 没有训练模型，只有几何启发式。 |
| geometry fallback | 大框、电容、GND，以及 legacy contour。 |
| legacy fallback | `vision_v3.py` 按 component key 去重后追加 legacy 元件。 |
| 重复去除 | 主分支按 key；大型框按边界近似去重；空间 IoU 去重不完整。 |
| NMS / matching | box-designator 与 value 使用 SciPy Hungarian；无统一 proposal NMS。 |
| 固定阈值 | Hough、三角形尺寸、距离上限、文本遮罩边界等。 |
| 数据统计先验 | `v3_priors.json` 的 source/type anchor cluster 与尺寸中位数。 |
| 过拟合风险 | `source_from_name()` 从文件名取 source；先验按 source 分支；fallback 按 key 去重会保留空间重复框。 |

### 2.3 最大优势与问题

优势是工程完整、官方 schema 已对齐、内部坐标统一为左上并在导出时只翻转一次、文本角色与下游接口可直接沿用。最大问题是 Component 建立仍依赖 OCR：OCR 漏掉 designator 时，普通符号通常无法出现。V3 前 150 例 Component macro F1 为 0.1888，预测 3578、GT 6080，召回约 16.7%，说明 symbol detector 是结构性瓶颈。

## 三、项目 B：EDA_components_handoff_20260914

### 3.1 真实文件与入口

| 能力 | 文件 / 函数 |
|---|---|
| 推理入口 | `scripts/predict_components.py` → `ComponentRecognizer.predict()` |
| YOLO detector | `src/pcb_parser/recognition.py::ComponentRecognizer.detect()` |
| 权重 | `artifacts/models/component_yolo11n_baseline/weights/best.pt`，约 5.57 MB |
| tile inference | `tile_origins()`；默认 1280、overlap 192 |
| tile→global | 每个 xyxy 加 tile 的 left/top |
| NMS | `components.py::non_maximum_suppression()`，当前是 class-agnostic greedy NMS |
| OCR | `ComponentRecognizer.recognize_text()`，EasyOCR 640 tile、96 overlap、2×放大 |
| designator | `associate_designators()`，类型兼容 + 距离 + OCR 置信度，全局最大权匹配 |
| Name | `associate_component_names()`，box 类候选过滤 + 全局最大权匹配 |
| value | `associate_values()`，r/c 最近兼容 value；不是全局唯一分配 |
| 训练 | `prepare_components_dataset.py`、`train_component_detector.py` |
| 评测 | `evaluate_component_baseline.py`、`evaluation.py` |
| 特殊回退 | `detect_mosfet_symbols()`，0057 风格二值模板 |

### 3.2 原始数据流

```text
PNG
  ↓ 1280 tiled YOLO11n
tile bbox/type/confidence
  ↓ global coordinate restore
class-agnostic NMS
  ├─ 0057-style MOSFET template proposal
  ↓
Component detections

PNG
  ↓ 640 tiled EasyOCR ×2
OCR tokens
  ↓
associate_designators() → associate_component_names()
→ associate_values() → assign_named_fallback_designators()
  ↓
components-only JSON（pins/nets 为空）
```

### 3.3 十五项审计回答

| 问题 | 实际实现 |
|---|---|
| Component bbox | YOLO11n 检测框；tile 恢复至全图；另有固定 MOSFET 模板框。 |
| Component type | YOLO 40 类名称直接输出。权重内嵌类别表。 |
| designator | OCR token 经 regex、type-prefix 兼容和全局匹配；之后还会对有 Name 的 unresolved 元件自动编号。 |
| value | r/c 附近最近 compatible token，每个 token 可被多个元件复用。 |
| Name | box/amp/block 等与型号 token 做全局匹配。 |
| OCR 时机 | Symbol detection 与 OCR 独立，最终关联。 |
| 文字驱动 proposal | 没有，YOLO 可在没有 designator 时保留 unresolved 框。 |
| 独立 Symbol Detector | 有，YOLO11n。 |
| geometry fallback | 只有小 MOSFET 模板，不是通用几何。 |
| legacy fallback | 无。 |
| 重复去除 | class-agnostic NMS；相互重叠的不同类型也会互相抑制。 |
| NMS / matching | greedy NMS；designator 与 Name 用依赖无关的 Hungarian 实现；value 为逐组件最近邻。 |
| 固定阈值 | detector conf 0.10、NMS 0.5、tile 1280/overlap 192、关联阈值 0.05/0.08。 |
| 数据统计先验 | type-prefix 常量；未使用 V3 的位置先验。 |
| 过拟合风险 | 0057 模板明确针对单一风格；自动编号会把不可观察 ID 伪装成预测；40 类未覆盖官方 43 类全部别名。 |

### 3.4 权重和训练隔离

权重文件真实存在。其内嵌类别包括 `gnd/r/c/v/box/.../seg` 共 40 类，包含官方历史拼写 `battary`、`transfomer` 和 `m3螺丝`；缺少 V3 官方类型表中的 `bjt_npn`、`dc`、`mosfet_npn`。训练脚本强制使用 0001～0150，并按 `case_id % 5 == 0` 得到 30 个内部 dev；交接报告称 dev type+bbox F1 为 0.5833，而严格 ComponentF1 仅 0.0326。该差距证明检测器比文字身份恢复强。

现有 `best.pt` 是用 120 个非 dev 案例训练的。因此：30 个固定 dev 案例可用于无训练重叠的迁移判断；若在全部 150 例报告 detector 指标，必须明确其中 120 例属于 in-sample development diagnostic，不能声称是泛化成绩。

## 四、共同模块、差异与融合决策

| 模块 | V3 | 同组方案 | V4 决策 |
|---|---|---|---|
| Scene / schema | 完整并已连接下游 | components-only | 保留 V3 |
| Symbol bbox/type | OCR+几何先验，召回低 | YOLO，dev type+bbox F1 0.5833 | 同组 YOLO 作为主 proposal |
| OCR | RapidOCR，已有缓存 | EasyOCR，需要单独模型 | 第一轮保留 V3 RapidOCR |
| Text role | 组件与 pin 角色已有拆分 | regex 较窄 | 保留并扩展 V3 |
| Designator matching | 先有文字再造框 | 全局类型约束匹配 | 采用全局一对一匹配，加入尺度/方向特征 |
| Name | 主线较弱 | box 类全局匹配 | 吸收过滤与匹配思想 |
| Value | 全局一对一 | 逐元件最近邻 | 保留 V3 全局唯一思想，改为统一 matcher |
| 大 box / GND / C | 几何检测 | 无 | 仅在最终 F 阶段作高精度补漏 |
| NMS | 不统一 | class-agnostic | 建立 type-aware proposal fusion |
| 自动位号 | legacy 可形成 key | 明确自动编号 | V4 禁止自动生成可观察 designator |
| 0057 模板 | 无 | 有 | 主线关闭并记录为未采用 |
| 下游 Pin/Wire/Topology | 完整 | 无 | 字节级冻结 V3 文件 |

## 五、V4 拟采用数据流

```text
Image ───────────────→ Tiled YOLO ─→ bbox/type/confidence ─┐
  │                                                        │
  └→ RapidOCR → Text Role Classification ──────────────────┤
                                                           ↓
                              Proposal Fusion / Type Resolution
                                                           ↓
                            Global Designator Assignment (1:1)
                                                           ↓
                            Name Assignment → Value Assignment
                                                           ↓
                                   Final V3-compatible Component
                                                           ↓
                              Frozen Pin → Wire → Topology → JSON
```

## 六、代码处理清单

### 保留

- V3 `schema.py`、`official_types.py`、`text_detection.py` 的角色体系。
- V3 RapidOCR 与缓存。
- V3 `detect_ground_components()`、`detect_large_box_components()`、`detect_capacitor_centers()`，仅作可消融 fallback。
- V3 diagnostics、strict validator、Pin/Wire/Topology/Submission/Evaluator。
- 同组 `tile_origins()`、tile→global 坐标与现有 `best.pt`。
- 同组 designator/Name 全局匹配的核心思想。

### 替换或修改

- 用 `component_detector_yolo.py` adapter 隔离 Ultralytics 结构。
- 用统一 `ComponentProposal` 与 type-aware NMS/fusion 替换按 key 去重。
- designator 采用新的全局 cost；Name 与 value 分别独立匹配并记录来源。
- 所有关键阈值集中在配置对象。
- 文件名 source prior 从 V4 主线移除。

### 默认关闭

- 读取顺序自动生成 designator。
- 0057 固定 MOSFET 模板。
- 同组 EasyOCR。
- 任何 `case_id` 或 source-name 分支。

## 七、第一轮实验协议

按 A～F 顺序运行，只有新增阶段；下游保持同一版本：

1. A：原 V3 baseline。
2. B：YOLO proposal only，沿用 V3 文本关联策略。
3. C：B + Designator Association V4。
4. D：C + Name Association。
5. E：D + Value Association。
6. F：E + Geometry Fallback。

主表覆盖 0001～0150，并同时单列无训练重叠的 30-case dev。全部 150 例只用于项目既定开发范围，不能被描述为 detector 泛化评估。评测除 ComponentF1 外，还分开报告 type+bbox、bbox、type、designator OCR、designator association、Name、value、重复率、每图误报和 fallback 效果。Pin/Net/Line 仅作冻结回归监测。

