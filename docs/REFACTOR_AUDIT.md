# V4.1 行为保持型重构审计

## 审计范围与约束

本审计基于 `PCB_Competition_Solution_V4_1_ComponentFusion.zip` 的实际源码、导入关系、现有测试、150 例预测结果和报告完成。当前目录不是 Git 仓库，因此本轮在独立副本中工作，不修改原工程目录和原压缩包。开发数据边界继续固定为 `0001～0150`，`0151～0200` 与 `10_GTcase` 不参与重构开发或验证。

本轮只调整调用结构。YOLO 权重、OCR 后端、全部阈值、正则表达式、匹配代价、Pin/Wire/Topology 算法、坐标变换、导出格式和评测器均保持不变。

## 当前真实入口和 Pipeline

入口是 `main.py::main()`，单图执行函数是 `main.py::run_one()`。默认参数为：

- Pipeline：`v4_component`
- Component 消融阶段：`BEST`
- Text rules：`v4_1`
- OCR：`hybrid`
- Component 权重：`models/component_yolo11n_continue_v2_best.pt`
- Wire：V3
- Topology：V3
- Variant：`baseline`

默认真实数据流如下：

```text
main.py
  ├─ 构造 HybridOCR(TiledEasyOCR, RapidOCR)
  ├─ 构造 YoloComponentDetector
  ├─ read_image()
  ├─ vision_v4.detect_scene_v4()
  │    ├─ ocr.recognize()                         Text/OCR
  │    ├─ component_detection_v4.detect_components_v4()
  │    │    ├─ YOLO tiled inference + class-aware NMS
  │    │    ├─ optional geometry proposals
  │    │    ├─ proposal fusion
  │    │    ├─ token role classification
  │    │    └─ designator/name/value assignment
  │    ├─ optional selective local OCR + component rerun
  │    ├─ pin_detection.terminal_candidates_v3() Pin Localization
  │    └─ pin_semantics.assign_pin_semantics()    Pin Semantics
  ├─ wire_v3.extract_wire_v3()
  ├─ topology_v3.build_topology_v3()
  ├─ submission.export()
  ├─ submission.validate_strict()
  └─ 写 result.json / diagnostics.json / debug images
```

## 核心数据结构

`pcb/schema.py` 定义稳定公共结构：

- `Text`：OCR 文本、OpenCV 左上坐标 bbox、置信度和来源。
- `Component`：key、官方 type、bbox、name、value、pins、body_bbox、confidence 等。
- `Pin`：导出后缀 number、name、tip、base、side、置信度、可导出标记等。
- `Net`：pin 引用列表和物理线段列表。
- `Scene`：图像尺寸、components、texts、nets、diagnostics。

内部几何统一使用 OpenCV 左上原点；`submission.export()` 只在 IO 边界调用 `opencv_to_target()` / `opencv_bbox_to_target()` 翻转一次为官方左下原点坐标。

## 当前职责归属

| 职责 | 当前实际文件/函数 |
|---|---|
| Text/OCR | `pcb/vision.py::OCR`、`pcb/ocr_backends.py` |
| Text role | `pcb/text_detection.py::classify_tokens` |
| Component V4.1 | `pcb/component_detection_v4.py::detect_components_v4` |
| YOLO | `pcb/component_detector_yolo.py::YoloComponentDetector` |
| Component text association | `pcb/component_text_assignment.py` |
| Component proposal fusion | `pcb/component_proposal_fusion.py` |
| Pin Localization V3 | `pcb/pin_detection.py::terminal_candidates_v3` |
| Pin Semantics V3 | `pcb/pin_semantics.py::assign_pin_semantics` |
| Wire V1/V2/V3 | `pcb/wire.py`、`pcb/wire_v2.py`、`pcb/wire_v3.py` |
| Topology V1/V2/V3 | `pcb/topology.py`、`pcb/topology_v2.py`、`pcb/topology_v3.py` |
| Submission | `pcb/submission.py::export/validate_strict` |
| 坐标 | `pcb/coordinates.py` |
| Evaluation | 根目录 `evaluate*.py` 与 `pcb/evaluate.py` |
| Batch IO/holdout guard | `main.py`、`pcb/io.py`、`pcb/data_policy.py` |

## 主要耦合点

1. `vision_v4.detect_scene_v4()` 同时承担 Text、Component、Pin Localization、Pin Semantics 四个阶段，导致替换 Pin 时必须修改 Component 前端文件。
2. `main.run_one()` 直接写死不同 Pipeline 对应的 Wire/Topology 组合，Pipeline 知道具体算法函数。
3. OCR 对象、YOLO detector、component stage 字符串和 text rules 通过位置参数跨层传递，没有统一运行上下文。
4. Component 的选择性局部 OCR 会更新 texts；该兼容行为需要通过 Component Stage 输出更新后的 texts，不能简单丢弃。
5. Pin Localization 与 Pin Semantics 通过未定义契约的 terminal 字典交换 `tip/base/side/method/wire_support_score`。
6. Wire V3 依赖 `Scene.components[].pins[].base/tip` 和 `scene.diagnostics["wire_stroke_width"]`；Topology 依赖 Wire mask 并原地写入 `scene.nets` 与 diagnostics。
7. `pcb/io.py` 已经是模块文件，不能在不增加迁移风险的前提下同时建立同名 `pcb/io/` 包。
8. Diagnostics 由多个算法原地合并，重构必须保留现有顶层字段和原有更新顺序。

## 文件依赖概览

```text
main
 ├─ io, vision, submission
 ├─ vision_v4
 │   ├─ component_detection_v4
 │   │   ├─ component_detector_yolo
 │   │   ├─ component_proposal_fusion
 │   │   ├─ component_text_assignment
 │   │   └─ text_detection
 │   ├─ pin_detection
 │   ├─ pin_semantics
 │   └─ schema / official_types
 ├─ wire_v3 → wire_v2
 └─ topology_v3 → topology_v2 → wire helpers
```

未发现由上述主链构成的循环导入。主要风险来自未来让低层 Stage 反向导入 `pipeline.py`；目标架构会禁止这种方向。

## 重构风险与控制

| 风险 | 控制措施 |
|---|---|
| OCR/YOLO 调用次数或顺序变化 | Wrapper 按旧函数的顺序调用；先做合成夹具 parity，再跑真实结果 parity |
| 选择性局部 OCR 丢失 | Component V4 Stage 原样保留两遍 component 流程并输出更新后的 texts |
| Component 可变对象被复制或排序 | Stage 间传递原对象和原列表，不新增排序/深拷贝 |
| Pin 坐标被再次转换 | Stage 全程使用 OpenCV 坐标，只有 Submission Stage 转换 |
| Wire/Topology diagnostics 改变 | 继续调用原函数并允许其原地更新同一 `Scene` |
| JSON 字段或顺序变化 | Submission Stage 直接调用原 `export()` 和 `validate_strict()` |
| 默认 CLI 行为变化 | 保留 legacy 默认入口；新增显式 `--orchestrator modular` |
| 同名 `pcb/io.py` 冲突 | 新建 `pcb/output/` 包作为 Submission Stage 包装层 |
| 缺少项目内 OCR vendor | 记录为交付环境问题；验证使用原 V4.1 同版本 runtime，不改变 OCR 实现 |

## 重构边界

公共层只包含：Stage 接口、输出容器、Registry、Pipeline 配置、只读运行上下文和编排。算法主体继续保留在现有文件。新目录中的版本文件仅作 Adapter/Wrapper，不复制或改写算法公式。

第一阶段完成条件：当前 V4.1 的每个 Stage 都能通过 Registry 构造，模块化 Pipeline 能按旧顺序调用同一算法，并得到逐字段相同的 prediction。未通过 parity 前不搬迁任何算法主体。
