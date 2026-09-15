# 迁移与适配映射

本轮采用 Wrap First。算法主体未搬迁，新增路径是稳定接口或兼容导出。

| 原文件/函数 | 新边界 | 是否改算法 | 说明 |
|---|---|---:|---|
| `main.py::run_one` | 保留 Legacy | 否 | 原函数主体保留 |
| 无 | `main.py::run_one_modular` | 否 | 新入口只调用公共 Pipeline 并沿用原写盘/debug 逻辑 |
| `vision.py::OCR` | `text/ocr.py` | 否 | 兼容导出 |
| `ocr_backends.py::*OCR` | `text/stage.py::CurrentTextStage` | 否 | 调用已构造的同一 OCR 对象 |
| `text_detection.py` | `text/roles.py` | 否 | 兼容导出，原正则和规则未改 |
| `detect_components_v3` | `component/v3.py::ComponentStageV3` | 否 | Wrapper 保留原 missing-key fallback |
| `detect_components_v4` | `component/v4.py::ComponentStageV4` | 否 | Wrapper 保留选择性局部 OCR 的调用顺序 |
| `component_detector_yolo.py` | `component/detector_yolo.py` | 否 | 兼容导出；权重/阈值/NMS/tile 未改 |
| `component_text_assignment.py` | `component/text_assignment.py` | 否 | 兼容导出；代价和匹配未改 |
| `component_proposal_fusion.py` | `component/postprocess.py` | 否 | 兼容导出 |
| `component_detection.py` geometry 函数 | `component/geometry.py` | 否 | 兼容导出 |
| `terminal_candidates_v3` | `pin/localization/v3.py::PinLocalizationStageV3` | 否 | 原函数逐 Component 调用，原坐标裁剪不变 |
| `assign_pin_semantics` | `pin/semantics/v3.py::PinSemanticsStageV3` | 否 | 原函数逐 terminal 调用 |
| `extract_wire` | `wire_stage::WireStageV1` | 否 | Wrapper |
| `extract_wire_v2` | `wire_stage::WireStageV2` | 否 | Wrapper |
| `extract_wire_v3` | `wire_stage::WireStageV3` | 否 | Wrapper |
| `build_topology` | `topology_stage::TopologyStageV1` | 否 | Wrapper |
| `build_topology_v2` | `topology_stage::TopologyStageV2` | 否 | Wrapper |
| `build_topology_v3` | `topology_stage::TopologyStageV3` | 否 | Wrapper |
| `submission.export/validate_strict` | `output::OfficialSubmissionStage` | 否 | 原导出与验证双重调用 |
| `coordinates.py` | `output/coordinates.py` | 否 | 兼容导出，转换仍只在 Submission 边界发生 |
| `schema.py` | 继续原路径 | 否 | 公共数据契约未定义第二套 schema |

## 目录命名说明

现有项目已有 `pcb/wire.py`、`pcb/topology.py` 和 `pcb/io.py`。Python 同一路径同时存在同名模块与包会造成导入歧义，因此新包装包使用 `wire_stage/`、`topology_stage/` 和 `output/`。这避免重命名经过验证的旧模块，也降低行为变化风险。

## 未迁移内容

`evaluate*.py`、训练/消融脚本、annotation reader 和 official target reader 不属于 image-only 推理 Stage，本轮保持原路径和行为。后续如需模块化 Evaluation，应单独做行为快照和 parity，不能混入本次推理重构。
