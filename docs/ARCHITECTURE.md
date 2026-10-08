# 模块化 Pipeline 架构

## 总体结构

```text
PNG / OpenCV image
        │
        ▼
Text Stage ───────────────► texts
        │
        ▼
Component Stage ──────────► components + roles + updated texts
        │
        ▼
Pin Localization Stage ───► terminal geometry
        │
        ▼
Pin Semantics Stage ──────► pins attached to components
        │
        ▼
Wire Stage ───────────────► wire mask and debug artifacts
        │
        ▼
Topology Stage ───────────► scene.nets and skeleton
        │
        ▼
Submission Stage ─────────► official result.json
```

`pcb/core/pipeline.py` 只负责按顺序调用 Stage 和传递标准对象。它不包含 YOLO、OCR、阈值、Pin、Wire 或图算法。算法仍位于原有文件中，新 Stage 文件只调用原函数。

## 公共层

- `pcb/core/interfaces.py`：Stage 输入输出契约。
- `pcb/core/context.py`：图像名、尺寸、配置、OCR、detector 和可选 keypoint provider。
- `pcb/core/config.py`：集中 Stage 版本选择。
- `pcb/core/registry.py`：版本名到 Stage 类的显式映射。
- `pcb/core/pipeline.py`：稳定编排。
- `pcb/schema.py`：继续作为领域数据契约，未修改。

`PipelineContext` 不保存 components、pins、wires 或 nets。主要数据继续作为 Stage 的显式输入输出，避免形成隐藏状态容器。

## Stage 与现有算法

| Stage | 已注册版本 | 实际调用 |
|---|---|---|
| Text | `current` | 运行 CLI 已构造的原 OCR 对象 |
| Component | `v3` | `detect_components_v3()` + 原 legacy fallback |
| Component | `v4` | `detect_components_v4()` + 原选择性局部 OCR 流程 |
| Pin Localization | `v3` | `terminal_candidates_v3()` |
| Pin Semantics | `v3` | `assign_pin_semantics()` |
| Wire | `v1`、`v2`、`v3` | `extract_wire*()` |
| Topology | `v1`、`v2`、`v3` | `build_topology*()` |
| Submission | `official` | `export()` + `validate_strict()` |

Registry 只登记真实存在的算法。当前没有 Pin V4，因此没有创建假实现。

## 配置与运行

当前等价配置在 `configs/current.json`：

```json
{
  "text": "current",
  "component": "v4",
  "pin_localization": "v3",
  "pin_semantics": "v3",
  "wire": "v3",
  "topology": "v3",
  "submission": "official"
}
```

旧入口仍是默认值：

```powershell
python main.py --image input.png --output result.json
```

显式运行模块化入口：

```powershell
python main.py --image input.png --output result.json `
  --orchestrator modular `
  --pipeline_config configs/current.json
```

项目使用 JSON 配置而非 YAML，目的是不增加新的配置解析依赖。配置的职责相同。

## 如何新增 Pin Localization V4

1. 新建 `pcb/pin/localization/v4.py`，实现与 `PinLocalizationStage` 相同的 `run()`。
2. 在 `build_default_registry()` 中登记：

```python
registry.register("pin_localization", "v4", PinLocalizationStageV4)
```

3. 复制 `configs/current.json` 并只修改：

```json
"pin_localization": "v4"
```

Component、Pin Semantics、Wire、Topology 和 Pipeline 均不需要修改。

## 如何新增 Component V5

实现 `ComponentStageV5.run(image, text, context) -> ComponentStageOutput`，注册为 `component.v5`，随后只改配置。Pin 模块只依赖统一 `Component`、texts 和 roles，不导入 V5 内部代码。

## Legacy 与 Modular

- `--orchestrator legacy`：原 `run_one()`，也是默认入口。
- `--orchestrator modular`：Registry + Stage Pipeline。

两条路径并存用于回归。150例验证确认 Modular 的最终 prediction 与重构前 Legacy 基线逐字段一致后，才可由团队决定是否将 Modular 改为默认入口。

## 依赖方向

```text
main → core → Stage wrappers → existing algorithms → schema/helpers
```

算法文件不得反向导入 `pipeline.py`。负责人应通过实现契约和 Registry 接入版本，不应让一个 Stage 直接调用另一个 Stage 的内部函数。
