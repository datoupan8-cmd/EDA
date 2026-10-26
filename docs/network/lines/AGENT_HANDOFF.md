# NetLine Owner 交接说明

更新日期：2026-10-24。本文记录当前 NetLine 模块接口、已验证结果和跨 Owner 待办，供后续开发与集成使用。它不授权切换默认版本或修改其他 Owner 的文件。

## 1. 职责与修改边界

NetLine 把既有 `WireGraph` 和 `NetworkPlan` 转成线段表达。它负责线段几何，不负责导线检测、路径构建、网络分组或 Pin 定位。

NetLine Owner 可维护 `pcb/net_lines/`、`configs/net_lines/`、`tests/net_lines/`、`models/net_lines/` 和 `docs/network/lines/`。不要在本模块任务中修改 `pcb/wire_stage/`、`pcb/network_shared/`、`pcb/net_hypergraph/`、`pcb/topology_stage/`、Component、Pin、公共 schema、Submission 或正式流水线入口；接口需求应交由相应 Owner 单独审议。

输入坐标中 `WireGraph.paths` 使用 `(row, column)`；输出线段使用原图左上角坐标 `(x, y)`。NetLine 不翻转 y，不改变 `group_id`、Pin 成员或网络归属，也不把几何接近当作电气连接证明。

## 2. 已注册后端与默认值

`pcb/net_lines/registry.py` 注册三个后端：

| 版本 | 用途与输入几何 | 行为 |
|---|---|---|
| `legacy_v2` | `WireGraph.reference_segments` | 保留现有基准线段输出，并按组添加去重的接触点零长度段。 |
| `path_v1` | `WireGraph.paths` | 将每条路径的相邻点逐边输出；保留所有路径点、转角和路径顺序。 |
| `path_collinear_v1` | `WireGraph.paths` | 在单条路径内部，只合并严格共线且同向的连续边；不跨路径、组或网络合并。 |

当前 `configs/net_lines/current.json` 为 `{"version":"legacy_v2"}`。**正式默认仍是 `legacy_v2`**；实验结果不构成切换默认的授权。三个版本均保留为独立注册项。

## 3. 接口和调用

三个实现均提供：

```python
run(graph, plan) -> GeometryResult
run_with_provenance(graph, plan) -> tuple[GeometryResult, GeometryProvenance]
```

`GeometryResult` 中的组顺序和 `group_id` 应与 `NetworkPlan.groups` 对齐。每个组只表达该组的线段；NetLine 不重新分配路径或 Pin。

直接从注册表创建实例可使用 `pcb/network_shared/selection.py` 中的 `backends(versions)`，版本映射必须同时含 `net_hypergraph` 和 `net_lines`。Topology V3 也支持显式注入：

```python
TopologyStageV3(network_versions={
    "net_hypergraph": "legacy_v2",
    "net_lines": "path_collinear_v1",
})
```

这只描述现有 Python 接口。正式 CLI 目前没有 `--net-lines-version` 参数；如需 CLI 覆写，应由 Integration Owner 实施，并保证不传参数时沿用现有选择、不改写默认配置、也不影响其他 Stage。不要通过修改默认配置来模拟单次实验。

## 4. provenance sidecar

`pcb/net_lines/provenance.py` 定义 `GeometryProvenance`、`GroupProvenance`、`SegmentProvenance` 和 `SegmentSource`。调用示例：

```python
geometry, trace = PathLines().run_with_provenance(graph, plan)
# 或
geometry, trace = CollinearPathLines().run_with_provenance(graph, plan)
```

trace 按输出组和段顺序对齐，记录 `group_id`、组内 `segment_index`、输出端点及来源。路径来源包含 `path_id` 和 path-local `source_edge_indices`；参考线段来源可包含 `reference_segment_index`；接触点来源包含 `contact_point_index`。来源不能验证时使用 `resolution="unknown"` 和原因，不得按坐标重合或距离猜测。

`path_v1` 每条输出段对应一条路径边。`path_collinear_v1` 的一个输出段可对应同一路径上连续的多条源边；索引按原始路径顺序记录。`legacy_v2` 只有在 `WireGraph.reference_segment_edge_ranges` 与参考段、路径端点都对齐并通过校验时，才报告完整源边索引，否则标为 unknown。

**provenance 可追踪“输出段 → group → path → 源边”，但不能直接证明 Pin→edge attachment。** `NetworkPlan` 可把 `group_id` 关联到该组 Pin 成员，sidecar 没有可靠的单 Pin—单线段电气配对引用。不得以最近距离、投影或几何相交填补这个缺口。

sidecar 是可选、调用期间的旁路返回值，不改变 `run()` 返回类型或公共 `GeometryResult`。常规组装链路只调用 `run()`，不会自动保留或写出 trace。因此它可用于 NetLine 调试与验证；若最终结果、缓存或评测产物需要持久化 provenance，需 Integration/Submission Owner 决定承载方式及 schema 兼容性。

## 5. 已完成的验证与证据边界

### 13 样本几何比较

实验记录位于 `experiments/open_gt_wire_geometry_20261024_200822/`。在获准的 13 个开发样本已有结果上，`path_collinear_v1` 相比 `path_v1` 将线段数从 17,774 降至 2,925，约减少 83.5%；报告中的宏观 GT 几何覆盖相同，几何精度也接近。该对照说明表示更紧凑，**不证明官方识别准确率提升**，也不作为更换默认后端的依据。报告中的 Wire F1 为暂定第三方评测结果，不是官方得分。

### 样本 2 provenance

样本级验证记录位于 `experiments/netline_provenance_sample2_20261024_203218/`。它复用保存的样本 2 Scene 与 Wire mask，没有重跑图片推理。对保存 mask 重建 Graph、对保存 Scene 与 Graph 重建 Plan，并与旧追踪中的 mask 哈希、路径数、边数及相关组路径关系核对。完整 Graph/Plan 对象当时未落盘，所以这是可复现的重建验证，不是原对象逐字节回读。

| 后端 | 组数 | 输出段 | 完整追溯的源边 | 多边合并段 | 结果 |
|---|---:|---:|---:|---:|---|
| `path_v1` | 45 | 1,938 | 1,938/1,938 | 0 | trace 与 `run()` 几何一致；无缺失、重复或额外源边。 |
| `path_collinear_v1` | 45 | 346 | 1,938/1,938 | 206 | 每个合并段的来源在同一路径内连续、升序且严格同向共线；未跨组、路径、拐角或反向点。 |

两后端输出均与 trace 端点和组内段索引对齐，输入 Graph/Plan 未改变。该样本有 169 条 reference segment，legacy 的 provenance 映射 169/169 完整。本样本 Plan 没有 contact points，因此真实样本未验证接触点来源；重复接触和零长度段行为由 NetLine 合成测试覆盖。

与来源对应的定向测试命令及最近一次结果：

```powershell
D:\EDA\EDA\.venv\Scripts\python.exe -B -m unittest tests.net_lines.test_provenance tests.net_lines.test_reference_provenance -v
```

13 项通过，退出码 0。NetLine 完整测试此前记录为 49/49 通过；本次交接只更新文档，没有重跑全套测试。详情以对应实验目录的 `report.md` 和 `results.json` 为准。

## 6. 已知上游问题

问题在 Wire Owner 的 palette 候选选择，不在 NetLine。普遍性核查记录位于 `experiments/wire_palette_prevalence_20261024_202141/report.md`。在该报告的 2 px 诊断口径下，13 个样本均出现 palette 选择造成的线几何漏检；共有 737 条 GT 边覆盖低于 80%，其中 706 条（95.8%）的首次缺失归于 palette 选择：144 条归于 72% 分数门槛，560 条归于最多选择两类的上限。另有少量 line-width/morphology 与 suppression/region 过滤案例。该报告的覆盖口径是诊断指标，不是官方 Wire F1。

样本 2 的隔离 0.71 门槛实验位于 `experiments/wire_palette_gate_071_20261024_183215/`：目标线恢复，但跨 13 样本回放只有样本 2 改变；这不是已接受的生产修复。NetLine 不应自行调整 Wire 阈值或在 NetLine 内弥补上游 mask 缺口。Wire Owner 应评估候选多样性、top-two 限制及次级过滤原因，并独立决定生产改动和验证范围。

## 7. 交接后的跨 Owner 事项

**必须在跨模块交付前决定：**

1. **Integration Owner：正式入口覆写。** 如需命令行选择候选，评估增加可选 `--net-lines-version` 并只覆盖本次 Topology V3 的 NetLine 版本映射；无参数时行为完全不变，其他 Stage 映射不变，非法版本沿用注册表错误，非 V3 行为要明确并测试。NetLine Owner 不修改 `tools/run_team_pipeline.py`。
2. **Integration/Submission Owner：provenance 持久化。** 只有下游明确需要最终文件携带 trace 时才设计 sidecar 的保存位置、生命周期和兼容性；本轮不修改最终导出 schema。
3. **NetHypergraph Owner（仅当要求逐 Pin→edge 证明时）：attachment 引用。** 需要上游为 Pin attachment 提供稳定引用，并定义它如何对应 Plan 路径或几何对象。最近距离或相交不能替代该契约。本轮组—路径来源验证不依赖该功能。
4. **Wire Owner：palette 选择问题。** 单独跟进上节所列 Wire mask 漏检；NetLine 仅消费现有 Graph/Plan，不应代替 Wire 修复。

## 8. 默认与后续验证原则

- 保持 `legacy_v2` 为默认，直到团队批准切换并有固定、可信评估证据。
- 不把本地几何覆盖、线段数量、连通关系等价或暂定评测器分数称为官方准确率。
- 对候选版本继续运行合成边界测试；跨模块集成按 Owner 授权单独完成。
- 任何逐 Pin→edge 或最终导出 provenance 需求先形成接口决策，不在 NetLine 内推断或扩展公共 schema。
## 9. Checkpoint 状态与未提交跨 Owner 依赖（2026-10-26）

本分支按阶段性 checkpoint 提交 NetLine 自有目录内容；这不是跨模块集成验收。以下三项本地修改不属于本次 NetLine 提交，必须由对应 Owner 另行审议、提交并集成：

- `pcb/network_shared/contracts.py`：为 `WireGraph` 增加可选的 `reference_segment_edge_ranges` 来源字段。缺少此字段时，NetLine 不能从 `reference_segments` 可靠恢复源路径边，只能将该来源标为 `unknown`。
- `pcb/network_shared/graph.py`：在构建 Graph 时生成并按顺序对齐上述参考线段来源记录。缺少此生成逻辑时，没有可验证的参考线段到源边映射。
- `pcb/wire_stage/v1.py`：增加保留简化顶点来源索引的能力，供 Graph 构造建立半开源边区间。缺少该来源索引时，不能从简化后的几何安全推导原始路径边范围。

这三项跨 Owner 文件在 checkpoint 中保持为本地未提交修改；本次提交不包含它们，也未删除或回退它们。`legacy_v2` 的参考线段来源追踪只有在上述字段和 Graph 生成逻辑存在且通过校验时才可解析为源边；在不含该上游依赖的基线中应保持 `unknown`。`tests/net_lines/test_reference_provenance.py` 中依赖该映射的场景不能脱离这些上游改动单独复现。

本地完整 NetLine 测试结果为 **62/62 通过**，但该结果是在同时包含上述三项未提交上游修改的工作区环境中取得；它不表示本 checkpoint 分支独立具备全部 provenance 能力，也不等同于跨模块集成验收。提交后若仅基于 checkpoint 分支运行依赖参考线段来源映射的测试，相关映射可能为 `unknown`，该差异是已知集成依赖，不应通过猜测来源掩盖。