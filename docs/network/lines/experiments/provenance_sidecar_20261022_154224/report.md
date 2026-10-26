# NetLine provenance sidecar P0

时间：2026-10-22（Asia/Shanghai）
工作区：`D:\EDA\EDA-NetLine-Isolated`
Python：`D:\EDA\EDA\.venv\Scripts\python.exe`（Python 3.12.10）

## 已核实的输入/输出契约

- `WireGraph.paths` 是 tuple of paths；每个 path 点是 `(row, column)` 整数坐标。
- `WireGraph.reference_segments` 按 path ID 索引，元素是几何线段列表，但不带原始 path edge 区间。
- `NetworkPlan.groups` 按调用顺序持有 `NetGroup(group_id, pins, path_ids, contact_points)`。
- 三个后端原有 `run(graph, plan)` 返回共享 `GeometryResult(tuple[GroupLines, ...])`；每个 `GroupLines` 持有 `group_id` 和按原顺序排列的段。
- 共线候选只在同一路径内依据 `cross == 0 and dot > 0` 合并；此规则没有加入容差。

## 实现

新增 `pcb/net_lines/provenance.py` 中的不可变 sidecar 类型：

- `GeometryProvenance.groups` 与 GeometryResult 的组顺序、group ID 对齐。
- 每个 `SegmentProvenance` 使用 `(group_id, segment_index)` 作为当前一次运行内的稳定引用，并保存端点和一个或多个来源记录。
- `SegmentSource` 记录来源种类、可用的 path ID、path-local source edge index 元组、reference segment index 或 contact point 输入索引；另含 `complete/unknown` 状态和 unknown 原因。

三个后端都提供新增的 `run_with_provenance(graph, plan) -> (GeometryResult, GeometryProvenance)`。原有 `run()` 仍返回原 `GeometryResult`，侧车不进入共享结果结构；不开启 trace 时不收集来源索引。

- `path_v1`：每条输出路径段对应恰好一个相邻点边索引 `(i,)`。行列到 `(x, y)` 的转换不改变 path-local 索引。重复坐标形成的零长度路径边仍有自己的索引；空路径和单点路径没有路径段记录。
- `path_collinear_v1`：合并段保存全部且有序的源边索引。合成测试验证每个输入边只映射一次、无遗漏，转角、反向边、退化边、path/group 边界保持分离。原 `_merge_path` 几何辅助函数保留 geometry-only 语义。
- `legacy_v2`：每个 reference segment 记录所在 path ID 和 reference segment index，但其 `source_edge_indices` 为 `None`、`resolution="unknown"`，原因明确指出上游 reference segments 不保留原始 path-edge 区间。没有按几何重合或距离推断来源。
- 接触点来源记录其组内输入索引。相同零段仍按原有规则去重，多个重复 contact 输入索引共同引用唯一输出段。Sidecar 不含 Pin 字段，不代表 Pin attachment 或电气连接。

## 测试与检查

最终完整 NetLine 测试：58/58 通过；详细命令、退出码及 scope/diff 结果见同目录 `test_results.json`。

新增测试覆盖：三个 backend 的 run 返回契约及 trace 几何相等；路径坐标/边索引；共线边完整映射与转角；组和路径隔离；重复接触点；路径零长度边与 contact 同段；legacy unknown 来源；空/单点路径；输入只读和重复运行确定性。

## 边界与剩余工作

- 输出引用是确定性的 run-local 索引；不是跨运行持久 ID，也未连接最终导出的 `edge_id`。
- 目前 NetLine 能追溯 path 候选自己的输入边；不能从 `reference_segments[path_id][j]` 还原原始路径边区间。要补全 legacy 来源，需要上游 graph builder 提供 reference-segment 到 path-edge interval 的显式映射。
- `contact_points` 只有坐标，没有可靠的 Pin 配对引用，因此 sidecar 仅记录 contact 输入索引；不得据此生成 Pin→edge 关系。
- 本轮未修改 registry、默认配置、正式流水线、共享 GeometryResult 或其他 Owner 文件；没有运行推理、数据评估、Git 写操作。
