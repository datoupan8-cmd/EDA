# NetLine provenance 样本 2 验证

## 结论

在样本 2 已保存的真实 Wire mask 与预测 Scene 上，重建 Graph 和 Plan 后，`path_v1` 与 `path_collinear_v1` 的 provenance 均完整对应实际输出几何。未发现 NetLine provenance 错误。`path_collinear_v1` 没有跨路径、跨组/网络、拐角或反向点合并。

这支持 NetLine 的**组—路径—源边几何追溯**进入模块交接；不代表已建立逐 Pin→edge 电气连接，也不代表 Ground Truth 准确率提升。

## 输入与重建依据

- 输入只复用 `wire_case2_trace_20261024_165326/formal_output/2/scene.json` 和 `wire_mask.png`，没有重跑图片推理，也没有读取标注。
- mask SHA-256 为 `41a3a232212aa30b695bd23e7e2adfcde505d3571ad9c3b150a86a8470fe28d4`，与原追踪记录一致。
- 从保存 mask 调用现有 `build_graph()` 得到 128 条路径、1,938 条相邻源边；与追踪记录相同。记录中的 path 6、7、8、9 坐标及点数逐项一致。
- 从保存的预测 Scene、重建 Graph 调用 `LegacyHypergraph.run(mode="directional", merge_labels=False)` 重建 Plan：45 个组、128 条路径各归属一个组；追踪记录中的 group 7/path_ids `[0,7,10]` 和 group 8/path_ids `[6,8,9]` 精确一致。35 个含 Pin 组的 Pin 成员集合与保存 Scene 的 35 个导出 net 一致。完整 Graph/Plan 对象原先未落盘，因此这是基于保存输入和同一实现的可重复重建，而非原对象逐字节回读。
- `reference_segment_edge_ranges` 与 169 条 reference segment 对齐；本次同时调用 `legacy_v2.run_with_provenance()` 核验，169/169 来源完整、unknown 为 0。

## provenance 核验结果

| 后端 | 组数 | 输出段 | 合并多条源边的段 | 被追溯源边 | 缺失/重复/多余源边 | 几何与 `run()` 一致 |
|---|---:|---:|---:|---:|---:|---|
| `path_v1` | 45 | 1,938 | 0 | 1,938/1,938 | 0 / 0 / 0 | 是 |
| `path_collinear_v1` | 45 | 346 | 206 | 1,938/1,938 | 0 / 0 / 0 | 是 |

逐段检查确认：

- trace 的 group 顺序、段索引和端点与 `GeometryResult` 一一对齐；输出组顺序及路径顺序与 Plan 一致。
- 每个路径来源都属于该段的 `group_id` 所列 `path_ids`；每个输出段只引用一条源路径。没有跨路径合并，也没有路径被分配到多个组。
- 源边索引连续、升序、无重复；源边首尾连接，首末坐标精确重建输出段。每条被 Plan 选择的原始路径边恰好出现一次。
- 206 个多边源段的连续源边逐对满足严格 `cross == 0` 且 `dot > 0`；因此未跨拐角、退化反向或折返点。
- 两个后端的 group 7、8 几何与既有样本追踪保存值一致。运行前后 Graph、Plan 的路径、reference segment 和图像数组快照不变。
- 样本 Plan 的 contact_points 数为 0，因此本次真实样本没有 contact source 记录可验；重复接触点和零长度段规则由现有合成 provenance 测试覆盖。

## 对下游接口的含义

现有 sidecar 足以在本次后端调用中追溯“输出段 → group → path → 源边”，并可通过 `NetworkPlan` 的 `group_id` 关联该组 Pin 成员。它不建立某一条边对应某一个 Pin 的电气 attachment 证明。不得以距离或相交补造该关系。

sidecar 是可选的运行时对象；常规组装链路调用后端 `run()` 并消费 `GeometryResult`，不会自动保存 provenance。当前组—路径追溯不要求修改 NetHypergraph/Topology 接口；若未来需要逐 Pin→edge attachment，应由上游提供明确、稳定的 attachment 引用。若要把 sidecar 持久化到最终导出，则需另行由 Integration/Submission owner 审议，不属于本次 NetLine 改动。

## 验证

命令：

```powershell
D:\EDA\EDA\.venv\Scripts\python.exe -B -m unittest tests.net_lines.test_provenance tests.net_lines.test_reference_provenance -v
```

结果：13 项通过，0 失败，退出码 0。没有修改生产代码、默认后端、其他 Owner 文件或既有实验产物。