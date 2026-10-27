# Wire v8/v9 checkpoint and cross-module handoff

## 范围与结论

本文汇总授权开发集 `0001–0013` 的本地诊断结果。几何覆盖和连接配对均为诊断口径，不是官方竞赛评分。GT 只用于离线核对。三个 Wire 版本为 `v3_frame_guard`、`v8_baseline_anchored_recall` 和 `v9_branch_salvage_recall`；正式默认仍是 `v3_frame_guard`。

## Wire 几何与下游覆盖

| 版本 | Wire mask 覆盖率 | GT 边覆盖≥80% | 最终 Scene.nets 几何覆盖率 |
|---|---:|---:|---:|
| v3_frame_guard | 11.46% | 16/744 | 7.59% |
| v8_baseline_anchored_recall | 44.06% | 273/744 | 36.53% |
| v9_branch_salvage_recall | 50.86% | 318/744 | 42.40% |

覆盖率是 5 px 容差下按 GT 边采样计算的本地几何指标，不代表电气正确率。

## 可匹配引脚的连接配对

比较方法是将 GT 与预测中的 `component.pin` 标识精确匹配；GT 中跨多个网络出现的歧义 Pin 不参与配对评分；每个网络按引脚集合展开成无序 Pin 对。该配对展开是诊断方法，不宣称等同于官方评分规则。

| 版本 | TP | FN | FP | 召回率 | 配对精确率 | 匹配且无歧义的 Pin |
|---|---:|---:|---:|---:|---:|---:|
| v3_frame_guard | 2 | 324 | 54 | 0.61% | 3.57% | 370 |
| v8_baseline_anchored_recall | 2 | 324 | 54 | 0.61% | 3.57% | 370 |
| v9_branch_salvage_recall | 2 | 324 | 54 | 0.61% | 3.57% | 370 |

13 个样本累计另有 89 个 GT Pin 标识未匹配、61 个预测 Pin 标识未匹配、0 个歧义 GT Pin 标识。未匹配 Pin 未视为正确，也未计入上述 Pin 对指标。v8→v9 的可匹配预测 Pin 对中，新增正确 0、新增错误 0、丢失原正确配对 0。三版本的可匹配配对结果相同；因此 v9 的几何收益没有转化成可测得的 Pin 对连接收益。现有 54 个 FP 也不能归因于 Wire 版本差异。

## 首次损失阶段

| 版本 | Wire mask | Graph paths | Graph reference segments | Plan 选中路径 | GeometryResult | Scene.nets |
|---|---:|---:|---:|---:|---:|---:|
| v3_frame_guard | 11.46% | 8.41% | 8.41% | 8.41% | 8.41% | 7.59% |
| v8_baseline_anchored_recall | 44.06% | 41.35% | 41.35% | 41.35% | 41.35% | 36.53% |
| v9_branch_salvage_recall | 50.86% | 48.29% | 48.29% | 48.29% | 48.29% | 42.40% |

共 744 条 GT 边。Graph 阶段较 Wire mask 首次降低覆盖；v3/v8/v9 分别有 7/7/7 条此前 mask 覆盖≥80% 的边在 Graph 后低于 80%。reference simplification、Plan 选择和 GeometryResult 阶段没有进一步的整体覆盖下降。由于本分支的 `WireGraph` 没有逐段 source-edge provenance，path ID 的空间邻近只能作为线索，不能证明 GT 边与特定 path/源边的对应关系。

### GeometryResult 到 Scene.nets

组装函数 `pcb/network_shared/composite.py::assemble` 只为 `g.pins` 非空的 Plan 组建立 `Net`。本次回放中，v9 的 GeometryResult 有 2,622 条线段，最终 Scene.nets 有 2,216 条；另外 406 条线段属于无 Pin 组，组装时没有导出。v3/v8 分别有 384/396 条这种未导出的组线段。该事实证明当前组装条件会丢弃无 Pin 组的几何；它**不证明**这些几何应被视为有效电气网络，也不证明丢弃行为违反产品契约。

## 代表性样本证据

- 样本 `0001`，GT `net_4/edge_4`，CV 坐标端点 `(426,402)–(277,402)`：Wire、Graph、reference、Plan 和 GeometryResult 的覆盖均为 100%，Scene.nets 为 0%。空间邻近的 Graph `path_id=58` 被 Plan 放入 `group_id=58`，该组 `pins=[]`；按当前 `assemble` 条件该组不导出。path 与 GT 边的联系仍是几何线索，不是 provenance 证明。
- 同样在 `0001`，`net_4/edge_6` 的覆盖从 GeometryResult 的 98.88% 降到 Scene.nets 的 0%；附近也是 `path_id=58`、无 Pin 的 `group_id=58`。
- 样本 `0001`，GT `net_12/edge_2`，端点 `(1325,551)–(1325,570)`：Wire mask 覆盖 100%，Graph paths 约 60%，其后阶段维持约 60%。附近路径 ID `179/186/187` 落在含 `U3.UNK1` 的 `group_id=184`，路径 `195` 落在含 `U3.UNK2` 的 `group_id=196`。这是位置与组分配证据，缺少源边 provenance，不能据此断言精确路径来源或根因究竟是 Graph 断裂、Topology 分组还是 GT 表达差异。

## 各负责人需要核查的问题

- **Graph owner**：检查 Wire mask 到 `WireGraph.paths` 的 7 条高覆盖边跌破 80% 的情况，尤其 `0001 net_12/edge_2` 附近的路径分段；明确 skeleton tracing、路径筛选和简化各自造成的几何损失。建议提供可选的路径/source-edge 诊断映射，避免仅靠空间邻近归因。
- **Pin owner**：解释 89 个未匹配 GT Pin 标识与 61 个未匹配预测 Pin 标识（按样本限定）来自缺检、编号/命名规范差异还是输出字段缺失；未解决前，连接风险结论只覆盖 370 个精确匹配且无歧义的 Pin。
- **Topology owner**：在相同匹配 Pin 子集上核对 GT/预测组成员差异（324 个漏配对、54 个额外配对），确认差异是否来自真实拓扑推断、GT hyperGraph 语义或当前配对展开方法；不要将其整体归咎于 Wire。
- **Integration owner**：确认 `assemble` 对无 Pin 组的过滤是否是正式输出契约。若无 Pin 组几何需要保留，先明确它应如何在现有导出格式中表达及其电气语义；不要直接把这些组标记为有效网络。

## 验证范围与限制

已使用保存的 Component/Pin 预测重建 13 个样本的 Wire→Graph→Topology→NetLine/assemble 链路，没有重新运行 Component/Pin 模型。连接结果只覆盖精确匹配且无歧义的 Pin；几何距离不证明电气连接。当前 checkout 不含 `reference_segment_edge_ranges`，因此没有逐 GT 边到源 Graph 边的可靠 provenance。所有结果限于 13 个开发样本，不是官方分数，也不能证明对未知图纸的泛化。

配套详细阶段报告与逐样本结果：
- `experiments/v9_connection_stage_loss_detailed_20261027_210100/report.md`
- `experiments/v9_connection_stage_loss_detailed_20261027_210100/results.json`（本地完整逐配对明细；提交时不纳入大体积结果 JSON）
