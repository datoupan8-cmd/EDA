# NetLine 整合版使用说明（2026-10-27）

本模块已经接入总程序 `tools/run_team_pipeline.py`，支持普通 V2/V3 和当前正式供电端子 V2；历史供电端子 V1 也可用。保留现有 N1/N2 元件、L39 引脚、Wire 和供电端子逻辑，未导入压缩包内其他模块的旧副本。

## 版本选择

| 版本 | 行为 | 定位 |
|---|---|---|
| `legacy_v2` | 保留原参考线段及去重接触点 | 当前默认和回退对照 |
| `path_v1` | 输出每条原始路径的相邻点边 | 独立候选，通常产生较多碎段 |
| `path_collinear_v1` | 仅合并单条路径内严格共线且同向的连续边 | 独立候选，不跨路径或网络合并 |

不传新参数时，继续读取 `configs/net_lines/current.json`，当前仍为 `legacy_v2`。下面的参数只影响本次运行，不写配置，不改变 NetHypergraph 选择或上游模型：

```powershell
python tools/run_team_pipeline.py --list-stages --net-lines-version path_collinear_v1
python tools/run_team_pipeline.py --image "D:/PCB/example.png" --output runs/netline_candidate_01 --net-lines-version path_collinear_v1
```

输入图片来自官方数据集时，仅允许 `200_train_cases/0001..0150`，总程序原有白名单继续执行。每次使用新输出目录，不能覆盖旧结果。未知版本和不支持独立网络模块的 V1 Topology 会在模型加载前报错。

## 来源追踪文件

增加 `--save-net-lines-provenance`，可以在每张图片的 `result.json` 旁保存 `net_lines_provenance.json`：

```powershell
python tools/run_team_pipeline.py --image "D:/PCB/example.png" --output runs/netline_trace_01 --net-lines-version path_collinear_v1 --save-net-lines-provenance
```

sidecar 包含本次版本、`NetworkPlan`、线段几何和来源记录，按 `group_id`、段序号、path ID 和原始边索引对应。坐标明确为原图左上角原点的 x/y；官方 `result.json` 继续由原导出器转换坐标，字段结构不增加。`report.json` 记录实际版本及是否保存来源追踪。

三种版本都支持追踪。旧版参考线段的源边区间由共享图构造提供；重复坐标无法唯一匹配、数值精度丢失或来源区间不合法时保留 `unknown`，不按距离猜测。sidecar 描述组、路径和线段来源，不证明单个 Pin 与单条 edge 的 attachment。

sidecar 仅在显式启用时生成；每张图片替换上一次调用记录，失败调用不能复用旧记录。它不写入正式提交 JSON，也不会修改输入路径或引脚。

## 验证与保留

- 上游压缩包原有 NetLine 测试：62/62 通过，原缺失的来源索引依赖已补齐。
- 整合后的团队无数据测试：478/478 通过，包括 CLI 文件输出与供电端子兼容测试。
- 与整合前比较的 64 组合成场景：完整 Scene、诊断、骨架、路径和参考线段相同。
- 另有 1000 组随机路径比较：新索引构造与旧 `simplify()` 的几何完全相同。
- 没有运行官方数据推理或 GT 评分，没有读取保留集，不声明真实准确率提升。

基线提交为 `ff562391aa8976291afdec516e0fba5661464253`；回退标签为 `backup/20261027-190719-before-netline-integration`。源码、配置、模型哈希和环境版本的备份与测试日志保存在 `E:/EDA/deliveries/backups/netline-integration-20261027-190719/`。代码验证后创建可追溯提交，实际提交号记录在该目录的 `integration-completion.json`。

原包交接说明保存在 [UPSTREAM_HANDOFF_20261024.md](UPSTREAM_HANDOFF_20261024.md)，仅代表原包 checkpoint 状态；本页描述本次整合后的状态。原包的实验报告未作为本次验证证据导入。

本次保留 `legacy_v2` 为默认。两个候选的真实得分仍需在冻结上游输入的开发集上独立比较；减少碎段或保持几何覆盖不能替代 NetLine 评分对照。
