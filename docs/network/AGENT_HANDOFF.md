# 给网络负责人 Agent 的交接说明

请先完整阅读本文及 [网络使用说明](README.md)，再读自己负责的目录。当前这是已完成验证的结构拆分；不要重新复制旧 Topology 或修改 Pin/Component。

## 起点与当前状态

- 仓库：https://github.com/datoupan8-cmd/EDA
- 统一下载/PR目标分支：`integration/component-pin`。
- 本次拆分工作分支：`feature/network-metrics-split`。
- 默认保留 Component `peer_latest`（同学d648403）、Pin定位/语义 `l39`、Wire `v3_frame_guard`、父Topology `v3`。
- 网络内部两块默认均为 `legacy_v2`，本轮不包含算法提分、模型变化、Docker构建或官方提交。
- 原负责人未提交的 Pin 研究成果未带入这次网络提交；不要擅自接入 L42/L43 或修改正式 Pin 选择。

## 你只维护哪些文件

按用户实际分工选择一个 owner；没有明确分工时先确认，不自行把其他模块纳入范围。

| owner | 允许修改 |
|---|---|
| net_hypergraph | pcb/net_hypergraph/、configs/net_hypergraph/、tests/net_hypergraph/、models/net_hypergraph/、docs/network/hypergraph/ |
| net_lines | pcb/net_lines/、configs/net_lines/、tests/net_lines/、models/net_lines/、docs/network/lines/ |
| wire | pcb/wire_stage/、tests/wire/、models/wire/、docs/network/wire/；configs/wire_topology/current.json仅wire字段 |
| network | 上述三项的并集，适用于同一负责人确实负责全部网络工作 |

**禁止日常修改：** `pcb/component/`、`pcb/text/`、`pcb/pin/`及对应配置；`pcb/core/`、`pcb/schema.py`、`pcb/submission.py`、坐标转换、评测器；`pcb/network_shared/`、`pcb/topology_stage/`、`tests/topology/`及冻结基准；公共模型与其他负责人的文件。

需要新接口时，先提交独立的集成需求，描述当前契约为何不足；不要借 `--owner integration` 绕过范围检查。

## 修改方式

1. 从最新整合分支创建自己的工作分支：`feature/hypergraph-名字-序号`、`feature/net-lines-名字-序号`、`feature/wire-名字-序号`或`feature/network-名字-序号`。
2. 保留自己的 `legacy_v2` 对照，新增版本文件，只在本目录 registry.py 注册，且只改本模块 current.json 的 version。
3. NetHypergraph 输入是只读 SceneView/WireGraph，输出 LogicalResult(NetworkPlan, diagnostics)；只能引用现有Pin和路径。
4. NetLine 输入是 WireGraph/NetworkPlan，输出 GeometryResult；group_id集合必须与plan完全一致，不改网络成员，不回写Pin。
5. 模块内部坐标均为原图左上角像素x/y；唯独骨架paths使用row/column。导出器统一处理最终坐标，不自己翻转Y。
6. 新版本使用团队入口运行。旧build_topology_v2/v3兼容函数默认固定旧算法，不用于启用新选择。
7. 本次Pipeline启动时固定两块版本，report.json记录network_versions。一次运行期间不改配置。

## 每轮实验纪律

先写：本轮改什么、已有证据是什么、预期改善哪项、成功/失败怎么判定。每轮只改一个主要变量，冻结另外两块以及Pin/Component/OCR。不得读保留集，不按Check反馈反复挑阈值。

结构测试证明边界和兼容，不能替代得分实验。先小样本，再固定30例，确有收益再150例；算法采用以比赛综合收益为依据，完整记录Pin/Component是否变化及网络各项收益/成本，不因任何单例回退直接无限微调。失败先解释原因，再决定停下或改下一项。

NetHypergraph与NetLine是维护边界，**不是完全独立的评分口径**。网络匹配变化可能影响线段分数。Wire改变共同基础掩膜时也会同时影响两者。

## 必须执行的检查

在工程根目录（环境已准备好）执行；下面示例为NetLine：

```powershell
python tools/show_network_versions.py
python tools/check_network_change_scope.py --owner net_lines --base HEAD
python tools/check_network_change_scope.py --owner net_lines --base origin/integration/component-pin
python tools/run_team_tests.py
```

换其他分工时替换owner。范围检查包含本分支提交与未提交文件；发现跨模块变更即拆出，不修改检查工具来让它通过。不要更新冻结legacy测试期望来掩盖回退。

正常运行：

```powershell
python tools/run_team_pipeline.py --image "D:\EDA_Data\sample.png" --output runs\network_smoke_01 --device cuda
```

开发评分需要自行取得0001～0150开发集，再按团队运行说明使用 `--dataset`、`--cases`、`--evaluate`。不得读取0151～0200、Golden、10GT或QuickTest。

## 提交与汇报

仅add自己目录，检查暂存清单后commit/push自己的分支，PR目标为integration/component-pin。不要覆盖整合分支上的其他更新，不做force push。

交付时报告：改变的主要变量、两块网络版本、比较样本、各项分数前后值、收益覆盖范围、失败/回退原因、是否碰到公共接口。提交信息和PR描述围绕最终实现，不复制聊天历史。

本轮不操作Docker镜像或官方平台。需要比赛测试时由用户另外确认。
