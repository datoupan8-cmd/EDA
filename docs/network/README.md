# 网络两项的拆分、运行与修改说明

## NetLine 包整合（2026-10-27）

总程序已接入 `path_v1`、`path_collinear_v1` 和可选来源追踪文件，普通 V2/V3、供电端子 V1/V2 均支持。通过 `--net-lines-version` 单次选择，增加 `--save-net-lines-provenance` 可保存独立 sidecar；默认仍为 `legacy_v2`。具体命令、兼容性和本次验证见 [NetLine 整合说明](lines/README.md)。

## 当前默认：供电端子 V2（2026-10-27）

标准团队配置现在选择 `v3_power_terminals_v2`，无需额外传 `--power-terminals`。上游固定 N1/N2＋L39，两个网络插槽仍为 `legacy_v2`。完整 150 例的分数、误连接代价和版本证据见 [CURRENT150_POWER_RESULTS_20261027.md](CURRENT150_POWER_RESULTS_20261027.md)。当前工作分支固定为 `feature/pin-l45-validated-20261012`。

复现原版时显式传 `--config docs/network/CURRENT150_BASELINE_CONFIG_20261027.json`；复现 V1 时加 `--power-terminals --power-version v1`。原版历史 parity 工具也使用该保留配置。下面的拆分说明和分支示例为历史记录，当前分支与配置以本节及 AGENTS.md 为准。

## 历史拆分说明

本轮是结构拆分，不是新的提分实验。保留正式 Component/Text、L39 Pin、Wire V3/F 和全部默认算法参数，把原 Topology 的内部工作分为两个可独立替换的模块。

代码先保存在 `feature/network-metrics-split`，再快进到统一分支 `integration/component-pin`。日常完整运行仍使用 `tools/run_team_pipeline.py`。不需要新建一套工程，也不需要本轮重建镜像。

## 1. 拆开的是哪些工作

```text
原图 → 元件与引脚（原样保留）
       ↓
Wire：提取导线掩膜（原样保留）
       ↓
共享基础线图：骨架、路径、固定的吸附参考线段
       ↓
NetHypergraph：交点连通判断、引脚吸附、网络成员归属
       ↓  NetworkPlan：成员 + 路径ID
NetLine：按照指定的路径，生成最终线段坐标
       ↓
共享组装器 → 原 result.json：nets[*].hyperGraph + edges
```

例如 U1.1 和 R1.2 应该属于同一网络，由 NetHypergraph 决定；连接它们的折线需要几个线段、端点在哪里，由 NetLine 决定。NetLine 不能重新给引脚分网，也不能修改 Pin 的位置、编号或脚名。

Wire 是两块共同的上游：修改 Wire 的墨迹掩膜仍可能同时影响两项指标。基础图的骨架化、路径编号以及参考线段由集成层维护，普通算法迭代不修改。

**模块拆开不代表评分完全独立。** 当前本地评测先匹配网络，再在匹配网络中比较线段；分网变了，NetLineF1 可能随之变化。官方评分细节仍以官方反馈为准。本轮保证维护边界和输入输出约束，不承诺两个分数互不影响。

## 2. 修改范围

| 工作 | 算法目录 | 注册 | 版本选择 | 测试 |
|---|---|---|---|---|
| Wire 掩膜 | pcb/wire_stage/ | 该目录 registry.py | configs/wire_topology/current.json **仅 wire 字段** | tests/wire/ |
| NetHypergraph 成员归属 | pcb/net_hypergraph/ | 该目录 registry.py | configs/net_hypergraph/current.json | tests/net_hypergraph/ |
| NetLine 线段表达 | pcb/net_lines/ | 该目录 registry.py | configs/net_lines/current.json | tests/net_lines/ |
| 集成维护 | pcb/network_shared/、pcb/topology_stage/ | 兼容与组装 | 公共策略如需变化单独协商 | tests/topology/ |

如果同一位同学负责全部网络工作，可以在一个网络分支修改前面三行；每次实验仍只改一个主要变量。只负责一项时，不修改另外两项。

Component/Text、Pin、公共 `pcb/core/`、`pcb/schema.py`、Submission、坐标转换、评测器以及他人模型不属于网络负责人的日常修改范围。新文件与文档也需放在自己目录中。

两个新的 current.json 默认都是：

```json
{"version": "legacy_v2"}
```

文件只能包含 version。名字是各自注册的实现名称，不能把 Pin 或另一网络模块的选择夹带进来。父级 Topology 保持 v3；不要通过改成 v1 绕过拆分。历史 v1 保留旧实现，不使用新插槽。

## 3. 下载和运行

在自己的 PowerShell 中执行；目录可自行选择：

```powershell
git lfs install
git clone --branch integration/component-pin https://github.com/datoupan8-cmd/EDA.git C:\EDA
Set-Location C:\EDA
git lfs pull
```

已有独立环境就继续使用；首次安装依赖和模型的详细操作见 [团队运行说明](../TEAM_RUN_GUIDE.md)。不要仅复制两个模块文件，运行需要完整工程与 LFS 模型。

```powershell
python tools/show_network_versions.py
python tools/check_team_environment.py
python tools/run_team_tests.py
python tools/run_team_pipeline.py --image "D:\EDA_Data\sample.png" --output runs\network_smoke_01 --device cuda
```

没有 GPU 时显式选择 `--device cpu`。单图命令不读取 GT。输出仍是 result.json、diagnostics.json、report.json；完整运行的启动日志和 report.json 额外记录本次固定的两个 network_versions，官方 result.json 结构不变。

一次 Pipeline 构造时固定两个版本；不要在一次运行中修改 current.json。每次比较使用新输出目录和同一设备。

## 4. 开发一个自己的版本

先从干净的整合分支拉取最新代码，再建自己的分支：

```powershell
git switch integration/component-pin
git pull --ff-only
git switch -c feature/net-lines-yourname-01
```

其他分支命名：成员归属使用 `feature/hypergraph-yourname-01`；Wire 使用 `feature/wire-yourname-01`；一人负责多块使用 `feature/network-yourname-01`。这些前缀对应 GitHub 自动范围检查。

以 NetLine 为例：

1. 新增 `pcb/net_lines/my_v1.py`，实现 `run(graph, plan) -> GeometryResult`。
2. 在 `pcb/net_lines/registry.py` 注册唯一名称，例如 `lines_my_v1`。
3. 只将 `configs/net_lines/current.json` 的 version 改为该名称。
4. 添加 `tests/net_lines/` 测试与 `docs/network/lines/` 实验记录。

```python
# 在自己的 register_stages(registry) 内添加
from .my_v1 import MyLines
registry.register("net_lines", "lines_my_v1", MyLines)
```

NetHypergraph 对应注册 kind 为 `net_hypergraph`，接口为
`run(scene_view, graph, *, mode, merge_labels) -> LogicalResult`。
无需改公共注册表、team.json 或 Pin/Component 文件。

**保留 legacy_v2 原样作为固定对照。** 旧版兼容函数 `build_topology_v2/v3` 不传新 backend 时使用冻结的旧算法；新版本的正常入口是团队 Pipeline 的 TopologyStageV2/V3，它们读取两份独立选择。不要改旧兼容函数来接新算法。

## 5. 接口需要理解的细节

具体类型见 `pcb/network_shared/contracts.py`；这是网络私有视图，不是另一套公共 Scene/Pin。

| 对象 | 含义与限制 |
|---|---|
| SceneView | 原元件、引脚、文字的只读快照；不含可修改的公共 Scene |
| WireGraph.paths | 骨架路径，像素坐标为 **row/column** |
| WireGraph.reference_segments | 固定旧算法的吸附参考，坐标为原图 **x/y**；不能被最终线段简化反向修改 |
| NetGroup | group_id、pins、path_ids、contact_points；均为不可变元组 |
| NetworkPlan | 有序 NetGroup 集合；同一个现有 Pin ref、路径ID不可重复归属 |
| LogicalResult | plan 加本模块诊断；不得生成新的 Pin 身份 |
| GroupLines | 使用 plan 中的 group_id，返回原图 x/y 线段端点 |
| GeometryResult | 每个逻辑 group 恰好返回一个 GroupLines，不得增加或删除 group |

Pin ref 沿用 `component.key + "." + pin.number`。group_id 是内部 ID，不是导出 net 名。公共组装器按原顺序生成 Net，过滤没有 Pin 的网络并交给原导出器处理。

NumPy 掩膜、厚度图和骨架是私有只读输入。需要临时可写数组时自行 copy，不要解除只读标志。需要改变网络成员时应修改 NetHypergraph；需要改变共享图构造时提交独立集成需求，不从 NetLine 反向修改它。

历史交点判断、短桥、标签合并和吸附规则保存在 `pcb/net_hypergraph/legacy_v2.py`；历史线段与直接接触点的输出保存在 `pcb/net_lines/legacy_v2.py`。本次参数与逻辑未调优。

## 6. 检查和提交

先检查当前工作区，再检查整个分支。新的范围工具会同时检查提交、暂存、未暂存和非忽略的新文件：

```powershell
python tools/check_network_change_scope.py --owner net_lines --base HEAD
python tools/check_network_change_scope.py --owner net_lines --base origin/integration/component-pin
python tools/run_team_tests.py
```

成员归属改 owner 为 `net_hypergraph`，Wire 改为 `wire`，负责前面三块改为 `network`。不要用 `integration` 绕过自己范围；该模式只用于已明确授权的公共拆分。

老 `tools/check_module_scope.py --owner wire_topology` 不再作为细分网络分支的验收工具；Pin/Component 仍使用原工具。Git 不在 PATH 时，给新工具补 `--git "本机git.exe绝对路径"`。

算法实验的步骤仍是：先写假设和成功/失败标准 → 固定输入与配置 → 改一个主要变量 → 小样本筛选 → 固定30例 → 有实际收益再做150例。完整评分通过团队入口加 `--evaluate`；不要混用图像重放的一致性结果和正确率。

仅允许开发集0001～0150。0151～0200、Golden、10GT、QuickTest继续禁止用于开发；本次35例结构验证也未读GT。

只提交自己的文件，以 NetLine 为例：

```powershell
git add pcb/net_lines configs/net_lines tests/net_lines
# 如果新增实验文档，另行 add docs/network/lines 下的具体文件
git diff --cached --stat
git commit -m "feat(net-lines): describe the actual change"
git push -u origin feature/net-lines-yourname-01
```

创建 PR 时目标仍是 `integration/component-pin`。新网络分支的 GitHub 自动检查使用对应 owner。不要 `git add .`，不要上传数据、运行目录或别人的改动。模型变化另行记录并使用 LFS。

## 7. 本次验证与之后的协作

拆分验证见 [验证记录](VALIDATION_20261012.md)。给同学 agent 的明确任务边界见 [交接说明](AGENT_HANDOFF.md)。

本轮一次性修改了网络兼容层、网络测试及说明工具；普通算法迭代不需要再改这些公共文件。两个工作分支通过 PR 合入同一整合分支后，大家下载统一版。GitHub不会自动拼接尚未合入的分支。

修改线段模块不会通过共享可变 Scene 改坏 Pin；但改变 Wire 或网络成员后，下游评分可能变化，仍需完整 A/B。遇到新的接口需求先提独立改动，不在自己的模块分支顺手重写 Pin 或导出格式。
