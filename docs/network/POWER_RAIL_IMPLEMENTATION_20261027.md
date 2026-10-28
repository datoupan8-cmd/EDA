# 电源端子接入实现与工程验证

## 结果与状态

电源端子接入已完成，标准团队入口增加 `--power-terminals`，选择 `v3_power_terminals_v1`。默认 v3 保留，待完整固定对照后决定默认采用。本报告是工程正确性验证，不是评分提升或官网成绩报告。

| 案例 | 新增 GND 引用 | 新增供电引用 | 未接入情况 |
|---|---:|---:|---|
| 0001 | 22 | 15 | 7 个符号没有符合当前规则的导线证据 |
| 0002 | 4 | 0 | 本图预测无供电符号 |
| 0040 | 2 | 1 | 4 个无导线证据，2 个供电 value 不明 |

0001、0002 使用本轮当前配置生成的冻结 Scene 和掩膜重放；0040 从原图经过完整标准团队入口运行，最终命令使用新开关。每个新增端子都已写入 `pins`，并真实出现在 `nets[*].hyperGraph` 中，严格格式校验通过。

0001 按交接包评分器归一化后得到 GND、3.3V、1.1V 三条电源轨，包含顶点数分别为 23、14、4。该统计只读取预测，无 GT。原始网络 65→67，归一化逻辑网络 65→33；物理分组不合并，变化来自新增成员使已有分支获得轨身份。

## 实现

1. 复用 `LegacyHypergraph` 已生成的分组、路径和原始方向吸附结果。
2. 对 `gnd`、`v`，优先将有原吸附证据的隐藏端子提升为可导出的单端子；保留连接点坐标。
3. 缺少原端子时，只在符号边界附近寻找已有骨架的真实端点，并检查距离、方向和多个候选组的歧义。半径沿用原分网的线宽自适应半径，不扩大找线范围。
4. 确定后使用本地 schema 单端子编号，加入该路径既有网络组。此编号不冒充 OCR 观察到的物理脚号。
5. 已有可导出电源引脚保持原样；供电 value 不明不新增公共轨引用。组件 value、Name、key、bbox 不改。

适配层在 `pcb/topology_stage/power_terminals.py`，标准运行封装在 `power_v1.py`。没有修改公共 Schema、Submission、NetworkPlan 契约、Wire、legacy_v2 分网或线段实现。全局轨归并由评分器执行，不生成跨分支的虚构线段。

## 验证证据

- 完整数据无关回归：394 项通过。
- 集成范围检查：相对 `afc74d8` 通过，无越界文件；未放宽检查工具。
- 0001/0002：基线逐字段精确重放；组件属性、普通元件所有引脚、已有组 ID/路径归属/接触点、所有组线段几何均保持一致。
- 0040：完整 CLI 入口成功，新增的三个引用全部进入最终网络；GT 打开数 0。
- 针对性用例覆盖：无导线拒绝、未知电压拒绝、GND 类型身份、已有输出保护、同距离不同组拒绝、纯导线组接入、重复执行不重复添加。

本次尝试图像推理编号为 0001、0002、0003、0040；0003 在用户要求先完成接入后结束，未完成且不参与统计。本阶段不打开标注，不使用 0151-0200、10_GTcase 或官方反馈案例。150 例完整评分对照尚未完成，不能从本报告推断整体提分。

证据根目录：`runs/power_rails_20261027/`。

- `focused_replay/report.json`：0001/0002 输入哈希、引用数和不变量检查。
- `prediction_only_smoke/predictions_sealed.json`：独立重放候选哈希，GT_opens=0。
- `prediction_only_smoke/plan.json`：代码、配置源文件与模型 SHA-256。
- `full_cli_power0040/report.json`、`full_cli_smoke_summary.json`：实际 CLI 配置及结果哈希。
- `team_tests_final.log`、`scope_final.json`：回归与范围检查。
- `inference_environment.json`：实际推理使用的 Python 与附加依赖路径。

## 运行

在项目根目录、既有可用运行环境中执行：

```powershell
python tools/run_team_pipeline.py --dataset "C:/Users/pps/Desktop/EDA0.5/赛题六公开数据集/赛题六公开数据集/200_train_cases" --cases 0001 --power-terminals --save-debug --output runs/power_example_0001 --device cuda --ocr-device cuda
```

不传 `--evaluate` 不打开 GT；每次使用新的输出目录。`--power-terminals` 要求原 topology 为 v3，只切换适配层。其余选择固定为：Text `recovery_rotation_vertical_v1`、Component `rotation_vertical_n1_n2_v1`、Pin `l39/l39`、Wire `v3_frame_guard`、NetHypergraph/NetLine `legacy_v2/legacy_v2`。

交接包中的独立评分器已做路径移植，源码功能定义和 assignment 与原包一致；原始/移植哈希见 `peer2055_provenance.json`。标准入口可显式选择 `--evaluator pin_protocol2055`、`legacy_v2` 或 PDF 本地解释；默认 pin_protocol2055。这些均不是官方评测源码。本轮没有执行评分，不按分数挑选口径。

## 版本保留

- 基线：`afc74d8ae6bae461493a4f5b05d9a5976f1f24c9`。
- 请求/方案提交：`81bb547fbe232869d22c0d93e26b7590321a6d01`。
- 实现提交：`87def450241788c859eee2ef467a929440572b0d`。
- 回退标签：`backup/20261027-101643-before-power-rails`。
- 聚焦接入前另保留 `backup/20261027-103639-power-before-focused-integration` 及 `runs/power_rails_20261027/before_focused_integration/` 中的未提交差异和原文件。
- 分支：`feature/pin-l45-validated-20261012`，未推送。

无新增、替换或训练模型；选定 YOLO11m SHA-256 为 `38555d3e6b4617429a51e1e5afdfd982e5c58a0c67ba334df12b23fbaec01685`，PARSeq schematic120 为 `f42061a590476675269ec0ded0d4027261d6ff712ed58615f8c4ba5fc8d55417`，L39 点位权重为 `a488d0dc4897440ab7c0241c9044e1f260d02f8e2151ad80e39bcb121b4eec69`。其余模型实际哈希在上述 plan 中检查并保留。Windows 运行沿用既有 Python 3.12.14、Torch 2.14.0+cu132 及保留的 OCR 附加依赖；未重建或发布容器。
