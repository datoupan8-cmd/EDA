# 最新Component与Pin的团队整合结果

## 1. 目标与保留来源

本次仅完成架构与协作整合，没有训练模型、调阈值或优化算法。独立整合分支为 `integration/component-pin`，其合并父版本同时保留双方的完整历史：

| 来源 | 固定版本 | 保留内容 |
|---|---|---|
| 同学 `feature/modular-baseline` | `d648403438bb54811176f6bf1f340e09df1ce1e6` | 最新Component、文字理解、诊断与相关测试 |
| 我们 `feature/pin-l39-l40` | `af8bed7ed48562a137efb1ed2c38faeac3aef14b` | 最新已验证L39 Pin组合、原Wire/Topology及模型资产 |

整合前再次核对远程来源，两分支均未有更新。原始分支和原本地工程未被覆盖；原工程HEAD仍为af8bed7，原有 `pcb/pin_semantics.py` 的未提交修改仍保留。

## 2. 默认可运行组合

统一入口：`tools/run_team_pipeline.py`，统一清单：`configs/team.json`。

```text
Image -> Text/current -> Component/peer_latest
      -> Pin Localization/l39 -> Pin Semantics/l39
      -> Wire/v3_frame_guard -> Topology/v3 -> 原Submission JSON
```

L39继续采用原来的L1定位、L3完整词读取、E6角色处理、L36保序联合关联和L37官方公开库约束。所有原权重与参数保持原样。F框线隔离只是从旧实验函数封装成标准Wire版本。

同学的Pin V4另外保留为 `peer_v4`，我们的历史Pin `v4` 也继续保留。这两个版本没有相互覆盖。`configs/team_peer_pin.json` 可切换同学的Pin方案；`configs/team_l39_reference.json` 可切回原Component与L39的对照组合；`configs/current.json` 保持原样，历史入口仍可运行。

## 3. 为什么这次能够分别上传

过去公共Registry和总配置都列着具体版本，两个人新增算法时仍会修改相同文件；部分实现也在跨模块的实验脚本中。

现在每个负责人有独立的实际实现目录、Registry和配置片段：

| 负责人 | 实际算法 | 版本注册 | 默认选择 |
|---|---|---|---|
| Component/Text | `pcb/component/`、`pcb/text/` | 两个目录各自registry.py | `configs/component/current.json` |
| Pin | `pcb/pin/` | `pcb/pin/registry.py` | `configs/pin/current.json` |
| Wire/Topology | `pcb/wire_stage/`、`pcb/topology_stage/` | 两个目录各自registry.py | `configs/wire_topology/current.json` |

公共Pipeline只调用契约接口，总Registry只收集模块注册，总配置只引用片段。配置加载器拒绝一个负责人选择其他模块的版本。旧根目录文件保留兼容导入，不是日后算法修改的主要位置。

模块范围检查和GitHub自动测试会提示跨模块修改，但不能代替代码评审。各自上传自己的工作分支，再通过PR合入整合分支。GitHub不会自动拼接不同分支；大家最后下载整合分支。

## 4. 行为与模型验证

- Python语法检查通过。
- **204项自动测试全部通过**：原Component/Pin测试、Stage契约、旧新接口桥接、版本注册、配置隔离、数据边界和新旧组合一致性。
- **35个迁移文件、190个函数/类**通过AST主体一致性比较，比较时仅排除导入语句。资源位置与导入正确性另外由真实运行验证。完整迁移表见 `team_migration_manifest.json`。
- YOLO权重、Pin权重、官方公开库SHA256通过检查；EasyOCR与RapidOCR模型均可用。本轮没有新增模型资产。
- 对开发图片 **0001、0014、0087**，原Component＋新标准L39输出与先前保存的L39结果逐字段一致，包括Component、Pin和网络。
- 同学最新Component＋L39的默认组合也完成 **3/3** 同样本推理；这3张图上prediction逐字段一致，全部JSON合法。
- 从工程外的工作目录调用统一入口成功，不依赖原电脑的当前目录或原工程绝对路径。
- 另从Git待上传内容导出独立副本，未复制原工程的未提交文件或运行缓存；该副本的204项测试再次通过，全部必要模型和公开库检查通过。这验证了交付文件的完整性。

这3例的本地非官方宏平均F1为：Component 0.412168、Pin 0.397945、NetHypergraph 0.424732、NetLine 0.067598、PinPair 0.174737，总诊断分37.855275；前后相同。**它不是150例结果，也不是官方成绩。** 本轮没有再次运行150例，没有读取0151～0200、Golden、10GT或QuickTest，也没有上传官方评测平台。

可复核证据保存在 `team_validation.json`；个人运行的详细prediction和日志留在本地runs中，不作为大量重复文件上传。

## 5. 运行与协作边界

代码、模型和公开库均从工程相对路径加载；数据集由每位同学自行准备并通过命令行指定。完整下载应使用Git＋Git LFS，网页ZIP可能只包含模型指针。

本机验证环境为Windows、Python3.14、CUDA；Python3.12是文档建议环境及GitHub自动检查环境。不同系统、依赖版本、CPU/GPU速度和浮点结果仍可能有差异，本次没有声称跨设备逐字节一致。CI通过与否应查看GitHub实际运行结果，不能把本地测试当作云端测试。

源码不互相覆盖，不代表评分不会相互影响：Component改变框或文字时，Pin接收的输入也会改变。因此每个人除了自己的专项检查，仍应执行完整小样本与同条件整体A/B。

L39中旧评分辅助仍有串行临时状态，原行为原样保留；本轮没有改为并行多线程推理。公共schema、官方输出字段、坐标语义、Submission和评测算法没有修改。

## 6. 接下来各自开发

从整合分支创建 `feature/component-姓名-编号`、`feature/pin-姓名-编号` 或 `feature/wire-姓名-编号`。新增版本只修改本人算法目录、Registry与配置，测试放本人tests目录。公共接口变更应单独协调。

详见 [完整运行与上传说明](TEAM_RUN_GUIDE.md) 和 [模块契约](TEAM_MODULE_CONTRACTS.md)。这些说明包含下载、环境安装、单图运行、小样本/30例/150例评测、前后比较、修改位置、提交与合入步骤。
