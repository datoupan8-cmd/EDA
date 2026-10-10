# 团队整合版：下载、运行、验证与各自开发说明

这份说明中的命令都在 **PowerShell** 中执行。代码在每位同学自己的电脑上运行，GitHub负责保存与交换源码，不负责执行模型。

## 1. 这次保留了什么

统一整合分支：`integration/component-pin`。

默认组合为：同学最新Component/Text（原分支d648403）＋我们最新已验证L39 Pin组合（原分支af8bed7的L1/L3/E6/L36/L37）＋原Wire V3/F框线隔离＋Topology V3＋原Submission。

同学的Pin V4另外保留为 `peer_v4`；它不会覆盖我们的Pin V4，更不会替代默认L39。旧版本继续可以选择。原始两个分支、你原来的本地工程及未提交修改保持保留。

**运行统一整合版请使用 `tools/run_team_pipeline.py`。** 历史 `main.py` 和旧实验工具保留用于追溯；不加选择就运行历史入口，不能代表当前最新版组合。

## 2. 第一次在另一台电脑上下载

需要：Git、Git LFS、Python 3.11以上；建议使用独立Python环境。有NVIDIA显卡可使用对应CUDA版PyTorch，没有显卡可显式选CPU，速度会慢。依赖安装不等于已证明不同系统的结果逐字节相同。

下面的 `C:\EDA` 是**同学自己选择的目录**，不是必须与作者电脑相同：

```powershell
git lfs install
git clone --branch integration/component-pin https://github.com/datoupan8-cmd/EDA.git C:\EDA
Set-Location C:\EDA
git lfs pull
```

不要只下载几份Python文件。完整测试需要完整工程、模型、公开元件库、配置及下游模块。GitHub网页的ZIP未必带有真正的LFS模型内容，因此优先使用上述下载方式。

建立环境（已经有可用环境时直接激活自己的环境，不需要重复创建）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后续命令统一使用这个解释器，不依赖PowerShell脚本执行策略：

```powershell
$Python = ".\.venv\Scripts\python.exe"
& $Python tools/check_team_environment.py
& $Python tools/run_team_pipeline.py --list-stages
& $Python tools/run_team_tests.py
```

若使用已有Conda环境，激活后把 `$Python = "python"` 即可。若检查显示CUDA不可用，请按自己的显卡驱动安装匹配的PyTorch，或先使用 `--device cpu`。本说明不要求安装作者的Conda绝对路径。

环境检查会核对YOLO、Pin模型、公开元件库和OCR模型；若提示模型缺失或错误，先执行 `git lfs pull`，不要随便换一个best.pt。

## 3. 模型在哪里

| 内容 | 工程内路径 | 用途 |
|---|---|---|
| 元件YOLO | `models/component_yolo11n_continue_v2_best.pt` | 原有元件视觉检测 |
| EasyOCR | `models/easyocr/` | 原有全图分块文字读取 |
| RapidOCR | 原有vendor或安装包模型 | 混合OCR与局部完整词读取 |
| Pin L1权重 | `reports/pin_learned_locator_l1/best.pt` | 原有box引脚定位 |
| 官方公开元件库 | `assets/cpntLibrary.json` | 原有L37语义约束 |

本轮没有训练、更换模型或调整阈值。默认模型与库均从工程内读取，数据集由每个人在命令行指定。

## 4. 单张图片跑完整流程

```powershell
& $Python tools/run_team_pipeline.py --image "D:\EDA_Data\sample.png" --output runs\single_01 --device cuda
```

没有CUDA时：

```powershell
& $Python tools/run_team_pipeline.py --image "D:\EDA_Data\sample.png" --output runs\single_cpu_01 --device cpu
```

`auto`会选择本机可用设备并打印实际选择。CPU/CUDA执行环境不同，不能直接把两者差异算成算法收益。

输出是 `runs/<输出目录>/<图片名>/result.json`、diagnostics.json，以及整次运行的report.json。默认只读取PNG及模型/缓存，不读取target。加 `--save-debug` 可保存内部Scene和wire mask。

输出目录必须是新的；重复测试换个名字，例如 single_02，避免覆盖上次证据。

## 5. 在自己电脑上测试正确率

每个人都需自行准备**最新版官方公开开发数据**，目录包含 `200_train_cases/0001`～`0150`。数据不上传GitHub，不会随源码下载。

```powershell
$DatasetRoot = "D:\EDA_Data\200_train_cases"
& $Python tools/run_team_pipeline.py --dataset $DatasetRoot --cases 0001,0014,0087 --output runs\smoke_01 --device cuda --evaluate
```

工具先完成图片推理并保存prediction，之后独立读取对应target评分。不会把GT传给检测、OCR、Pin或网络算法。

固定30例（历史开发Check，**不是未使用过的封存验证集**）：

```powershell
$Cases30 = ((5..150 | Where-Object { $_ % 5 -eq 0 }) | ForEach-Object { "{0:D4}" -f $_ }) -join ","
& $Python tools/run_team_pipeline.py --dataset $DatasetRoot --cases $Cases30 --output runs\check30_before --device cuda --evaluate
```

全150例：

```powershell
& $Python tools/run_team_pipeline.py --dataset $DatasetRoot --output runs\dev150_01 --device cuda --evaluate
```

不加 `--evaluate` 就只推理。**严禁在开发中读取0151～0200、Golden、10GT或QuickTest。** 工具拒绝这些输入；不得通过改名来绕过保护。

查看 `runs/<目录>/report.json` 中的 `aggregate.overall`：FinalScore、各项macro_f1、TP/Pred/GT计数。所有这些都是本地非官方诊断结果。官方环境上传与评测另行确认，本工具不会上传平台。

## 6. 修改前后怎样比较

1. 未改算法前，固定配置、设备和case名单，输出到 before。
2. 只改自己的一个主要变量，再跑同一名单到 after。
3. 使用下面的比较工具同时看Component、Pin、网络指标；不能混用macro和micro。

```powershell
& $Python tools/compare_team_runs.py --before runs\check30_before\report.json --after runs\check30_after\report.json --output runs\comparison_01.md
```

源码改动后不要继续消费旧的“最终prediction”当新输出。OCR/YOLO和局部词缓存是按图像、模型、参数等内容签名复用，正常开发无需主动清空。如果你改变了未进入缓存签名的实现逻辑，使用新的 `--cache-dir runs\cache_mychange_01`，避免把旧读数当新算法结果。

## 7. 为什么现在文件名不同

以前检测、文字关联、实验脚本散在根目录；两个人也都使用过Pin V4这个名字。整合后将实际实现放进负责人目录，旧名字大多只是兼容导入。

| 以前看到的文件 | 今后主要修改位置 |
|---|---|
| component_detection_v4.py | `pcb/component/latest/detection.py` |
| component_text_assignment.py | `pcb/component/latest/text_assignment.py` |
| 元件文字分类 | `pcb/component/latest/text_roles.py` |
| component_detector_yolo.py | `pcb/component/detector_yolo.py` |
| component_proposal_fusion.py | `pcb/component/proposal_fusion.py` |
| component_detection.py几何补充 | `pcb/component/geometry.py` |
| ocr_backends.py | `pcb/text/ocr_backends.py` |
| L1/L3/E6/L36/L37实验实现 | `pcb/pin/frozen_l39/` 对应文件 |
| wire.py / wire_v2.py / wire_v3.py | `pcb/wire_stage/v1.py` / v2.py / v3.py |
| topology.py / topology_v2.py / topology_v3.py | `pcb/topology_stage/v1.py` / v2.py / v3.py |

历史实验文件保留为冻结对照，不作为新的主线修改入口。完整迁移关系见 `docs/team_migration_manifest.json`。

## 8. Component负责人怎么改

先从新的整合分支创建自己的分支：

```powershell
git switch integration/component-pin
git pull --ff-only
git switch -c feature/component-yourname-01
```

主要修改 `pcb/component/`、`pcb/text/`；新测试放 `tests/component/`。需要新增版本时，在自己的 `pcb/component/registry.py` 注册唯一名字，并只更新 `configs/component/current.json`。不要修改Pin注册、Pin配置、公共Pipeline或schema。

```python
# pcb/component/registry.py 的 register_stages(registry) 中
from .my_v5 import ComponentStageV5
registry.register("component", "component_v5", ComponentStageV5)
```

然后将自己配置中的 `component` 设成 `component_v5`。其余字段按原配置保留。新Stage沿用ComponentStageOutput，不重定义Component类。

## 9. Pin负责人怎么改

创建 `feature/pin-yourname-01`，主要修改 `pcb/pin/`，新测试放 `tests/pin/`。当前有效实现位于：

- locator.py：L1定位；
- word_reader.py / terminal_reader.py：局部完整文字读取；
- contextual_roles.py / bbox_semantics.py / ordered_v6.py：原角色与关联基础；
- joint_alignment.py：L36联合保序关联；
- library_semantics.py：L37公开库约束；
- stages.py：Stage封装和运行资源。

建议新增自己的版本目录或Stage，保留 frozen_l39 作为对照。只在 `pcb/pin/registry.py` 注册新名字，再修改 `configs/pin/current.json`。不需要编辑Component或Wire内部代码。

新定位仍返回PinLocalizationOutput；新语义需要局部OCR时使用 `run_image(...)`，不在Context里隐藏图片或全部算法状态。字段/坐标语义见MODULE契约文档。

## 10. Wire/Topology负责人怎么改

创建 `feature/wire-yourname-01`，主要修改 `pcb/wire_stage/`、`pcb/topology_stage/`。新增版本只注册在这两个目录各自的registry，选择写入 `configs/wire_topology/current.json`。

F框线隔离现在是标准Wire版本 `v3_frame_guard` 的一部分。若未来新Wire已经包含同等处理，明确选择对应Wire版本，避免重复隔离；不要顺手改Pin的位置来补网络错误。

## 11. 改完怎么检查、上传

先检查自己的变更范围，以Pin为例：

```powershell
git status --short
& $Python tools/check_module_scope.py --owner pin --base HEAD
& $Python tools/run_team_tests.py
```

Component改 `--owner component`，导线网络改 `--owner wire_topology`。范围检查会指出是否动到了别人的模块或公共文件；公共变更需要单独协商，不能悄悄夹在算法更新中。

准备创建PR时，再检查本分支相对整合分支的已提交改动：`& $Python tools/check_module_scope.py --owner pin --base origin/integration/component-pin --branch`。它从两分支共同起点比较，避免把同学随后上传的更新误算成自己的跨模块改动。GitHub的自动检查也使用这个口径。

再做小样本/固定30例A/B；需要定版时做150例。同一台机器、同一数据版本、同一名单比较。允许按综合得分决定方案，不要求所有模块指标都单独上涨。

只上传自己负责的源码、配置、测试与必要小文档，以Pin为例：

```powershell
git add pcb/pin configs/pin tests/pin
git diff --cached --stat
git commit -m "feat(pin): describe the actual change"
git push -u origin feature/pin-yourname-01
```

不要 `git add .` 把数据、运行缓存或别人的修改一起上传。模型新增/更换属于另外的算法实验，必须明确记录，并使用Git LFS。

## 12. 他传他的、我传我的，最后怎么下载统一版

每个人上传**自己的工作分支**。在GitHub创建Pull Request，目标选择 `integration/component-pin`。整合负责人核对测试和整体收益后，将模块更新合入整合分支。

GitHub不会自动把两个工作分支拼起来；**大家下载和运行的是整合分支**。两个人改不同模块目录时，通常不产生源码冲突；公共接口、同一模型资产或同一模块文件仍需要协调。

统一版更新后，使用干净的整合分支同步：

```powershell
git switch integration/component-pin
git pull --ff-only
git lfs pull
& $Python tools/check_team_environment.py
```

自己分支有未提交修改时先保留在自己的分支，不要reset/clean。新一轮开发可从更新后的整合分支再建自己的分支。

## 13. 一句话操作顺序

下载完整整合分支 → 拉取LFS模型 → 准备自己的Python环境与开发数据 → 跑环境检查及自动测试 → 跑完整小样本 → 创建自己的模块分支 → 只改自己的算法/注册/配置 → 同条件A/B → 只提交自己文件 → PR合入整合分支 → 大家重新下载统一版。

“模块互不覆盖”是源码协作保证，不是成绩互不影响的保证。比如Component框变了，Pin收到的输入也会变，所以最终仍需整体测试。
