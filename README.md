# PCB Competition Solution V4.1 — Modular Baseline

这是 PCB 原理图自动解析比赛的可运行、可替换模块基线。当前默认流程保持 V4.1 已验证算法不变：

```text
PNG
├─ RapidOCR 全图 + EasyOCR 全分块 → Text
├─ YOLO tiled inference → Component proposals
├─ OCR role + 全局关联 → Component
├─ Pin Localization → Pin Semantics
├─ Wire Extraction → Topology
└─ strict validator → 官方契约 result.json
```

公共编排结构为：

```text
Text → Component → Pin Localization → Pin Semantics → Wire → Topology → Submission
```

`OFFICIAL_SCORE = FALSE`：仓库中的评测器是按公开规则实现的本地诊断工具，不是主办方正式 evaluator。

## 仓库边界

- 官方数据集不进入 Git；每位同学自行取得同一版本数据，并通过命令行传入路径。
- `runs/`、OCR cache、逐例预测和大体积评测明细不进入 Git。
- `.pt`、`.pth`、`.onnx` 和二进制运行时由 Git LFS 管理。
- `0151..0200` 与 `10_GTcase` 继续封存，开发命令只允许 `0001..0150`。
- 当前权重的来源、训练重叠和哈希见 `MODEL_WEIGHTS.md` 与 `models/SHA256SUMS.json`。

建议先建立 **Private GitHub repository**。官方数据、同组权重和部分第三方运行时的公开再分发权尚未完全确认；在权限确认前不要设为 Public。

## 第一次克隆

不要使用网页中的“Download ZIP”代替标准克隆。先安装 Git 与 Git LFS，然后执行：

```powershell
git lfs install
git clone https://github.com/你的组织/你的仓库.git
cd 你的仓库
git lfs pull
```

### 推荐：Conda 环境

`environment.yml` 固定 Python 3.14，这是当前 150 例 parity 基线使用的 Python 主版本。

```powershell
conda env create -f environment.yml
conda activate pcb-v4-1
python tools/repository_doctor.py
```

### 已有 Python / PyTorch 环境

```powershell
python -m pip install -r requirements.txt
python tools/repository_doctor.py
```

也可以使用一键脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -PythonExe python
```

`repository_doctor.py` 会检查依赖、模型是否只是 LFS 指针、模型 SHA256、Stage 注册和本机绝对路径。全部显示 `OK` 后再运行。

## 单张图片运行

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_one.ps1 `
  -Image "D:\pcb_data\example.png" `
  -OutputDir "runs\my_case" `
  -PythonExe python `
  -Device auto `
  -OcrDevice auto
```

输出：

```text
runs/my_case/result.json
runs/my_case/diagnostics.json
runs/my_case/*.png
```

等价的直接命令：

```powershell
python main.py `
  --image "D:\pcb_data\example.png" `
  --output "runs\my_case\result.json" `
  --orchestrator modular `
  --pipeline_config configs/current.json `
  --weights models/component_yolo11n_continue_v2_best.pt `
  --device auto `
  --ocr_backend hybrid `
  --ocr_device auto `
  --save_debug_images
```

所有项目内部路径都是相对于仓库根目录解析的；输入图片和官方数据集可位于任意磁盘。

## 在前 150 例验证准确率

数据路径必须直接指向官方 `200_train_cases` 文件夹：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/benchmark_dev150.ps1 `
  -DatasetRoot "D:\pcb_data\200_train_cases" `
  -Name "my_branch_dev150" `
  -PythonExe python `
  -Device auto `
  -OcrDevice auto
```

该命令依次完成：

1. Modular Pipeline 推理 0001～0150；
2. 总体本地评测；
3. Component 专项评测；
4. 150 例 strict submission validation。

本机生成的文件位于：

```text
runs/my_branch_dev150/
reports/my_branch_dev150_metrics.json
reports/my_branch_dev150_component_metrics.json
reports/my_branch_dev150_validation.json
```

这些文件默认不提交。当前不变基线见 `reports/github_baseline_reference.json`：ComponentF1 0.3810、PinF1 0.1751、NetHypergraphF1 0.1579、NetLineF1 0.0299、PinPairF1 0.0181、本地非官方总分 21.63。

## 自动测试

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py pcb tools evaluate_component_v4.py run_component_ablation.py
```

## Stage 选择

默认选择在 `configs/current.json`：

```json
{
  "text": "current",
  "component": "v4",
  "pin_localization": "v3",
  "pin_semantics": "v3",
  "wire": "v3",
  "topology": "v3",
  "submission": "official"
}
```

不要修改 `configs/current.json` 来做个人实验。复制为 `configs/examples/<你的实验>.json`，只替换你负责的 Stage 版本。Pipeline 通过 `pcb/core/registry.py` 创建对应实现，其他 Stage 不需要改 import。

| 负责人 | 主要修改范围 | 不应直接修改 |
|---|---|---|
| Text/OCR | `pcb/text/`，必要时新建 Text Stage 版本 | Component、Pin、Wire、Topology |
| Component | `pcb/component/` | Pin、Wire、Topology、Submission |
| Pin Localization | `pcb/pin/localization/` | Component、Pin Semantics、Wire |
| Pin Semantics | `pcb/pin/semantics/` | Component、Pin Localization、Wire |
| Wire | `pcb/wire_stage/` | Component、Pin、Topology 内部 |
| Topology | `pcb/topology_stage/` | Component、Pin、Wire 内部 |

新增版本时通常只需要三类变更：你的 Stage 新文件、`registry.py` 的一条注册、你自己的配置和测试。完整步骤见 `docs/GITHUB_COLLABORATION.md`。

## OCR 与模型

- 默认 OCR：`hybrid`，即全图 RapidOCR 加所有 640 像素 EasyOCR tile。
- 默认 YOLO：`models/component_yolo11n_continue_v2_best.pt`。
- 当前 Windows/Python 3.14 会使用仓库内已验证的 OCR runtime；其他兼容环境使用 `requirements.txt` 安装的包。
- `models/SHA256SUMS.json` 是模型与 OCR 文件的一致性依据。
- CPU 可以运行，但完整 150 例会很慢；GPU 用 `--device 0 --ocr_device cuda`。

## 协作规则

1. `main` 始终保持可运行；每项工作使用 `feature/<模块>-<版本>` 分支。
2. 不直接改别人的 Stage，也不要在 `main.py` 添加个人算法分支。
3. 不提交 `runs/`、数据集、cache、环境目录和逐例评测输出。
4. 改模型权重前先讨论，并通过 Git LFS 提交，同时更新 SHA256。
5. 每个 PR 写清楚配置、测试范围、旧/新指标和是否接触封存集。
6. 公共契约 `pcb/schema.py`、`pcb/core/interfaces.py`、Submission 和坐标转换需要全组评审。

## 文档

- `docs/ARCHITECTURE.md`：Pipeline 与 Registry。
- `docs/MODULE_CONTRACTS.md`：各 Stage 的输入输出。
- `docs/GITHUB_COLLABORATION.md`：上传、分支和只修改自己模块的完整步骤。
- `docs/MIGRATION_MAP.md`：旧文件到模块化结构的映射。
- `docs/REFACTOR_REPORT.md`：150 例行为一致性证据。
- `CONTRIBUTING.md`：提交和 PR 最低要求。
