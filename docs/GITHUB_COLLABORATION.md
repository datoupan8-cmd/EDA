# GitHub 上传与多人分模块开发

## 一、现在能否上传

代码已按仓库方式准备：算法和配置使用仓库相对路径；官方数据、运行结果、cache、环境目录和历史绝对路径 split 被排除；模型由 Git LFS 管理；模型哈希可自动验证。

首次建议建立 Private 仓库。原因是当前包含同组训练权重、OCR 模型和第三方运行时，而这些内容的公开再分发权限没有全部确认。Private 仓库可供组内协作；转 Public 前再完成许可证和权重授权审查。

## 二、仓库创建者第一次上传

在 GitHub 网页新建一个空的 Private repository。不要勾选自动创建 README、`.gitignore` 或 License，因为本地已经有这些文件。然后在项目根目录执行：

```powershell
git init -b main
git lfs install --local
git add .
git status
git commit -m "chore: establish portable modular baseline"
git remote add origin https://github.com/组织名/仓库名.git
git push -u origin main
```

如果 `git status` 显示 `runs/`、官方数据集、`.venv` 或个人路径文件已加入，先停止提交并检查 `.gitignore`。

推送后在 GitHub 仓库 Settings 中：

1. 邀请组员；
2. 保护 `main`；
3. 要求通过 Pull Request 合并；
4. 至少要求一名同学 review；
5. 禁止 force push；
6. 如果以后开启 CI，再把测试设为 required status check。

## 三、同学在另一台电脑第一次使用

```powershell
git lfs install
git clone https://github.com/组织名/仓库名.git
cd 仓库名
git lfs pull
conda env create -f environment.yml
conda activate pcb-v4-1
python tools/repository_doctor.py
python -m unittest discover -s tests -v
```

`repository_doctor.py` 若提示某个模型是 `Git LFS pointer only`，说明权重没有真正下载，执行 `git lfs pull`。如果提示哈希不一致，不要继续比较准确率，应重新拉取文件或确认权重更新。

官方数据集放在仓库外，例如 `D:\pcb_data\200_train_cases`。不要复制进 Git 仓库。

## 四、所有人如何得到可比较的准确率

每个人使用同一版官方数据，先检查目录名确实为 `200_train_cases`，再运行：

```powershell
python tools/run_dev_benchmark.py `
  --dataset-root "D:\pcb_data\200_train_cases" `
  --name "姓名_分支名" `
  --device auto `
  --ocr-device auto
```

工具固定运行 0001～0150，并调用相同 evaluator 和 strict validator。生成结果保留在本机，不进入 Git。PR 中只填写汇总指标、运行环境和配置。

不同电脑若要严格比较，至少核对：

- `models/SHA256SUMS.json` 全部通过；
- `configs/current.json` 或实验配置内容一致；
- Python、PyTorch、Ultralytics、OpenCV、OCR 版本；
- CPU/GPU 和 OCR device；
- 官方数据版本与 case 数量。

当前参考环境和指标见 `reports/github_baseline_reference.json`。跨操作系统或依赖版本可能产生少量数值差异，必须记录环境，不应把这种差异误认为算法提升。

## 五、只修改自己的模块：通用规则

不要编辑已有稳定版本，例如 `v3.py`、`v4.py`。新建下一个版本文件，让旧基线随时可回滚。

一次个人算法开发通常只改四处：

1. 自己 Stage 目录中的新实现；
2. 自己 Stage 目录中的测试；
3. `pcb/core/registry.py` 中一条新版本注册；
4. `configs/examples/` 中一个个人实验配置。

不需要修改其他 Stage，也不需要改 `main.py`。

### 示例：只改 Pin Localization

第一步，新建分支：

```powershell
git switch main
git pull --ff-only
git switch -c feature/pin-localization-v4
```

第二步，新建 `pcb/pin/localization/v4.py`。实现 `PinLocalizationStage` 契约，输入和输出以 `pcb/pin/localization/stage.py`、`pcb/core/interfaces.py`、`docs/MODULE_CONTRACTS.md` 为准。不要改变 Component、Pin Semantics 或 Wire 的数据结构。

第三步，在 `pcb/core/registry.py` 导入并注册：

```python
from ..pin.localization.v4 import PinLocalizationStageV4

registry.register("pin_localization", "v4", PinLocalizationStageV4)
```

第四步，复制配置：

```powershell
Copy-Item configs/current.json configs/examples/pin_localization_v4.json
```

只把这一项改为：

```json
"pin_localization": "v4"
```

其他字段保持不变。运行时选择个人配置：

```powershell
python main.py `
  --image "D:\pcb_data\example.png" `
  --output "runs\pin_v4\result.json" `
  --orchestrator modular `
  --pipeline_config configs/examples/pin_localization_v4.json
```

第五步，新增 `tests/pin/test_pin_localization_v4.py`，至少验证返回结构、tip/base/side 合法和不会修改输入 Component。

第六步，先跑小样本排错，再跑固定 150 例，与 `configs/current.json` 基线比较。提交时只应看到本模块文件、测试、个人配置和 Registry 一行。

```powershell
git status --short
git diff -- pcb/pin/localization pcb/core/registry.py configs/examples tests/pin
```

### 只改其他 Stage 时对应目录

| 工作内容 | 新版本放置位置 | Registry kind |
|---|---|---|
| Component | `pcb/component/v5.py` | `component` |
| Pin Localization | `pcb/pin/localization/v4.py` | `pin_localization` |
| Pin Semantics | `pcb/pin/semantics/v4.py` | `pin_semantics` |
| Wire | `pcb/wire_stage/v4.py` | `wire` |
| Topology | `pcb/topology_stage/v4.py` | `topology` |
| Text/OCR | `pcb/text/v2.py` | `text` |

文件名中的版本只是示例，应使用团队约定且未被 Registry 占用的名称。

## 六、哪些文件不能个人直接改

这些文件定义全组公共边界，个人 PR 若必须修改，应单独说明并由其他模块负责人 review：

- `pcb/schema.py`：Component、Pin、Net 等语义；
- `pcb/core/interfaces.py`：Stage 契约；
- `pcb/core/pipeline.py`：公共执行顺序；
- `pcb/output/`、`pcb/submission.py`：比赛 JSON；
- `pcb/coordinates.py`：坐标系；
- `main.py`：统一入口；
- `evaluate_v2.py`：全组评分基准；
- `configs/current.json`：全组当前稳定基线。

如果自己的算法“必须修改公共 schema 才能工作”，先在 PR 说明需要新增的字段、默认值、下游兼容方式和迁移测试，不能直接把其他模块改到能跑为止。

## 七、日常同步与提交

```powershell
git add pcb/pin/localization/v4.py tests/pin/test_pin_localization_v4.py `
  configs/examples/pin_localization_v4.json pcb/core/registry.py
git commit -m "feat(pin-localization): add v4 stage"
git fetch origin
git rebase origin/main
python -m unittest discover -s tests -v
git push -u origin feature/pin-localization-v4
```

然后在 GitHub 建 Pull Request。合并后其他同学执行：

```powershell
git switch main
git pull --ff-only
git lfs pull
```

## 八、避免相互干扰的关键做法

- 每个人一个 feature branch，不共享同一个工作目录中的未提交改动。
- 不在旧版本文件上直接试验；新增版本并通过配置选择。
- 不把个人阈值塞进公共 `current.json`，先放个人配置。
- 不提交运行产物；PR 只保留可审查的汇总。
- 不混合两项优化；Component 与 Pin 修改应分开 PR。
- 合并前用当前 `main` rebase，并重新跑合同测试和本模块 A/B。
- 合并算法 PR 时再决定是否更新 `configs/current.json`；“代码已加入”不等于“成为默认版本”。
