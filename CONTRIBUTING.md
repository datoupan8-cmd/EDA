# 协作与提交规则

## 开始工作

```powershell
git switch main
git pull --ff-only
git switch -c feature/pin-localization-v4
python tools/repository_doctor.py
python -m unittest discover -s tests -v
```

一个分支只解决一个 Stage 或一个清晰问题。不要把算法改动、格式整理、依赖升级和模型替换放进同一个 PR。

## 允许的修改范围

- Component：`pcb/component/`。
- Pin Localization：`pcb/pin/localization/`。
- Pin Semantics：`pcb/pin/semantics/`。
- Wire：`pcb/wire_stage/`。
- Topology：`pcb/topology_stage/`。
- Text/OCR：`pcb/text/`。

新增版本后，可以在 `pcb/core/registry.py` 增加一条注册，并在 `configs/examples/` 新建个人配置。不要覆盖 `configs/current.json` 或现有版本实现。

以下是公共契约，修改前应由全组评审：

- `pcb/schema.py`
- `pcb/core/interfaces.py`
- `pcb/core/pipeline.py`
- `pcb/output/`
- `pcb/coordinates.py`
- `main.py`
- `evaluate_v2.py`

## 提交前

```powershell
python -m unittest discover -s tests -v
python tools/repository_doctor.py
git status --short
git diff --check
```

需要报告准确率时，使用同一版官方数据和同一配置运行 `tools/run_dev_benchmark.py`。禁止查看或使用 0151～0200 与 10_GTcase。

不要提交：官方数据、`runs/`、OCR cache、逐例预测、`.venv`、个人绝对路径、账号密钥。模型文件必须由 Git LFS 管理。

## Pull Request 必须说明

- 修改了哪个 Stage、为什么改；
- 使用哪个配置；
- 测试是否通过；
- 0001～0150 的旧/新本地指标；
- 是否改变模型、阈值、schema 或坐标语义；
- `SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE`。

详细工作流见 `docs/GITHUB_COLLABORATION.md`。
