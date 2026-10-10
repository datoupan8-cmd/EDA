# Pin L39：GitHub协作分支与运行说明

这是已冻结、完成本地150例诊断的Pin比赛候选。没有因为上传而调整算法、模型或阈值。

分支：`feature/pin-l39-l40`。原`main`和`feature/modular-baseline`不被覆盖。本分支基于原模块化基线；协作分支后来出现的同名Pin v4及Component改动属于另一条开发线，尚未融合或替代本候选。合并前必须人工处理同名版本冲突，不能直接选择全部覆盖。

## 下载并准备环境

```powershell
git clone --branch feature/pin-l39-l40 https://github.com/datoupan8-cmd/EDA.git EDA_Pin
Set-Location EDA_Pin
git lfs install
git lfs pull
```

必须取得LFS的真实权重，不能把指针文件当模型。模型、OCR离线资产和公开元件库随本分支提供，不需要原电脑的Downloads、Desktop或缓存目录。Python、兼容依赖和可用CUDA GPU仍需自行准备；环境依赖见原requirements.txt及requirements.runtime.txt，后者用于补充官方基础环境，不能盲目用来替换PyTorch等基础包。

## 运行Pin组合：单张图片验证

```powershell
python -B tools/run_pin_combined_l38.py --image D:/EDA/dev_png_input/0001-KiCad.png --output runs/pin_case_01 --library assets/cpntLibrary.json --fresh-cache
```

输入是一张PNG，输出用全新目录。替换自己的输入、输出路径即可；工程内部路径均由代码位置解析。Linux将`python`改为`python3`。此命令从图像运行已冻结的Pin组合，生成result.json及诊断信息，不读取GT。

本次GitHub新增内容仅为Pin实现、必要支持helper、权重、公开库、实验配置、测试和简洁报告。官方部署适配器、Docker构建和比赛上传不在此次Pin提交范围；本机独立候选目录仍单独保留，等待用户确认后再处理官方评测。

原`configs/current.json`仍为旧baseline。注册某个历史版本并不自动启用完整L38组合，运行新组合请使用上述明确入口。正式推理只读取图片，不读取target或旧prediction，不联网下载模型。

## 自己验证正确率

先运行不需要数据的测试：

```powershell
python -B -m unittest tests.pin.test_pin_combined_l38 tests.pin.test_pin_combined_l39 -v
python -B -m unittest discover -s tests/contracts -v
```

只有0001～0150允许开发评测。独立全量A/B工具提供`--dataset`和`--library`，需要自己准备最新公开开发集；GT只能由评测工具在预测保存后读取，不允许进入推理。

```powershell
python -B tools/validate_pin_combined_l39.py --task run --dataset D:/EDA/200_train_cases --library assets/cpntLibrary.json
python -B tools/validate_pin_combined_l39.py --task summarize --dataset D:/EDA/200_train_cases
python -B tools/validate_pin_combined_l39.py --task verify --dataset D:/EDA/200_train_cases
```

工具会在本地生成报告、预测与缓存，不需要上传这些数据。禁止读取0151～0200、Golden、10GT或QuickTest用于开发。

## 本候选具体组合

原V4 Component/OCR → L1 box位置模型 → E6文字角色准备 → L3完整词读取 → L36分侧联合保序关联 → L37公开库约束 → 原Wire V3加独立F框线隔离 → 原Topology V3 → 原submission格式。

组合编排：`experiments/pin_combined_l38.py`。核心Pin实现和历史可选版本在`pcb/pin/`；定位权重在`reports/pin_learned_locator_l1/best.pt`。这是冻结实现使用的相对路径，仅包含权重，不依赖历史报告。公共元件库是`assets/cpntLibrary.json`，SHA256由组装工具固定校验。Wire框线隔离是组合里的显式helper，未修改原Wire/Topology实现。

注册到Registry的v4～v9属于历史实验版本；切换一个版本号不等于启用完整L38组合。新增算法应放独立版本/实验配置，保留此候选用于回归，不改其他负责人的内部实现。

## 已有结果及限制

全部为本地非官方诊断：150/150；综合分21.633175→23.069626；Pin macro F1 0.175136→0.189135；严格Pin TP 1843→2246，新增403、原TP损失0。Component输出保持不变；网络有局部收益与回退，已计入综合分。

119张定位训练图包含在150例中，历史Check也用于开发，不能宣称独立泛化成绩。复杂box严格召回仍为6.31%，尚未解决全部复杂IC。3张独立目录冷启动输出与L39逐字段及SHA256一致；本B候选尚未通过官方容器和官方评分验收。

说明报告：`reports/pin_full_validation_l39/conclusion.md`。正式评测以官方平台要求为准，先核对容器入口、依赖、时间与输出格式，勿将GitHub上传等同于官方提交。未经用户确认，不上传官方平台、不启动评分。
