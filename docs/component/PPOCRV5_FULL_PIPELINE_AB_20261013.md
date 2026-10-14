# YOLO + PP-OCRv5完整流程对照（2026-10-13）

本次把YOLO + PP-OCRv5接入完整Component、Pin、Wire、Topology和Submission流程，与原YOLO + HybridOCR进行配对对照。完整结果在本文件的结果段落记录；运行证据位于`runs/component_ppocrv5_full_ab_dev150_20261013/`。

## 数据隔离与控制变量

- 默认使用按编号排序的0001–0150；脚本禁止重复编号、缺失图片、案例重定向以及0151–0200、10_GTcase等保留数据。本轮没有训练或修改算法阈值。
- 参照配置为`configs/component/team_contextual_values_v2.json`，候选配置为`configs/component/team_ppocrv5_yolo_v1.json`。两者仅`text`选择不同：current与ppocrv5_mobile。
- YOLO权重、contextual_values_v2、L39引脚模型及语义、Wire v3_frame_guard、Topology v3、NetHypergraph/NetLine legacy_v2均固定。
- 替换的是全图文字检测/识别；引脚模块自己的局部OCR保持原模型。新的全图文字会影响位号、型号、数值、引脚文字归属和连线文字抑制，因此须看四项指标，不能只看Component。
- 300份完整预测均先保存并记录SHA256，之后才读取150份对应GT；推理阶段不能读取标注。`predictions_sealed_before_GT.json`记录评测前的预测哈希。
- 源码、配置与模型在推理前冻结哈希，推理后及评测后都核对；默认配置只会在本次对照完成后根据结果决定是否切换。

## 本地评分口径

使用现有`evaluate_v2.py`，报告为`OFFICIAL_SCORE=false`。没有官方评分脚本，本轮不是官方Docker成绩。先按例计算F1，再取150例平均作为macro F1；另外保留汇总TP/预测数/GT数算出的micro F1。

本地加权分数为`100 × (0.30 × Component宏平均F1 + 0.25 × Pin宏平均F1 + 0.35 × NetHypergraph宏平均F1 + 0.10 × NetLine宏平均F1)`。前150例属于开发数据，不是独立保留验证成绩。

## 运行方式

统一完整入口：`tools/run_team_pipeline.py`；本次配对对照脚本：`pcb/component/experiments/run_ppocrv5_full_ab.py`。

```powershell
python -m pip install -r configs/component/requirements_ppocrv5.txt
python tools/run_team_pipeline.py --config configs/component/team_ppocrv5_yolo_v1.json --image '<开发PNG路径>' --output runs/ppocrv5_one_new --device cuda
```

第一行用于在其他已有Python环境中补齐依赖；本机已有可用依赖，本轮没有重复安装。三个OCR模型来自官方发布文件，来源、字节数和SHA256保存在`models/ocr_v5/manifest.json`。原引脚局部OCR和EasyOCR依赖仍需要保留。

复现前150例完整配对评测：

```powershell
python pcb/component/experiments/run_ppocrv5_full_ab.py --dataset '<200_train_cases绝对路径>' --output '<新的E盘输出目录>' --device cuda
```

脚本默认白名单0001–0150，日志会打印实际案例编号。输出目录必须是新的且位于原数据集之外。

本地运行环境为Python 3.12.14、RapidOCR 3.9.2、ONNX Runtime 1.30.0、Torch 2.14.0+cu132、Torchvision 0.29.0+cu132、RTX 3060 Laptop。运行环境检查记录在`runs/component_ppocrv5_runtime_check_20261013.json`，新增文件、模型、缓存和报告保存于E盘。

已有旧OCR与引脚读取缓存；候选新全图OCR复用上一轮的图像内容缓存，而新的引脚局部文字组合需要首次识别。因此本次两组耗时不能当成同等缓存状态下的速度对比。

## 结果

完整150例已完成，150例成功、0失败，访问保护拦截数0。300份预测哈希与评测前封存记录一致。完整推理与评分耗时2339.78秒，约39分钟；缓存状态不同，不能将两组耗时差直接解释为算法速度差。

| 指标（macro F1） | 原YOLO + HybridOCR | YOLO + PP-OCRv5 | 变化 |
|---|---:|---:|---:|
| Component | 0.382541 | 0.396791 | +0.014250 |
| Pin | 0.199957 | 0.210216 | +0.010260 |
| NetHypergraph | 0.195480 | 0.215247 | +0.019767 |
| NetLine | 0.037504 | 0.036363 | -0.001140 |
| 本地加权总分（百分制） | 23.691998 | 25.056419 | +1.364421 |

Component正确匹配从2205增到2315；Pin从2353增到2567；NetHypergraph成员正确匹配从2317增到2612。NetLine正确线段匹配从1130降到983。辅助PinPair宏平均F1从0.028492降到0.025639，它不进入上述四项加权公式，属于需要继续观察的结构指标。

| 逐例结果 | 提高 | 下降 | 不变 |
|---|---:|---:|---:|
| Component | 49 | 19 | 82 |
| Pin | 53 | 30 | 67 |
| NetHypergraph | 62 | 34 | 54 |
| NetLine | 53 | 45 | 52 |
| 本地加权总分 | 77 | 37 | 36 |

## 默认选择与核验

根据前150例的Component、Pin、NetHypergraph和加权总分整体收益，已经把`configs/component/current.json`的`text`切换为`ppocrv5_mobile`。其余默认算法选择保持原版本。NetLine和辅助PinPair有回退，报告保留负向结果，没有因它们权重较低而省略。

直接运行`tools/run_team_pipeline.py`现在使用YOLO + PP-OCRv5；需要原YOLO + HybridOCR时，显式传入`--config configs/component/team_contextual_values_v2.json`。依赖`rapidocr==3.9.2`已加入根目录`requirements.txt`，模型和新增依赖保存于E盘。没有制作新提交压缩包。

切换前已复核223份冻结源码/配置与11份模型/库哈希，并保存到`runs/component_ppocrv5_full_ab_dev150_20261013/source_snapshot/`及`verified_before_promotion.json`。切换后的默认配置与本次候选配置逐字段完全一致。默认配置的哈希会因这次有意切换而改变，应使用冻结源码快照复核原实验，而不是要求已切换配置仍具有旧哈希。

所有300份完整预测的Component字段（不含新增pins）与上一轮对应前端预测完全一致。本轮实际测出的Component成绩与上一轮前端相同，Pin/Net结果则来自本轮完整流程。

切换后使用默认入口、不显式指定配置，在0005案例完成一次独立完整流程检查：严格Schema验证通过，完整result.json与本轮0005候选结果完全一致，GT打开次数0。记录在`runs/component_ppocrv5_default_smoke_0005_20261013/report.json`。

标准测试入口`tools/run_team_tests.py`已通过305项测试，日志为`runs/component_ppocrv5_default_tests_20261013_v2.txt`。默认配置断言同步为新的OCR选择；测试发现器跳过没有测试源码的空目录，避免历史缓存目录造成发现错误。

本轮未读取后50例及10_GTcase，也没有运行官方Docker评分器。以上收益只说明当前前150例上的本地配对结果，不能承诺官方或保留数据上的收益。
