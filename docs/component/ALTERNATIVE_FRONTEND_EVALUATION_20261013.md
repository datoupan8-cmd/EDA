# Component视觉与文字替换实验（2026-10-13）

本轮实际实现了两种模型替换：Faster R-CNN ResNet50-FPN替换YOLO器件检测，以及PP-OCRv5移动版ONNX模型替换旧HybridOCR。四组对照已经完成。当前结果最好的方案是保留YOLO、替换为PP-OCRv5：前150例Component宏平均F1从0.382541提高到0.396791。新Faster R-CNN增加了召回，同时产生更多误检；目前不能用它替换默认检测器。

## 数据与评测口径

- 只使用公开训练集0001–0150。本轮未读取0151–0200或10_GTcase，也未使用此前保留验证的结果进行训练、调参或错误分析。
- Faster R-CNN固定使用编号不被5整除的120例训练。0005、0010、……、0150组成内部开发检查30例，训练进程禁止访问其图像与标注。
- 30例在以前的项目开发中也被使用过，因此不是独立保留验证集；前150例全量成绩中有120例是本轮检测器的训练拟合数据，不能当成泛化成绩。
- 使用冻结的本地`evaluate_v2.py`，没有运行官方Docker评分器，报告明确标记`OFFICIAL_SCORE=false`。这些分数是本地Component对照分数。
- 普通器件要求key直接匹配、type一致，bbox四坐标最大误差不超过20像素；box还要求Name一致，r/c还要求value一致。GND忽略序号，按空间匹配。本轮没有修改该评分逻辑。
- 每例算F1后再平均，得到macro F1；汇总全部TP/预测数/GT数后算F1，得到micro F1。下面主表使用macro F1，与上一轮本地报告保持同一口径。
- 四组共用Schema与文字归属、型号/数值解释规则。组合新路线的Component检测不调用YOLO预测，也不消费旧HybridOCR结果；完整团队运行时仍会初始化公共模型对象。
- 全部600份预测先保存，随后才读取150份GT进行评分。源码、配置、模型SHA256在运行前冻结，运行结束后检查一致。

## 完成的四组对照

| 方案 | 前150例macro F1 | 内部30例macro F1 | 120例训练拟合macro F1 | 前150例micro F1 | 前150例TP / 预测数 |
|---|---:|---:|---:|---:|---:|
| 当前：YOLO11n + 旧HybridOCR | 0.382541 | 0.368821 | 0.385971 | 0.375032 | 2205 / 5679 |
| Faster R-CNN + 旧HybridOCR | 0.354629 | 0.353783 | 0.354840 | 0.351531 | 2479 / 8024 |
| YOLO11n + PP-OCRv5 | **0.396791** | **0.394545** | **0.397353** | **0.393741** | 2315 / 5679 |
| Faster R-CNN + PP-OCRv5 | 0.372675 | 0.378900 | 0.371119 | 0.373086 | 2631 / 8024 |

前150例GT器件数6080；内部30例GT器件数1279。全部150例成功，0失败。完整数值及逐案例结果见`runs/component_alternative_frontends_dev150_20261013/report.json`。

仅换PP-OCRv5时，前150例macro F1增加0.014250（约1.425个百分点），正确匹配增加110；内部30例增加0.025724，正确匹配从423增加到467。逐案例49例提高、19例下降、82例不变。因此它有整体收益，也存在真实回退案例。

为了说明检测与文字的贡献，还计算了只要求type正确、bbox误差不超过20像素的一对一几何诊断；它忽略key、Name与value，既不是完整ComponentF1，也不是目标检测AP。

| 视觉检测器 | 前150例几何precision | 前150例几何recall | 前150例几何macro F1 | 内部30例几何macro F1 |
|---|---:|---:|---:|---:|
| YOLO11n | 0.842578 | 0.787007 | 0.819758 | 0.807114 |
| Faster R-CNN | 0.695663 | 0.918092 | 0.802532 | 0.740863 |

替换OCR时，器件数量、type和bbox的多重集合保持一致。因此新OCR的收益来自文字相关字段及归属变化。Faster R-CNN在当前设置下提高几何召回，但预测数从5679增至8024，几何精确率从0.842578降到0.695663；内部30例的几何F1也下降。这说明这份新检测器仍有误检与定位问题，不能只因TP增加就宣布改进。

## 实现与训练

视觉代码：`pcb/component/alternative_rcnn.py`。从[PyTorch公开COCO权重](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)初始化Faster R-CNN ResNet50-FPN；替换为本项目43类输出头，使用8/16/32/64/128像素锚框，768像素切片、192像素重叠以及按类NMS。模型从图像产生独立候选区域。检测置信度0.10和NMS IoU 0.50在本次对照前固定。

训练脚本：`pcb/component/experiments/train_faster_rcnn.py`。固定120例、12个epoch、每例每epoch抽取2个768像素裁剪；使用GT的器件框与类别，不学习GT位号、文字值、引脚或连通关系。随机种子20261013，AdamW初始学习率0.0002，余弦调度，AMP。实际训练425.66秒，训练GT打开120次，检查30例标注打开0次。未根据本轮评测继续调参或重训。

最终权重：`models/component_fasterrcnn/final.pth`，SHA256为`c5ad7552d2286173ff7280969a2c4e8d602cdf98c23e3ea63d6a8e7e4c2406b3`。训练报告及冻结训练源码位于`runs/component_faster_rcnn_train120_20261013/`。训练后只把checkpoint训练ID校验移至构建网络之前；冻结训练源码可按报告哈希复核。

文字代码：`pcb/component/alternative_ocr.py`。使用PP-OCRv5移动检测与识别ONNX模型，RapidOCR固定3.9.2；明确指定v5模型路径和版本，不使用包的默认模型。完整图加768像素切片，重叠128像素，切片放大2倍后识别，再还原坐标并去重。方向分类关闭。ONNX文件按官方模型配置中的SHA256核验。相关论文和模型来源见[算法目录](VISION_TEXT_ALGORITHM_CATALOG_20261013.md)。

新模型、额外依赖、缓存与报告均保存在E盘项目目录。已确认的本地运行环境是Python 3.12.14、Torch 2.14.0+cu132、Torchvision 0.29.0+cu132、RTX 3060 Laptop 6GB；这不是官方Ubuntu容器运行验证。

额外OCR依赖在`.runtime/alternative_frontend`：rapidocr==3.9.2、omegaconf==2.4.0、antlr4-python3-runtime==4.13.2、colorlog==6.12.0、tqdm==4.70.1。ONNX Runtime及图像处理依赖复用现有环境或vendor。移到其他机器时还需提供Torch/Torchvision与上述OCR依赖、`models/ocr_v5`模型；本轮没有制作提交压缩包。

## 可切换的完整入口

| 入口 | Component选择 | OCR选择 |
|---|---|---|
| `configs/team.json` | 当前contextual_values_v2 | PP-OCRv5（后续完整评测后更新） |
| `configs/component/team_contextual_values_v2.json` | contextual_values_v2 | 原HybridOCR |
| `configs/component/team_faster_rcnn_v1.json` | faster_rcnn_v1 | 当前HybridOCR |
| `configs/component/team_ppocrv5_yolo_v1.json` | contextual_values_v2 | ppocrv5_mobile |
| `configs/component/team_rcnn_ppocrv5_v1.json` | faster_rcnn_v1 | ppocrv5_mobile |

上述四组Component实验完成时默认配置保持原版本。随后已完成[YOLO + PP-OCRv5完整流程配对评测](PPOCRV5_FULL_PIPELINE_AB_20261013.md)：Component、Pin、NetHypergraph提高，本地加权分23.691998→25.056419，NetLine略降。因此后续默认全图OCR已更新为PP-OCRv5，Faster R-CNN仍保留为实验入口，原HybridOCR组合继续可选择。

在`E:\EDA\EDA`运行以下本地命令，可选择得分最高的OCR方案。`--image`只用于允许范围内的开发PNG，完整数据评测脚本默认只允许0001–0150。每次使用新的输出目录。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = 'E:\EDA\EDA\.runtime\alternative_frontend;E:\EDA\EDA\.runtime\py312'
& 'C:\Users\pps\Desktop\EDA0.5\.venv\Scripts\python.exe' tools/run_team_pipeline.py `
  --config configs/component/team_ppocrv5_yolo_v1.json `
  --image '<0001–0150内的PNG绝对路径>' `
  --output runs/component_ppocrv5_new_run --device cuda --save-debug
```

## 验证证据

- 305项维护中的测试通过，包括5项新测试：120/30固定分割、训练访问拦截、GT读取许可、独立候选组装、切片坐标还原。日志：`runs/component_alternative_frontends_tests_20261013.txt`。
- 全150例基线JSON逐份与上一轮`component_contextual_values_registry_dev150_20261013`对照，内容完全一致。
- 86份冻结源文件/配置哈希及全部模型哈希复核一致；600份预测文件存在。
- 复核记录与评测后收集的600份预测SHA256保存在`runs/component_alternative_frontends_dev150_20261013/verification.json`；这些预测哈希是评测后核验记录，不冒充评测前封存证据。
- 用允许的0005案例执行两次完整流程：YOLO + PP-OCRv5与Faster R-CNN + PP-OCRv5均完成Pin/Wire/Topology输出，并通过严格输出Schema检查。其Component字段（不含新增pins）分别与四组前端对照的0005预测完全一致。
- 两次完整流程的GT读取数均为0，耗时分别7.73秒和14.63秒。它们是单例流程检查，不代表Pin/Net准确率评测。报告：`runs/component_alternative_smoke_ppocrv5_0005_20261013/report.json`与`runs/component_alternative_smoke_rcnn_ppocrv5_0005_20261013/report.json`。

完整算法目录列出Faster R-CNN、RT-DETRv2、DINO、Grounding DINO、Mask R-CNN、SAM 2、PP-OCRv5、DBNet、CRAFT、SVTRv2、PARSeq与TrOCR的原论文、用途和实际测试状态。仅Faster R-CNN和PP-OCRv5是本轮新增的实际模型实验；其余候选不能声称已经提高本项目分数。
