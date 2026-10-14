# 可替换的视觉与文字算法（2026-10-13）

这里区分论文方法、对本项目的适用性推断和实际完成的实验。论文自己的通用数据集成绩不能换算成本项目ComponentF1。本轮实际新路线为Faster R-CNN + PP-OCRv5；RT-DETR/DINO/SAM等列为已调研候选，没有声称已训练或评测。

## 视觉检测与分割

| 算法与原始来源 | 核心方法 | 对Component的用途或限制 | 本轮状态 |
|---|---|---|---|
| [Faster R-CNN，2015](https://arxiv.org/abs/1506.01497) | RPN先提出候选区域，再做分类和框回归 | 已实际使用ResNet50-FPN多尺度特征及小目标锚框；替换YOLO检测器 | 已训练并纳入四组PNG对照 |
| [RT-DETRv2，2024](https://arxiv.org/abs/2407.17140) | 实时Transformer检测器，改进多尺度采样及训练策略 | 候选：可以作为另一条端到端检测路线，需要电路符号数据适配 | 调研，未实测 |
| [DINO，2022](https://arxiv.org/abs/2203.03605) | 对比去噪训练、混合查询选择和端到端框预测 | 候选：复杂、密集图的检测；训练与显存成本需要另测 | 调研，未实测 |
| [Grounding DINO，2023/2024](https://arxiv.org/abs/2303.05499) | 把语言提示融入开放集检测 | 候选：用“电阻、电容、芯片”等类别提示提出区域；预训练效果能否覆盖电路符号需要验证 | 调研，未实测 |
| [Mask R-CNN，2017](https://arxiv.org/abs/1703.06870) | Faster R-CNN增加实例掩膜分支 | 候选：分离器件本体与连线；真正训练掩膜分支需要实例掩膜标注，现有bbox不能等价替代 | 调研，未实测 |
| [SAM 2，2024](https://arxiv.org/abs/2408.00714) | 可提示的图像/视频分割模型 | 推断：可以辅助器件边界细化；本身不提供本项目器件分类与位号，不能单独完成Component输出 | 调研，未实测 |

## 文字检测与识别

| 算法与原始来源 | 核心方法 | 对Component的用途或限制 | 本轮状态 |
|---|---|---|---|
| [PP-OCRv5，PaddleOCR 3.0报告，2025](https://arxiv.org/abs/2507.05595) | 多语言文字检测/识别方案 | 替换旧HybridOCR，以ONNX移动检测和识别模型执行完整图加切片识别 | 已接入并纳入四组PNG对照 |
| [DBNet，2019/2020](https://arxiv.org/abs/1911.08947) | 分割网络结合可微二值化定位文字 | 候选：精细文字区域定位；检测框还须送入识别器读字符 | 调研；已有Paddle系OCR采用相关检测机制，未新增独立对照 |
| [CRAFT，2019](https://arxiv.org/abs/1904.01941) | 字符区域与字符间亲和分数恢复文字 | 候选：检测小字、断开字符；本身不输出文字内容 | 既有EasyOCR已使用CRAFT，本轮没有把它当作新算法重复实验 |
| [SVTRv2，2024/2025](https://arxiv.org/abs/2411.15858) | CTC识别，结合多尺寸缩放、特征重排和语义引导 | 候选：位号、型号及长芯片名识别；需要文字裁剪输入和符号字符集检查 | 调研，未独立实测 |
| [PARSeq，2022](https://arxiv.org/abs/2207.06966) | 置换自回归序列建模进行场景文字识别 | 候选：处理OCR字符混淆；仍需要先定位文字，并检查Ω/µ等字符覆盖 | 调研，未实测 |
| [TrOCR，2021/2022](https://arxiv.org/abs/2109.10282) | 预训练图像Transformer加文字Transformer生成字符序列 | 候选：打印体或手写文字裁剪识别；语言生成输出需遵守原图证据，不能据常识补出位号 | 调研，未实测 |

上述“候选/推断”是结合本项目需求的判断，不是论文已经证明能提高这份数据的结论。

## 实际替换路线

四组共用输出Schema和相同的文字归属、型号/数值解释逻辑，只替换检测器与OCR，从而隔离两种变化的贡献：

1. 当前基线：YOLO11n + 旧HybridOCR + contextual_values_v2。
2. 仅换视觉：新Faster R-CNN + 旧HybridOCR。
3. 仅换文字：YOLO11n + PP-OCRv5。
4. 同时换视觉和文字：新Faster R-CNN + PP-OCRv5。

组合路线的Component检测不执行YOLO预测，也不消费旧HybridOCR识别结果；普通完整团队入口仍会初始化公共运行时对象。文字归属保留同一规则，是控制变量，并非把Faster R-CNN包装在原YOLO检测结果之后。

Faster R-CNN从公开COCO预训练权重开始，只用0001–0150里编号不被5整除的120例训练12个epoch；0005、0010、…、0150为内部开发检查30例，训练进程禁止读取这些案例的图像和标注。所有训练、下载、依赖和输出均保存在E盘项目目录。该30例以前也用于过开发工作，不应当当作全新保留测试集。

本轮禁止访问0151–0200、10_GTcase等保留数据。前150例全量汇总中有120例属于新检测器的训练拟合数据，必须另看30例开发检查结果；全量拟合分数不能视作泛化成绩。

训练证据：`runs/component_faster_rcnn_train120_20261013/report.json`；最终权重：`models/component_fasterrcnn/final.pth`；算法对照：`runs/component_alternative_frontends_dev150_20261013/report.json`。

OCR模型来自[RapidAI官方模型配置](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml)，运行时固定RapidOCR 3.9.2，模型路径明确固定PP-OCRv5，未跟随该包默认的其他模型版本。下载后的三个ONNX文件均按官方SHA256核验；方向分类器仅为初始化依赖，实际文字识别关闭方向分类。

完整本地结果、测试及默认版本选择见[配套实验报告](ALTERNATIVE_FRONTEND_EVALUATION_20261013.md)。本轮四组中YOLO + PP-OCRv5最好：前150例宏平均ComponentF1为0.396791，原基线为0.382541；Faster R-CNN + PP-OCRv5为0.372675。随后已完成[完整流程配对评测](PPOCRV5_FULL_PIPELINE_AB_20261013.md)，本地加权分从23.691998升至25.056419，默认全图OCR已切换为PP-OCRv5；旧HybridOCR与Faster R-CNN实验入口均保留。
