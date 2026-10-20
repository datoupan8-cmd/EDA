# P01 Component器件筛选

P01在YOLO11m、PP-OCRv5 beam与局部PARSeq之后，用固定的逻辑回归模型判断器件输出是否有足够把握。输入为检测置信度、可见位号、数值/型号、框面积和类别，共12个特征；概率门槛固定为0.5，地符号保留。没有额外的ResNet、HOG或视觉语言模型。

`configs/team.json`引用的Component配置选择`ordered_yolo11m_local_parseq_p01_v1`。原`ordered_yolo11m_local_parseq_v1`等版本仍可显式选择。Pin/Net代码和配置保持现状；本轮只计算Component评分。

## 本地对照

以下为非官方代理评分器`evaluate_v2.py`的ComponentF1宏平均，不是比赛平台分数。只使用0001–0150；0151–0200及10_GTcase未使用。

| 开发范围 | 无P01基线 | P01 | 增量 |
|---|---:|---:|---:|
| 全150例 | 0.433059 | 0.540106 | +0.107047 |
| 固定train120 | 0.433706 | 0.542452 | +0.108746 |
| 内部check30 | 0.430468 | 0.530720 | +0.100251 |

146例上升、2例下降、2例不变。预测数量6163→3434，正确匹配2694→2642。收益来自精度和覆盖率的取舍；本地身份匹配规则可能与官方不同，需要平台确认。

逻辑回归只在train120拟合。内部check30也参与过此前ResNet候选的检查点选择，属于已使用的开发数据，不能当作独立保留验证。本轮P01复测没有重新训练或调整阈值；两组共300份输出全部封存后才打开标注计分。相同图像OCR结果供两组共用，YOLO11m和局部PARSeq重新推理，P01输出与先前固定实验逐例一致。

## 运行

先获取LFS权重并安装项目依赖：

```powershell
git lfs pull
python -m pip install -r requirements.txt
python tools/repository_doctor.py
python tools/run_team_pipeline.py --image <图片路径> --output runs/p01_sample --device cuda
```

单独对照P01与无P01基线：

```powershell
python tools/evaluate_component_p01.py --dataset <200_train_cases路径> --output runs/p01_compare
```

此脚本强制检查完整0001–0150白名单，记录实际案例编号，缺失或重复时停止；不会读取保留数据。无OCR缓存时重新运行同一PP-OCRv5前端。`--ocr-device cpu`可用于无OCR CUDA依赖的环境。不同设备和依赖可能影响耗时及数值结果。

`--shared-ocr`可选用已封存的图像文字结果，只用于固定OCR变量的Component对照：目录中需有`manifest.json`，声明完整150例、GT读取为0、`image_only=true`、OCR源码SHA256及每个案例文件的SHA256；每个案例文件只含`image_sha256`和`texts`。脚本核对当前图片哈希后使用这些文字，不接收GT标注。`--reference`可逐例核对另一次已落盘的图像推理输出。

## 提交前检查

拟提交内容导出的干净副本通过377项全项目测试，`repository_doctor.py`的依赖、模型SHA256及注册检查全部通过，`git diff --check`通过。其他未提交实验未复制到该副本。

## 模型与验证边界

- YOLO11m、PARSeq官方权重及train120微调权重由Git LFS管理；P01逻辑回归参数是小型JSON。全部加入模型SHA256清单。
- PARSeq上游推理源码、LICENSE、NOTICE和逐文件SHA256保存在`pcb/component/_vendor/parseq/`。推理依赖`timm==0.9.16`。
- 打包入口使用允许案例0001重新运行全页OCR和完整流程，输出模式正确、器件结果与单项对照一致。
- 本机Docker服务返回HTTP 500/超时，本轮没有完成官方容器复测。Windows试跑不等于官方容器或平台评分验证。
- 不提交数据集、逐例预测、运行缓存、Docker镜像、Python环境及其他候选模型。

SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE
