# PCB 原理图自动解析

本仓库用于“2026 中国研究生创芯大赛·EDA 精英挑战赛 - AI 驱动的 PCB 原理图自动解析”。目标是把一张原理图 PNG 转换为包含 `components`、`pins`、`nets` 的结构化 JSON。

当前按 [PROGRESS.md](PROGRESS.md) 中的 6 周计划推进。每个阶段都必须有可运行代码、自动检查和结果记录，完成后再进入下一阶段。

## 当前状态

- 已完成：赛题规则梳理、数据集清点、JSON 数据契约、题面对齐的本地评测器、首版元件检测与 OCR 基线。
- 正在进行：只改进 `components` 的框、类型、key、Name 和 value；暂不处理引脚和网络。
- 数据隔离：开发只使用 `200_train_cases/0001` 至 `0150`；其余案例只有收到用户明确的验证命令后才能读取。

## 快速开始

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest discover -s tests -v
& $python scripts\audit_dataset.py
& $python scripts\validate_target.py '赛题六公开数据集\200_train_cases\0001\0001-KiCad_target.json'
& $python scripts\prepare_components_dataset.py
& $python scripts\evaluate_component_baseline.py
```

首版 components 结果见 `reports/component_baseline_dev.md`。评测脚本只使用固定的内部 dev 案例（`0005, 0010, ... 0150`），不会读取保留数据。

单张图片推理示例：

```powershell
& $python scripts\predict_components.py `
  '赛题六公开数据集\赛题六公开数据集\200_train_cases\0015\0015-Datasheet.png' `
  --output 'artifacts\component_demo\0015.json' `
  --diagnostics 'artifacts\component_demo\0015_diagnostics.json' `
  --overlay 'reports\component_prediction_0015.png'
```

输出 JSON 有 `components`，并保留空的 `pins` 和 `nets` 容器；P3 不生成引脚或网络结果。
无法可靠识别 key 的检测框不会丢弃，而以 `__unresolved_NNNN` 作为唯一占位键；若框内能识别出器件型号则写入 `Name`，否则 `Name` 为 `null`。

## 目录

```text
src/pcb_parser/       数据契约、解析与后续推理模块
scripts/              数据审计和命令行工具
tests/                自动测试
docs/                 规则与设计决策
reports/              可复现的分析结果
赛题六公开数据集/     官方公开样本与元件库
```

## 关键约束

- 坐标系为图片左下角原点，所有坐标均为像素。
- 元件按 component key 直接匹配，key 错误会连带影响 pin 和 net。
- 引脚坐标、线段端点默认容差为 5 px；元件 bbox 默认容差为 20 px。
- 单案例权重：ComponentF1 30%、PinF1 25%、NetHypergraphF1 35%、NetLineF1 10%。
- 全部案例须在 2 小时内完成，并需控制大模型 Token 消耗。
- P2 的 100 分仅用于验证评测器（GT 与同一份 GT 比较），不是识别模型正确率；真实 components 基线以内部 dev 报告为准。
