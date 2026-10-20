# Component 局部 OCR 与文字组实验

本文记录上一轮实验。当前默认已更新为 `ordered_value_groups_v2`；后续结果见 [数值归属 V2](VALUE_GROUPS_V2_20261016.md)。

只优化、评测 Component；不构造或运行 Pin、Wire、Topology、Net 阶段。

## 方法

- `local_ocr_v1`：先运行现有 `contextual_values_v2`，对缺位号、缺数值、缺型号的器件选择局部区域。按字体高度扩大器件周边区域并放大，使用现有 PP-OCRv5 模型补读；低置信度的已有文字框额外扩边后仅做识别。每张图最多 24 个区域、12 个文字框补读。
- `local_joint_v1`：使用完全相同的补读候选，将同列多行或同行相邻的位号、数值、型号作为关联证据；位号仍受类型兼容约束和全局匹配约束。数值组归属不明确时退回几何评分。
- `printed_groups_v1`：仅做文字组关联，不增加 OCR。
- **上一轮默认 `printed_value_groups_v1`**：保留 `contextual_values_v2` 的位号、型号、框、类型，只使用已确认的位号文字锚点重新分配 R/C/L 数值。该方案无需额外 OCR；开发集对照中比同时重分配位号更稳定。
- 同一物理区域的不同 OCR 假设只占一个匹配位置。所有新增文字必须来自图像识别；不生成位号，不猜型号，不做单位或字母大小写改写。
- YOLO 只运行一次，重用相同的 fused proposals。实验断言框、类型、数量完全一致。
- Component 专用补读不会替换提供给其他阶段的文本及角色列表。当前只完成器件专项验证，未验证下游综合效果。
- 修复 RapidOCR 每次调用的开关状态：完整 OCR 显式恢复 `use_det=True,use_rec=True`，防止仅识别调用影响下一次检测。
- 集成到 Pin L45 分支时遵守 Context.resources 仅保存设备、模型路径、缓存路径的契约，不存放 OCR 会话。局部 OCR 实验按 Stage 自行持有会话；默认 V2 不额外创建局部 OCR 会话。

## 独立评测

```powershell
python tools/evaluate_component_local_joint.py `
  --dataset '公开数据集/200_train_cases' `
  --output runs/component_local_joint_dev150_20261016_v1 `
  --variants baseline,local,local_joint,joint_only --evaluate
```

脚本逐一验证 `0001`—`0150` 精确目录，不枚举保留案例。默认使用全部 150 例，可用 `--cases 0001,0005` 指定开发子集；重复编号、越界编号、目录缺失或重定向会停止。日志记录实际案例编号。

所有选中案例、所有方案的器件预测保存并计算哈希后，才打开对应开发集 GT。只调用 `evaluate_v2.component_counts/component_key_map/prf`，不调用全项目 `evaluate_case`。这是本地代理评分，非官方评分器。

评分规则沿用现有本地实现：普通器件 key、type 严格匹配，bbox 最大坐标差不超过 20；box 额外检查 Name，r/c 额外检查 value；GND 按空间对应。报告含逐案例与宏平均、微平均、增加/减少分的案例数。YOLO 与全图 OCR 复用内容哈希缓存，因此时间主要用于比较追加处理开销，不能视为完整部署耗时。

当前默认选择与实验配置分离：`configs/component/local_ocr_v1.json`、`local_joint_v1.json`。是否启用以开发集实测为准，不因为加入算法就替换默认。

## 实测与选择

仅使用 0001—0150。沿用现有整页划分：编号可被 5 整除的 30 页作为内部检查，其余 120 页为开发部分。这两个部分都属于开发白名单，不能当作独立官方测试结果。0151—0200 和 10_GTcase 未使用。

| 方案 | 150 页宏平均 ComponentF1 | 正确器件数 | 内部 30 页宏平均 |
|---|---:|---:|---:|
| 原 `contextual_values_v2` | 0.396791 | 2315 | 0.394545 |
| 仅局部补读 | 0.396979 | 2307 | 0.393931 |
| 局部补读＋完整文字组关联 | 0.411599 | 2427 | 0.404003 |
| 仅完整文字组关联 | 0.410586 | 2432 | 0.405733 |
| **保留位号／型号，仅改数值组归属** | **0.410919** | **2435** | **0.406923** |

选择最后一项作为默认：较原版多 120 个正确器件，宏平均增加 0.014128；26 页提高、7 页下降，其余持平。虽然组合方案的全体宏平均略高，最后一项在内部检查、器件总正确数和追加耗时上更好，并避免了位号重分配的副作用。局部补读保留实验开关，未默认启用。

缓存 YOLO／全图 OCR 后，本机数值归属阶段平均新增约 0.036 秒／页；这是追加处理耗时，不是从零运行完整程序的时间。无新增权重或依赖安装。

证据目录：

- `runs/component_local_joint_dev150_20261016_v2/report.json`：四方案完整对照；与初版逐例分数一致。
- `runs/component_value_groups_ab_dev150_20261016/report.json`：最终数值归属拆分对照。
- 每个目录的 `predictions_sealed_before_GT.json`：150 页全部方案保存后再读取 GT 的预测哈希。
- `runs/component_shared_engine_smoke_20261016/report.json`：此前使用全新 OCR 缓存的三页连续图像核验；共享会话的实验实现未作为本次分支集成的运行接口。

上述均为本地代理分数，不能与比赛平台约 0.5 直接横向比较。官方评分脚本仍未提供。

## 参考资料与改造范围

1. [Automated Digitization of Engineering Schematics (2026)](https://doi.org/10.3390/electronics15173785)：参考多行文字链与文字—器件关联思路；本实现使用几何文字组，未训练论文的 GraphSAGE。
2. [SINA (2026)](https://arxiv.org/html/2607.01609v1)：参考分开处理文字提取和器件语义归属；未采用 GPT-4o，也不涉及其网络还原部分。
3. [PARSeq (ECCV 2022)](https://github.com/baudm/parseq)：后续可用作疑难短文本第二识别器；本轮未安装或接入。
4. [PaddleOCR recognition 文档](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)：英文专用 PP-OCRv5 为后续对照候选；本轮重用原有多语言 mobile ONNX 模型。
5. [工程图小文字漏检讨论 #13164](https://github.com/PaddlePaddle/PaddleOCR/discussions/13164)、[文字框截断讨论 #14787](https://github.com/PaddlePaddle/PaddleOCR/discussions/14787)：参考局部尺度与文字框扩边排查方向，未照搬帖中的参数。

上述文献分数不是本项目分数。本轮实现属于借鉴后的独立实验，不声称完整复现论文。
