# Component前端论文调研与实验（2026-10-13）

本轮默认Component已更新为`contextual_values_v2`，保留`text_enriched_v1`和`peer_latest`作为对照。开发、规则设计和实验只使用0001–0150，没有使用0151–0200或10_GTcase。新版尚未在保留集验证。

## 论文与采用范围

- [SINA: A Circuit Schematic Image-to-Netlist Generator Using Artificial Intelligence，2026](https://arxiv.org/abs/2601.22114)：组合YOLOv11、EasyOCR、连通域和VLM；文字抽取与语义归属分开处理。本项目已有YOLO/OCR和全局匹配，本轮从其“结合上下文确定文字归属”的方向出发，实现本地、确定性的OCR冲突处理。不是复刻其VLM，也没有采用论文的成绩作为本项目成绩。
- [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection，2022](https://arxiv.org/abs/2202.06934)：重叠切片、放大后推理并合并检测，用于小目标识别。本项目已采用原尺寸切片，本轮进一步测试2倍尺度、并要求未匹配的可见位号支持新增器件。固定30例的误报抵消了收益，保留为实验代码，不接入默认版本。
- [Enhancing Circuit Diagram Understanding via Near Sight Correction Using VLMs，ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/MMFM/html/Kulkarni_Enhancing_Circuit_Diagram_Understanding_via_Near_Sight_Correction_Using_VLMs_ICCVW_2025_paper.html)：先显式标记连接关键点，再交给VLM理解连接。该方向更适合后续拓扑工作，本轮没有作为Component改进依据。

## 已接入的改动

两个OCR引擎可能把同一块印刷文字分别读成`R3`和`33`。旧合并过程只去重相同内容，`33`仍可被分配成阻值。新规则只在裸数字与更高置信度的位号/信号文字重叠时取消其数值候选资格，重叠阈值沿用现有OCR去重的IoU 0.3，文字证据置信度要求至少0.85。另单独过滤芯片框内部的裸整数；框外的整数阻值和显式带单位的值保留。

出现这些冲突时，只重新执行原数值归属，并继续应用V1的型号与电容单位缩写补读。明确的位号与数值合并标签仍绑定原器件。Component返回给Pin的原始文字角色不改；器件数、位号、类型、bbox、body_bbox及端点候选不改。

## PNG前端对照结果

固定0001–0150，全部成功。实际执行Text/Component，使用原有按内容和模型签名校验的开发缓存。所有A/B预测保存后才读取150份开发标注。评分使用现有`evaluate_v2.py`，属于Windows本地非官方Component评测，不是官方Docker评分器，也没有重跑四项完整评测。

| 指标 | 原默认V1 | 当前V2 |
|---|---:|---:|
| Component宏平均F1 | 0.381383293 | 0.382541363 |
| Component微平均F1 | 0.373160983 | 0.375031890 |
| 正确器件TP | 2194 | 2205 |
| 输出器件数 | 5679 | 5679 |
| 标注器件数 | 6080 | 6080 |

9例改善、0例回退、141例相同。收益较小；这是开发集结果，尚不能据此判定保留集收益。

最初数字冲突消融中，仅重叠读数处理增加10个TP；仅芯片内部裸数字过滤增加1个TP；组合增加11个TP。注册后的默认版本与该组合在150例的全部预测逐字段相同，V1对照也与上一轮150例预测逐字段相同。源代码、配置和YOLO权重哈希均已校验。

结果与复现证据：

- `runs/component_contextual_values_dev150_20261013/report.json`：数字冲突消融。
- `runs/component_contextual_values_registry_dev150_20261013/report.json`：注册版本PNG前端A/B。
- 同目录`verification.json`：150例预测一致性、哈希、案例范围与改善/回退统计。
- `runs/component_contextual_values_default_smoke_0006_20261013/report.json`及`verification.json`：0006完整默认流程运行通过，Component字段与前端结果完全一致，GT打开次数0。
- `runs/component_contextual_values_final_tests_20261013.txt`：300项维护测试全部通过。初次全套测试使用了没有PyTorch的Python 3.14，相关用例无法导入；最终测试已使用实际程序的Python 3.12/PyTorch环境。维护测试发现仍跳过只有旧缓存、没有`__init__.py`的`tests/wire`目录，实际维护测试未被删减。

## 未采用的实验

| 实验 | 范围 | 原宏F1 | 实验宏F1 | 结论 |
|---|---|---:|---:|---|
| 2倍切片补检，带可见位号约束 | 固定30例：0005、0010、…、0150 | 0.366935893 | 0.366834281 | 新增5框、1个正确匹配；1例提升、2例回退，未采用 |
| 在全局匹配时按唯一位号分组 | 0001–0150 | 0.381383293 | 0.381054370 | TP减少2个；1例提升、3例回退，未采用 |
| 位号分组加数字上下文 | 0001–0150 | 0.381383293 | 0.382212440 | 有3例回退，弱于仅数字上下文，未采用 |

报告分别为`runs/component_slice_recovery_dev30_20261013/report.json`、`runs/component_unique_designators_dev150_20261013/report.json`。这些候选保留为开发实验，未注册为默认生产策略，没有按案例编号设置例外。

## 使用与复现

当前默认入口：

```powershell
python tools/run_team_pipeline.py --image D:/EDA_Data/sample.png --output runs/component_v2_sample --device cuda
```

V1对照入口：

```powershell
python tools/run_team_pipeline.py --config configs/component/team_text_enriched_v1.json --image D:/EDA_Data/sample.png --output runs/component_v1_sample --device cuda
```

复现V1/V2的开发集Component前端A/B（默认白名单0001–0150）：

```powershell
python -m pcb.component.experiments.run_frontend_ab --dataset D:/EDA_Data/200_train_cases --output runs/component_v1_v2_new --baseline-config configs/component/team_text_enriched_v1.json --candidate-config configs/component/team_contextual_values_v2.json --device cuda
```

缺失案例、重复输入编号或超出开发范围均停止，不扩大范围。正式使用仍通过完整团队入口；前端实验输出的空Pin/Net字段只用于Component评分。
