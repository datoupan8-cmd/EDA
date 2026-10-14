# 器件文字字段改进：2026-10-13

本阶段完成时默认Component为`text_enriched_v1`，基础检测/OCR仍使用原`peer_latest`。此后默认版本已更新为`contextual_values_v2`，最新结果见[Component论文调研与实验](COMPONENT_RESEARCH_20261013.md)。本阶段没有训练新模型，也没有修改公共Pipeline、Schema、Submission、评分器或Pin/Wire/网络算法。

## 改动

- `BEST`原流程不执行芯片型号赋值。新版本只对尚无可信型号的对象使用可见OCR文字补型号；引脚名称不作为通用型号候选，短型号须有原337项公开元件库支持。公开库候选仅接受至少3脚的条目，避免供电标签和普通两端符号混入。
- 支持电容图中直接印出的`27p`、`100n`、`4.7u`等带单位缩写，保留印刷文本，不把单位缺失的数字猜成电容量。
- 已有型号和数值保留；数值补读遵守原阶段已有的文字归属，不能把一个已用于电阻等器件的文字再次给电容。
- 不改器件数量、位号、类型、bbox、body_bbox或端点候选。原几何匹配权重与接受阈值保留，没有阈值扫描。

## 开发集结果

仅0001～0150。固定30例（0005、0010、…、0150）筛选后，从PNG实际执行Text/Component的A/B流程；允许原有按内容与模型签名校验的缓存。两组全部预测保存后，才打开150份开发标注。这里是Component前端评测，不是Pin/Net的全量重跑，也不是官方平台分数。

| 指标 | 原方案 | 新方案 |
|---|---:|---:|
| Component宏平均F1 | 0.370137087 | 0.381383293 |
| Component微平均F1 | 0.363126116 | 0.373160983 |
| 正确器件TP | 2135 | 2194 |
| 输出器件数 | 5679 | 5679 |

38例改善、3例回退、109例相同。回退案例保留在报告中，没有按案例编号设置例外。采用较保守的型号证据规则；早期宽松规则宏平均0.381908535，但有7例回退，最终版本为0.381383293，回退减少到3例。

完整结果：`runs/component_text_enrichment_evidence_fresh_dev150_20261013/report.json`。报告含全部案例编号、源文件/评分器哈希和YOLO权重哈希，实际成功150例、失败0例。

原图完整程序单图检查0001通过严格输出校验，且最终Component字段与前端A/B逐字段相同；GT打开次数0。0002也完成过完整程序检查。维护测试最终结果见`runs/component_text_enrichment_final_tests_20261013.txt`。

测试发现时跳过仓库中仅有旧字节码缓存、没有`__init__.py`的`tests/wire`空目录；运行的其余维护测试与`tools/run_team_tests.py`的列表相同。没有删除缓存或改写测试发现工具。公共默认选择断言只同步新的Component版本名，旧算法的冻结输出期望未改。

## 使用与对照

团队默认入口：

```powershell
python tools/run_team_pipeline.py --image D:/EDA_Data/sample.png --output runs/component_sample --device cuda
```

原方案对照：

```powershell
python tools/run_team_pipeline.py --config configs/component/team_peer_latest.json --image D:/EDA_Data/sample.png --output runs/component_baseline --device cuda
```

复现前端开发集A/B：

```powershell
python -m pcb.component.experiments.run_frontend_ab --dataset D:/EDA_Data/200_train_cases --output runs/component_ab_new --device cuda
```

开发脚本默认白名单0001～0150并记录实际案例编号；缺少案例、重复编号或非法范围会停止。保留集未使用。前端A/B输出的空Pin/Net字段仅是开发中间结果，正式程序输出使用团队完整入口。
