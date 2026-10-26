# Wire palette 缺陷普遍性核查

## 范围与方法

- 使用获准开发集 D:/EDA/open_GT 0001–0013 的图像与对应标注；未读取其他数据。
- 复用样本 0002–0013 的 D:\EDA\EDA-NetLine-Isolated\docs\network\lines\experiments\open_gt_baseline_20261024_161500_retry1 预测/诊断及样本 0001 的 D:\EDA\EDA-NetLine-Isolated\docs\network\lines\experiments\real_image_ab_20261020_230608\pipeline_output_retry1\1 结果；只重建并运行 Wire 候选链，没有重新执行整条推理流水线或 Component/Pin 模型。
- GT bottom-origin 坐标转换为 OpenCV 坐标 y_cv=image_height-y_GT。每条 GT 直边按不大于 1 px 间隔采样；以 wire mask 2 px 内作为主覆盖定义，5 px 作为宽容参考。覆盖低于 80% 定义为本报告的明显不足阈值，均为诊断指标。
- 源码确认顺序：dark/overlap-hue 候选 → text/component suppression → page-frame connected-component drop → 轴向 MORPH_OPEN 与 line-support 连通域保留 → score gate（72%）且最多选两类 → 2x2 close/边框清除 → V3 terminal corridor restore → component barrier → retained frame guard。

## 汇总结论

- GT 边共 744 条；72% 下覆盖 <80%：2 px 判定 737 条，5 px 判定 728 条。
- 2 px 低覆盖边中，palette selection 是首次缺失原因的 706 条（95.8%），涉及 13/13 个样本。细分：72% 分数门槛 144 条（涉及样本 0002、0007、0009）；最多两个候选的上限 560 条。
- 其余首次缺失类别：{"line_width_or_morphology_filter": 23, "suppression_or_region_filter": 8}。未观察到 raw color candidates 完全缺失或 frame guard 首次删除的确认案例。
- 0.71 只改变 1/13 个样本，恢复 40 条低覆盖边；样本 0002 的重建 0.71 mask 与既有实验 PNG 逐像素一致，XOR=0。完整 mask 连通域由 43 变为 36；新增差分为 6370 px，其中 6370 px 在任意 GT 边 5 px 区域内，0 px 在该范围外。落在 GT 周边不等于全部为真阳性。
- all-edge macro 几何覆盖（每条 GT edge 等权）：72% 为 2 px=0.064、5 px=0.115；0.71 为 2 px=0.112。这不是官方评分。

## 逐样本统计

|样本|GT edge|edge-mean ≤2px|edge-mean ≤5px|<80% ≤2px|palette 选择漏检|分数门槛 / top-two|0.71 恢复|mask px 0.72→0.71|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0001|116|0.161|0.238|112|100|0 / 99|0|18992→18992|
|0002|40|0.051|0.112|40|40|40 / 0|40|5181→11551|
|0003|46|0.048|0.104|45|45|0 / 45|0|5199→5199|
|0004|59|0.043|0.089|59|58|0 / 58|0|4598→4598|
|0005|57|0.017|0.055|57|55|0 / 55|0|5425→5425|
|0006|86|0.022|0.059|86|83|0 / 83|0|6531→6531|
|0007|68|0.015|0.047|68|67|67 / 0|0|5404→5404|
|0008|50|0.050|0.123|50|48|0 / 48|0|4763→4763|
|0009|39|0.043|0.077|38|37|37 / 0|0|3470→3470|
|0010|82|0.032|0.074|82|76|0 / 76|0|7300→7300|
|0011|24|0.066|0.103|24|22|0 / 22|0|5039→5039|
|0012|28|0.092|0.127|28|28|0 / 27|0|11790→11790|
|0013|49|0.151|0.215|48|47|0 / 47|0|16872→16872|

## 阶段归因

|首次导致覆盖不足的阶段|GT edge 数|
|---|---:|
|top_two_cap|560|
|score_ratio_gate|144|
|line_width_or_morphology_filter|23|
|suppression_or_region_filter|8|
|palette_selection_mixed_candidates|2|

## 逐边证据示例

- 0002 net_1/edge_1, length=128.0 px: final coverage(2px)=0.054; excluded hue_45_75 score=392.206 vs gate=393.817, candidate coverage=1.0, exclusion=score_ratio_gate; 0.71 coverage=1.000.
- 0003 net_1/edge_1, length=97.0 px: final coverage(2px)=0.082; excluded hue_45_75 score=274.483 vs gate=397.475, candidate coverage=1.0, exclusion=top_two_cap; 0.71 coverage=0.082.
- 0005 net_1/edge_1, length=55.0 px: final coverage(2px)=0.054; excluded hue_45_75 score=329.843 vs gate=361.207, candidate coverage=1.0, exclusion=top_two_cap; 0.71 coverage=0.054.
- 0010 net_1/edge_2, length=20.0 px: final coverage(2px)=0.381; excluded hue_45_75 score=367.609 vs gate=493.702, candidate coverage=1.0, exclusion=top_two_cap; 0.71 coverage=0.381.

## 复现质量与限制

- 样本 0002 的完整保存 scene.json 和 wire_mask.png 可逐像素复核；由导出 result.json 与 diagnostics.json 重建的当前注册 Wire stage mask 与保存 PNG 一致。
- 0001、0003–0013 未保存 scene/mask。使用导出 components/pins、text_roles、pin_events 重建 Scene；各样本 palette 选择与保存诊断一致，候选分数最大绝对差见 results.json，通常来自导出 bbox 的 0.001 精度舍入。最终像素数与诊断相同，只有 0010 相差 3 px；因此 0010 的少量边界归因应视为近似。重放不是历史 mask 逐像素比较。
- mask 中心线覆盖是几何诊断，不能证明电气连通，也不是官方 Wire F1。GT 外的新像素不能直接视为误检，因为标注可能不穷尽所有图上导线。

## 结论与优先级

- 72% palette 选择缺陷不是样本 2 的孤例：候选阶段已覆盖但最终选择漏掉的 GT 边在 13/13 样本均出现；该类占明显低覆盖边的大多数。分数门槛和 top-two cap 必须分开处理，不能只把 72% 下调少量就认为解决。
- 0.71 有跨样本可复现的改善证据，但改善仅来自样本 0002：40 条边恢复、+6370 mask px；其他 12 样本无变化。适合保留为隔离候选/对照，不建议直接入生产或改默认。
- 次级问题是 line-support 过滤和 text/component suppression：各有少量 GT 边在对应阶段失去覆盖；当前未发现 frame guard 首次删除目标的证据。建议先设计对 palette 多样性的候选选择策略，再逐例复核这些次级边。

未修改 Wire 生产代码、默认配置或 NetLine 后端；没有重跑模型，没有 Git 写操作。

逐样本与逐边机器结果：D:\EDA\EDA-NetLine-Isolated\docs\network\lines\experiments\wire_palette_prevalence_20261024_202141\results.json
复现脚本：D:\EDA\EDA-NetLine-Isolated\docs\network\lines\experiments\wire_palette_prevalence_20261024_202141\audit_wire_palette_prevalence.py
