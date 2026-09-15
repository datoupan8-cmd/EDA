# 同组元件 V2 与 Component V4 对比

## 结论

同组 V2 **适合融合，但应按模块选择性融合**。最有价值的是继续训练后的 YOLO 权重。它可以直接替换 V4 当前的旧 YOLO 权重，同时保留 V4 的 RapidOCR、全局文字关联、不可观测字段处理和下游 Pin/Wire/Topology。

不建议用同组 V2 的整套识别流程覆盖 V4，因为其中仍有自动生成位号、0057 专用 MOSFET 模板和宽松大小写评测等与当前比赛数据契约不一致的逻辑。EasyOCR 已经按用户确认的“每块均 OCR”方式接入，但只作为 RapidOCR 的补充 token 来源。

## 公平 A/B 方法

固定 V4 的 Component–Text Association、Pin、Wire、Topology 和 evaluator，只替换 YOLO 权重：

- 旧权重：`component_yolo11n_baseline_best.pt`
- 新权重：`component_yolo11n_continue_20260914_170933/weights/best.pt`
- 数据：最新版官方公开集 0001～0150
- 封存数据：0151～0200 和 10 Golden 未使用
- dev30：编号为 5 的倍数，未参与新权重训练

因此下面的主要差异可以归因于新权重，而不是 OCR 或下游逻辑变化。

## 主要结果

| 指标 | V4旧权重 | 同组V2新权重 | 变化 |
|---|---:|---:|---:|
| Component macro F1 | 0.3110 | 0.3762 | +0.0652 |
| Symbol type+bbox micro F1 | 0.6667 | 0.8137 | +0.1470 |
| BBox micro F1 | 0.7702 | 0.8768 | +0.1066 |
| Type accuracy | 0.8496 | 0.9249 | +0.0753 |
| Designator association accuracy | 0.5828 | 0.7219 | +0.1391 |
| Value accuracy | 0.5979 | 0.5880 | -0.0099 |
| False positive / image | 10.28 | 5.97 | -4.31 |
| Duplicate component pairs | 238 | 55 | -183 |
| Detection miss | 1327 | 661 | -666 |
| Pin macro F1 | 0.1025 | 0.1719 | +0.0695 |
| NetHypergraph macro F1 | 0.1112 | 0.1506 | +0.0394 |
| NetLine macro F1 | 0.0193 | 0.0263 | +0.0071 |
| PinPair macro F1 | 0.0092 | 0.0171 | +0.0079 |
| 本地非官方诊断总分 | 15.98 | 21.12 | +5.14 |

dev30 无训练重叠结果：

| 指标 | V4旧权重 | 同组V2新权重 | 变化 |
|---|---:|---:|---:|
| Component macro F1 | 0.3041 | 0.3580 | +0.0539 |
| Symbol type+bbox micro F1 | 0.6873 | 0.8031 | +0.1158 |
| BBox micro F1 | 0.7830 | 0.8660 | +0.0830 |
| Type accuracy | 0.8641 | 0.9265 | +0.0624 |
| Designator association accuracy | 0.5303 | 0.6596 | +0.1293 |
| False positive / image | 10.40 | 5.60 | -4.80 |
| Duplicate component pairs | 48 | 6 | -42 |
| 本地非官方诊断总分 | 14.82 | 19.91 | +5.10 |

新权重在 dev30 上仍然提升，说明结果不是仅靠记住训练样本。后续做模型方案定稿时仍建议使用 5-fold OOF 验证。

## 代码层差异

### 同组 V2 的有效改进

1. 继续训练 YOLO11n，加入确定性种子、early stopping、multi-scale 和 checkpoint 类别顺序校验。
2. value 规则覆盖电阻、电容、电感、晶振频率、保险丝电流、电压等物理量。
3. value 分配改为全局一对一匹配，避免一个 OCR value 同时分给多个元件。
4. 扩展电阻位号前缀。官方 0001～0150 中实际出现 `RC`、`RES`、`RP`、`RT`、`RV`、`RVC`。
5. 大小写无关地清理重复 designator。
6. 增加相应单元测试。

### V4 已经具备、无需重复迁移的能力

1. designator 的全局 Hungarian 一对一匹配。
2. value 的全局一对一匹配。
3. OCR token 角色分类。
4. `UNRESOLVED_xxxx`，避免把不可观测内部 ID 当 OCR 目标。
5. proposal source、文字关联和下游诊断接口。
6. RapidOCR 与现有 Pin/Wire/Topology 接口。

### 不应进入 V4 主线的逻辑

1. `assign_named_fallback_designators()`：看不到位号时自动生成 U1/Q1 等，可能与 GT 真实位号相反。
2. `detect_mosfet_symbols()`：包含针对训练案例 0057 的固定 29×20 模板，存在明显过拟合风险。
3. 整体切换到 EasyOCR：同条件 A/B 中 EasyOCR-only 总分仅 15.45，明显低于 RapidOCR-only 的 19.84。
4. 直接采用大小写不敏感 evaluator：官方字符串匹配是否忽略大小写尚未确认，只能作为辅助诊断。
5. 整体覆盖 V4 的 Name association：V4 先前消融中 Name 阶段没有稳定提升。

## 推荐融合顺序

1. **直接融合**：将新权重加入 V4，保留旧权重作为回滚项，并将新权重设为推荐默认值。
2. **改造后融合**：把 `RC/RES/RP/RT/RV/RVC` 等真实前缀纳入 V4 token 分类和 type compatibility，增加官方样本测试。
3. **改造后 A/B**：扩展 value 单位兼容到晶振、保险丝、电源等类型；保留当前关联算法，避免整段复制。
4. **未来训练脚本融合**：加入 checkpoint 类别顺序校验、seed、patience 和 multi-scale 参数。
5. **明确关闭**：自动位号、0057 模板、EasyOCR 单独替换 RapidOCR 和宽松主评测。

## EasyOCR 融合后的补充结论

在同组 V2 新权重固定后，又对无训练重叠的 dev30 做了 OCR 消融。RapidOCR 1.4.4 全图总分 19.84，EasyOCR 全分块单独运行 15.45，两者合并达到 21.00。完整 0001～0150 的最终 V4.1 达到 ComponentF1 0.3810、总分 21.63，并通过 150/150 contract validation。因此保留同组“分块 OCR”思想，但采用“RapidOCR 全图 + EasyOCR 每块 + token 去重”的组合，而没有照搬同组整条语义生成链。

## 仍存在的问题

- 混合 OCR 后 designator OCR micro F1 从 0.5101 提高到 0.5532，但仍有接近一半可观察位号没有正确进入 token 集。
- 最终 value accuracy 为 0.5431，低于原 V4 的 0.5979；扩展规则虽在同 OCR 条件下有正收益，但新增 OCR token 仍会造成错误 value 竞争。
- 虽然 Pin、Net 和 Line 随元件框变准而提升，但它们仍是主要瓶颈。
- 分来源看，KiCad 的 PinPairF1 下降，`other` 的 NetLineF1 小幅下降，融合后还应保留逐来源回归保护。

## 自动测试

同组 V2 共发现 41 项测试：40 项通过，1 项因当前检查环境缺少 `cv2` 而无法执行；该项正是 0057 专用 MOSFET 模板测试。其余数据边界、NMS、字段关联、评测和 schema 测试均通过。
