# 已测试但未进入 V4.1 主线的方法

## EasyOCR-only：关闭

在同一无训练重叠 dev30 上，EasyOCR 全分块单独运行总分 15.45，低于 RapidOCR 1.4.4 全图的 19.84。EasyOCR 保留为 Hybrid 的补充来源，不单独替换 RapidOCR。

## 选择性局部二次 OCR：保留为速度选项

该路径实现了初步关联后筛选缺失/低置信元件，再扩框、放大和增强 OCR。它的 ComponentF1 0.3646 略高于全分块 Hybrid 的 0.3639，但总分 20.82 低于 21.00，尤其 Line 和 PinPair 回退。因此 `selective_local` 可手动选择，默认仍是 `hybrid`。

## Name association：关闭

Stage C → D 后，ComponentF1 从 0.2543 降至 0.2402，总分从 14.27 降至 13.85。原因是 box 周围的 pin name、signal 和 model token 仍会被误分成 Component Name。代码和调试图保留，默认 BEST 不启用。

## 原 V3 geometry fallback：关闭

Stage E → F 后，ComponentF1 从 0.2969 降至 0.1764，总分从 15.55 降至 9.20，平均每图 false positive 从 10.28 增至 59.51。F 中全部 geometry proposal 的独立匹配 precision 约 4.15%；其中 large-box 与 capacitor fallback 尤其不稳定。实现保留用于后续逐类型重做，默认 BEST 不启用。

## 自动生成 designator：删除

同组代码会按阅读顺序为未识别元件生成 U1、Q1 等位号。图片没有文字证据时，这会制造高置信错误 key，因此 V4 使用 `UNRESOLVED_NNNN` 内部占位并在 diagnostics 中标记，不把占位符声称为 OCR 成功。

## 单案例 MOSFET template：不接入

同组源码包含针对 0057 Datasheet 约 29×20 像素图形的固定模板。它没有跨图泛化证据，存在明显过拟合风险，因此没有进入 proposal pool。

## HAWP / Netlistify Transformer：本轮未使用

本轮只修改 Component/OCR 前端。HAWP junction、Netlistify learned connectivity 和新的 Pin/Wire/Topology 模型均未启用，以保证 A/B 能归因到前端改动。
