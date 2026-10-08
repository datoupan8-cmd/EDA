# Changelog

## Modular Refactor — behavior preserving

- Added stable Text, Component, Pin Localization, Pin Semantics, Wire, Topology, and Submission Stage contracts.
- Added an explicit Registry and centralized JSON pipeline configuration.
- Added `--orchestrator modular` while retaining the Legacy entry as the default.
- Added contract tests, full pipeline parity tests, source-integrity checks, and 150-case prediction parity evidence.
- Kept model weights and algorithm source files unchanged; 150/150 predictions and all diagnostic F1 metrics are identical.

## V4.1 ComponentFusion

- 默认权重升级为同组 V2 的继续训练 YOLO11n；保留旧权重供回滚。
- 接入 EasyOCR 640 像素全分块、96 像素重叠、2 倍放大和跨块去重；默认与原 RapidOCR 合并。
- 增加 `rapidocr`、`easyocr_tiled`、`hybrid`、`selective_local` 四种可切换 OCR 模式。
- 实现用户提出的低置信器件局部二次 OCR：扩展器件区域、放大、CLAHE 增强，再重新执行全局文字关联。
- 扩展真实位号前缀和按类型约束的 Value 规则；拒绝把 `REF5040` 一类型号误作电阻位号。
- 迁移同组 V2 的数据边界、最新官方 target 直读、切片训练数据生成、随机种子、early stopping、multi-scale 和 checkpoint 类别顺序检查。
- 30 例无训练重叠消融选择 `hybrid` 为主线；`selective_local` 虽更快且 Component 略高，但总分、Line 和 PinPair 较低，因此保留为可选项。
- 修复 RapidOCR/YOLO 缓存指纹：分别纳入实际包版本、OCR 配置和权重完整 SHA256，防止不同版本错误共用结果。
- 修复干净解压包首次运行时 EasyOCR 缓存目录可能不存在的问题，并新增首次运行回归测试。
- 将相对 EasyOCR 模型目录固定解析到项目根目录，使 `main.py` 从项目目录外启动时也能找到随包模型。
- 将 EasyOCR 缓存文件名缩短为 96 位摘要，避免深层解压目录触发 Windows 路径长度限制。
- 对 YOLO 缓存应用相同的 96 位短文件名和写入前目录检查。
- 最终 0001～0150 达到 ComponentF1 0.3810、本地非官方总分 21.63，并通过 150/150 strict contract validation。

## V4 ComponentFusion

- 保留 V3 的 Scene/schema、OCR token role、Pin、Wire、Topology、submission 和 diagnostics。
- 接入同组 YOLO11n 元件检测权重，并增加 tiled inference、tile→global 坐标转换、缓存与 class-aware NMS。
- 新增 `ComponentProposal` 适配层，避免同组数据结构污染 V3 schema。
- 新增 YOLO/geometry proposal pool 与基于 IoU、中心距离、type 的空间去重。
- 新增 type/designator compatibility 和全局 Hungarian 位号一对一分配。
- 将 designator、Name、value 分开处理；YOLO 不生成语义字段。
- 禁用按阅读顺序自动生成 U1/R1 等位号。
- 移除同组代码中面向单个 Datasheet 尺寸的 MOSFET template 主线入口。
- 保留 B～F/BEST 可切换消融；默认 BEST 关闭负收益的 Name 和原几何回退。
- 每个 Component 在 diagnostics 中保留 proposal/designator/name/value 来源。
- 新增 Component 专项 evaluator、150 例逐案例结果、来源拆分、错误分类和 13 项 V4 测试。
- 保持 0151～0200 与 Golden Case 封存，开发工具硬拒绝越界。
