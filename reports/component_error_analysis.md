# Component V4.1 错误分类

| 错误类型 | 数量 |
|---|---:|
| Designator OCR Error | 1647 |
| False Positive | 896 |
| Detection Miss | 661 |
| Designator Association Error | 523 |
| Value Error | 435 |
| Type Error | 372 |
| BBox Error | 263 |
| Duplicate Prediction | 55 |

当前首要错误是 designator OCR 漏检，其次是 False Positive、Detection Miss 和 Designator Association Error。type+bbox micro F1 已达到 0.8137，但可观察位号 OCR micro F1 只有 0.5292；被识别出的位号中 71.75% 分到正确 symbol。检测和关联都改善后，Value Error 从 306 增到 435，成为明确回退点。

V4.1 下游相对原 V4 已改善，但 PinF1 0.1751、PinPairF1 0.0181 仍很低，singleton net 比例约 79.29%。下一轮应按 YOLO body bbox 重做 terminal search，并在每个 terminal 的方向窗口内识别 pin number/pinname；不能直接继续堆 Component 几何候选。
