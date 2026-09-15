# 0002：原 V4 与 V4.1

`OFFICIAL_SCORE = FALSE`。这里对比原 V4 BEST 与最终 V4.1；没有为 0002 添加任何 case-specific 规则。

| 指标 | 原 V4 | V4.1 | 变化 |
|---|---:|---:|---:|
| ComponentF1 | 0.6957 | 0.4167 | -0.2790 |
| Symbol type+bbox F1 | 0.8696 | 0.9167 | +0.0471 |
| BBox F1 | 0.9565 | 1.0000 | +0.0435 |
| PinF1 | 0.3404 | 0.4082 | +0.0677 |
| NetHypergraphF1 | 0.2979 | 0.4082 | +0.1103 |
| NetLineF1 | 0.1026 | 0.2118 | +0.1092 |
| PinPairF1 | 0.1200 | 0.1304 | +0.0104 |
| 本地非官方总分 | 40.83 | 39.11 | -1.72 |
| predicted components | 11 | 12 | +1 |
| GT components | 12 | 12 | 0 |
| predicted pins | 19 | 21 | +2 |
| singleton nets | 5 | 0 | -5 |
| empty-edge nets | 0 | 0 | 0 |
| unattached pins | 0 | 0 | 0 |

V4.1 在 0002 上找全了 12 个 bbox，并改善了 Pin、Net、Line 和 PinPair；严格 ComponentF1 却下降，说明部分正确 symbol 获得了错误的 key/value 语义。单例回归不能推翻 150 例整体提升，但它明确指出下一步需要收紧局部位号和 Value 关联。
