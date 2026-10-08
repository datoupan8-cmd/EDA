# Component V4 A/B 实验

OFFICIAL_SCORE = FALSE；只使用最新版官方 0001～0150。

| 阶段 | 说明 | ComponentF1 | type+bbox F1 | PinF1 | NetHypergraphF1 | NetLineF1 | PinPairF1 | 总分 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | V3 baseline | 0.1888 | 0.3052 | 0.0583 | 0.1862 | 0.0151 | 0.0151 | 13.79 |
| B | YOLO + greedy designator | 0.2539 | 0.6667 | 0.0992 | 0.1108 | 0.0192 | 0.0091 | 14.17 |
| C | + global designator | 0.2543 | 0.6667 | 0.1025 | 0.1112 | 0.0193 | 0.0092 | 14.27 |
| D | + Name | 0.2402 | 0.6667 | 0.1025 | 0.1112 | 0.0193 | 0.0092 | 13.85 |
| E | + value | 0.2969 | 0.6667 | 0.1025 | 0.1112 | 0.0193 | 0.0092 | 15.55 |
| F | + raw geometry fallback | 0.1764 | 0.4078 | 0.0516 | 0.0713 | 0.0121 | 0.0043 | 9.20 |
| BEST | C + value | 0.3110 | 0.6667 | 0.1025 | 0.1112 | 0.0193 | 0.0092 | 15.98 |

30 个没有参与 best.pt 训练的固定 dev 案例中，BEST 的 type+bbox F1 为 0.6873，严格 Component macro F1 为 0.3041。

D 相比 C 下降，说明当前 Name 过滤仍会把 pin/signal/model token 错配到 box；F 的几何候选精度仅约 4%，造成大量误报。最终默认选择 BEST：保留 C 的全局位号匹配和 E 中有效的 value，关闭 Name 与几何回退。