# V3 与 V4 BEST 对比

`OFFICIAL_SCORE = FALSE`。范围为最新版官方公开集 0001～0150；0151～0200 和 Golden Case 未用于开发。

| 指标 | V3 | V4 BEST | 变化 |
|---|---:|---:|---:|
| ComponentF1 macro | 0.1888 | 0.3110 | +0.1222 |
| Symbol type+bbox F1 micro | 0.3052 | 0.6667 | +0.3614 |
| BBox F1 micro | 0.3589 | 0.7702 | +0.4114 |
| PinF1 macro | 0.0583 | 0.1025 | +0.0441 |
| NetHypergraphF1 macro | 0.1862 | 0.1112 | -0.0749 |
| NetLineF1 macro | 0.0151 | 0.0193 | +0.0042 |
| PinPairF1 macro | 0.0151 | 0.0092 | -0.0058 |
| singleton ratio | 92.19% | 80.62% | -11.58pp |
| empty-edge ratio | 44.95% | 12.21% | -32.75pp |
| unattached-pin ratio | 55.79% | 22.00% | -33.79pp |
| 本地非官方总分 | 13.79 | 15.98 | +2.19 |

V4 的主要收益来自 symbol body bbox/type。Net 与 PinPair 下降说明下游并没有自动适应新 Component：严格位号只命中 1,088/3,498 个可观察 key，冻结的 Pin/Topology 又对新增框生成了大量关系。PinPair 真阳性仅从 152 增至 161，预测 pair 却从 1,519 增至 7,053，因此 precision 明显下降。

YOLO 权重训练过 150 例中的 120 例。用于判断 detector 泛化的固定 30 例无训练重叠切片中，V4 BEST 的 symbol type+bbox F1 为 0.6873，严格 Component macro F1 为 0.3041。
