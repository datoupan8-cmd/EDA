# 论文思想到 V4.1 代码的映射

| 来源 | 借鉴思想 | V4.1 代码 | 实验结果 |
|---|---|---|---|
| Netlistify | Component detection 与文字、连接任务解耦 | `component_detector_yolo.py`、`component_text_assignment.py` | 原 V4→V4.1：type+bbox F1 0.6667→0.8137 |
| Netlistify | Symbol detector 只回答 bbox/type，语义由独立 OCR 关联 | `ComponentProposal` adapter + V3 text roles | 避免 YOLO 伪造 R1/U1；strict Component 仍受 OCR 限制 |
| 同组工程的分块思想 | 大图分块后恢复全图坐标 | `ocr_backends.py::TiledEasyOCR` | dev30 位号 OCR micro F1 0.5101→0.5532；总分 19.84→21.00 |
| Topology-Consistent | point-first、wire evidence、component suppression | 原 V3 `pin_detection.py`、`wire_v3.py`、`topology_v3.py`，本轮冻结 | 用于观察新 bbox 对下游的真实影响，未把 CC 当作自动正确 net |
| HAWP | junction/keypoint provider 接口 | 原 V3 可选接口 | NOT USED IN V4 MAINLINE |
| Netlistify Transformer | learned connectivity | 无 | NOT USED IN V4 MAINLINE |

本轮论文价值主要体现在任务边界：YOLO 只负责“哪里有什么 symbol”，OCR 负责“它叫什么”，Pin/Wire/Topology 保持独立。HAWP 与 Netlistify Transformer 没有进入 V4.1 主线，避免产生无法归因的模型堆叠。
