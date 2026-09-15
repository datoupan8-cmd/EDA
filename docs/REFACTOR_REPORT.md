# 行为保持型模块化重构报告

## 完成结果

工程已从“`main.py` 与 `vision_v4.py` 直接选择并串联具体算法”重构为“公共 Pipeline + Stage 契约 + Registry + 集中配置”。原 Legacy 入口继续存在且仍为默认值；新 Modular 入口通过 `--orchestrator modular` 启用。

本轮没有改变算法、权重、OCR、阈值、正则、匹配代价、执行阶段顺序、坐标语义、submission schema 或 evaluator。

## 建立的 Stage

1. Text：当前 OCR 对象 → texts。
2. Component：image/texts → components、roles、最终 texts。
3. Pin Localization：components → terminal tip/base/side。
4. Pin Semantics：terminals → Pin number/name。
5. Wire：image/Scene → wire mask。
6. Topology：pins/wire → Scene nets。
7. Submission：Scene → 官方 JSON 并严格校验。

每个 Stage 的详细输入输出见 `MODULE_CONTRACTS.md`。

## Parity 验证

验证使用最新版官方开发集 `0001～0150`。`0151～0200` 和 `10_GTcase` 未读取、未评测、未调参。

| 验证项 | 结果 |
|---|---:|
| 原基线 prediction 快照 | 150 |
| Modular 成功运行 | 150/150 |
| 最终 JSON 逐字段完全一致 | 150/150 |
| prediction 差异案例 | 0 |
| 去除计时/cache 状态后的 diagnostics 语义一致 | 150/150 |
| 严格 submission contract | 150/150 |
| 自动测试 | 71/71 |

Prediction 比较同时检查字典键、插入顺序、列表顺序、数值和字符串。两个目录的每个 `result.json` SHA-256 也一致。

运行时间和 `cache_hit/cached_tile_count` 会因首次运行与缓存冷暖不同而变化，因此不作为算法 parity 字段；除此之外 diagnostics 的算法决定与结构一致。

## 重构前后指标

以下为同一 `evaluate_v2.py` 在同一150例上的本地非官方诊断分数。

| 指标 | Legacy V4.1 | Modular | 差值 |
|---|---:|---:|---:|
| Component macro F1 | 0.3809735754 | 0.3809735754 | 0 |
| Pin macro F1 | 0.1751359894 | 0.1751359894 | 0 |
| NetHypergraph macro F1 | 0.1578951602 | 0.1578951602 | 0 |
| NetLine macro F1 | 0.0299237859 | 0.0299237859 | 0 |
| PinPair macro F1 | 0.0180888025 | 0.0180888025 | 0 |
| FinalScore | 21.6331754610 | 21.6331754610 | 0 |

`OFFICIAL_SCORE = FALSE`。

## 算法与权重完整性

重构前记录的18个关键文件中，只有负责新增 CLI 和 Modular 入口的 `main.py` 发生预期变化。下列所有算法文件 SHA-256 与重构前相同：

- schema、coordinates、submission
- Component V4、YOLO detector、proposal fusion、text assignment、text rules
- Pin detection、Pin semantics
- Wire V1/V2/V3
- Topology V1/V2/V3
- Legacy `vision_v4.py`

主要权重哈希：

- `component_yolo11n_continue_v2_best.pt`：`51b4735d19315ba3ada49143e6f912aa414589f7961e140faab177dc6db7e776`
- `component_yolo11n_baseline_best.pt`：`42860935a0626a4445f2e30bf020c8e76ec01a87b816c6aacdf6288657bf1ac5`

## 多人协作入口

- Component 负责人：`pcb/component/**`；真实算法仍在现有 `component_*.py`，新版本通过 Stage 接入。
- Pin 负责人：`pcb/pin/localization/**`、`pcb/pin/semantics/**`。
- Wire 负责人：`pcb/wire_stage/**` 和对应原算法文件。
- Topology 负责人：`pcb/topology_stage/**` 和对应原算法文件。
- 公共契约：`pcb/core/**`、`pcb/schema.py`，由全组共同维护。

负责人新增算法版本时只实现相应 Stage、登记 Registry 并修改配置，不应修改其他 Stage 内部代码。

## Git 状态与建议提交

输入目录不是 Git 仓库，因此未创建 `refactor/modular-pipeline` 分支，也未伪造提交。建议导入 Git 后按以下粒度提交：

1. `docs: add pre-refactor audit and baseline manifest`
2. `refactor: add pipeline stage interfaces and context`
3. `refactor: add explicit stage registry and config`
4. `refactor: wrap current text and component stages`
5. `refactor: wrap pin localization and semantics`
6. `refactor: wrap wire topology and submission stages`
7. `refactor: add modular orchestration while retaining legacy entry`
8. `test: add stage contracts and full pipeline parity`
9. `docs: add architecture contracts and migration map`

## 是否可作为新的开发基线

可以。Modular 路径已在150例上与原 V4.1 prediction 完全一致，并保留 Legacy 回退。建议团队在下一次算法开发中显式使用 `--orchestrator modular --pipeline_config configs/current.json`；待多人环境再复核一次后，再单独讨论是否更改默认入口。

完整机器可读证据：

- `reports/refactor_baseline_manifest.json`
- `reports/refactor_parity_report.json`
- `reports/refactor_modular_metrics.json`
- `reports/refactor_validation_summary.json`
- `reports/refactor_test_results.txt`
