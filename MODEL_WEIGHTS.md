# 模型权重说明

模型文件由 Git LFS 管理。克隆仓库后先运行 `git lfs pull`，再用
`python tools/repository_doctor.py` 对照 `models/SHA256SUMS.json` 验证文件完整性。

V4.1 默认使用：

```text
models/component_yolo11n_continue_v2_best.pt
```

- 来源：同组项目 `EDA_components_handoff_20260914_v2` 的继续训练 YOLO11n。
- SHA256：`51B4735D19315BA3ADA49143E6F912AA414589F7961E140FAAB177DC6DB7E776`。
- 训练范围：同组代码记录为 0001～0150 中除 5 的倍数外的 120 例。
- 无训练重叠开发切片：0005、0010……0150，共 30 例。
- 用途：只负责 component symbol 的 bbox、type、confidence；不负责生成 designator、Name 或 value。

旧权重 `models/component_yolo11n_baseline_best.pt` 继续保留，可用于回滚和复现实验。固定 dev30 上，新权重将 symbol type+bbox F1 从 0.6873 提高到 0.8031，将严格 Component F1 从 0.3041 提高到 0.3580。

默认运行参数：

```text
tile_size=1280
overlap=192
confidence=0.25
class-aware NMS
```

如替换权重，请保持 Ultralytics YOLO 权重格式和官方 type 字符串，并重新运行 Component 专项评测。不能仅凭训练集内分数决定是否替换。
