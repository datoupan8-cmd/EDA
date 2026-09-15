# GitHub 仓库内容

仓库提交：

- 完整源代码、Stage Registry、配置、测试和说明文档；
- YOLO 与 OCR 模型，统一由 Git LFS 管理；
- OCR 兼容运行时及第三方说明；
- 紧凑的基线指标和 parity 证据；
- 跨电脑环境、单图运行和 150 例评测工具。

仓库不提交：

- 官方数据集和 target；
- `runs/`、submission、OCR/YOLO cache 和 debug 图片；
- 大体积逐案例评测明细和运行日志；
- `.venv`、IDE 配置和个人绝对路径；
- 本地 ZIP 交付包。

完整 150 例预测由每位开发者在本机生成。参考指标保存在
`reports/github_baseline_reference.json`。
