## 修改内容

- Stage：
- 新版本名：
- 解决的问题：

## 边界

- [ ] 没有读取或使用 0151～0200
- [ ] 没有读取或使用 10_GTcase
- [ ] 没有提交官方数据、runs、cache 或个人绝对路径
- [ ] 未修改无关 Stage
- [ ] schema/坐标/输出契约未改变；若改变，已明确说明并提供兼容测试

`SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE`

## 配置与环境

- 配置文件：
- 模型 SHA256：
- Python / PyTorch / Ultralytics / device：

## 验证

- [ ] `python tools/repository_doctor.py`
- [ ] `python -m unittest discover -s tests -v`
- [ ] strict validator 通过

| 本地非官方指标 | main 基线 | 本 PR | Delta |
|---|---:|---:|---:|
| ComponentF1 | | | |
| PinF1 | | | |
| NetHypergraphF1 | | | |
| NetLineF1 | | | |
| PinPairF1 | | | |
| FinalScore | | | |

## 风险与回滚

- 已知问题：
- 回滚配置：`configs/current.json`
