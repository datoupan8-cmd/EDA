# 重构中发现但本轮未修改的问题

以下内容可能影响准确率、维护性或部署，但本轮遵守行为保持要求，未做修复。

1. `vision_v4.detect_scene_v4()` 的 Legacy 实现仍混合 Text、Component 和 Pin。Modular 路径已拆开调用边界；旧函数保留用于 parity。
2. Component V4 的选择性局部 OCR 编排目前在 Legacy 函数和 V4 Wrapper 中各有一份等价代码。若未来抽成单一公共函数，必须再次做150例 parity。
3. Component V3 仍包含按 `image_name` 推断来源并使用先验的历史逻辑。本轮没有删除或修改。
4. `PinLocalizationStageV3` 继续使用当前模板/边界线方案，`PinSemanticsStageV3` 继续使用当前 OCR 和类型先验；其准确率问题不属于本轮。
5. Wire V3 和 Topology V3 仍依赖可变 Scene 及 diagnostics 中的部分信息；Stage 契约已经显式记录，但算法未改成纯函数。
6. `pcb/io.py`、`pcb/wire.py`、`pcb/topology.py` 的模块名阻止直接建立同名包，因此采用 `output/`、`wire_stage/` 和 `topology_stage/`。
7. 原上传 ZIP 未包含项目内 `vendor/` runtime；本重构副本从同一 V4.1 已验证交付副本恢复了原 EasyOCR/RapidOCR runtime，算法版本未变。后续打包应检查该目录是否保留。
8. GPU、OCR 线程和缓存冷暖会影响运行时间、cache diagnostics，极少数环境也可能产生数值非确定性。150例本机验证的 prediction 严格一致。
9. 默认入口仍是 Legacy。团队确认 Modular 作为新基线后，可在单独变更中调整默认值；本轮不改变默认行为。
10. 当前 Registry 是代码内显式登记。它满足组内版本组合，不支持第三方动态插件；这是有意限制。

这些问题只能在后续独立任务中处理，并为每项建立新的行为或指标基线。
