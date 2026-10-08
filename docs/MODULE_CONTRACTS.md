# 模块数据契约

## 坐标与可变性

所有 Stage 内部坐标均为原图像素坐标：原点在左上，x 向右，y 向下。只有 Submission Stage 调用原 `pcb.coordinates` 函数转换为官方左下原点坐标。

现有 `Component`、`Pin`、`Scene` 是可变对象。为保持行为，Stage 传递同一对象并允许原 Wire/Topology 函数原地写入 diagnostics 和 nets。本轮没有新增排序、复制或数值转换。

## Text Stage

输入：OpenCV BGR image、`PipelineContext`。

输出 `TextStageOutput`：

- `texts: list[Text]`
- `diagnostics: dict`
- `debug: dict`

`Text.bbox` 是文本外接框 `(x1,y1,x2,y2)`。Stage 只调用已经由 CLI 选择和构造的 OCR，不修改参数。

## Component Stage

输入：image、`TextStageOutput`、context。

输出 `ComponentStageOutput`：

- `components: list[Component]`
- `texts`：最终供后续使用的文本。选择性局部 OCR 启用时可包含原算法产生的更新结果。
- `roles`：原 `TokenRole` 列表。
- `diagnostics/debug`：原 Component 诊断与调试对象。

关键字段：

- `key`：内部及最终导出的元件键。
- `type`：官方 type taxonomy 的小写值。
- `bbox`：当前算法认定的元件框。
- `body_bbox`：存在时优先表示视觉器件主体；Pin/Wire 会优先使用它。
- `name`：Component Name/型号语义，保持现有实现。
- `value`：阻容感等值，保持现有实现。
- `confidence/source_id`：原置信度和来源。

## Pin Localization Stage

输入：image、Component 输出、context。

输出 `PinLocalizationOutput.terminals`，它是保持 Component 顺序的列表：

```text
[(Component, [terminal, ...]), ...]
```

terminal 字典必须保持当前字段语义：

- `tip`：器件外侧的电气端点候选，OpenCV 坐标。
- `base`：端子与器件 body 边界相接的位置。
- `side`：`left/right/top/bottom`。
- `method`：原定位方法。
- `wire_support_score`：原线条支持信息，允许为 `None`。

Localization 只回答“引脚在哪里”，不新增编号或名称逻辑。

## Pin Semantics Stage

输入：Component 输出、Pin Localization 输出、context。

输出 `PinSemanticsOutput`。它将原 `Pin` 对象赋给对应 `Component.pins`，并保留 `pin_events`。

`Pin` 关键字段：

- `number`：最终 pin key 的后缀语义。
- `name`：导出的 `pinname`。
- `tip/base/side`：必须与 Localization 结果一致。
- `exportable`：控制是否进入官方 JSON。
- `observable_number/internal_key/number_source`：保持已有可观测性语义。

Semantics Stage 不改变 terminal geometry。

## Wire Stage

输入：image、已经带 pins 的 `Scene`、context。

输出 `WireStageOutput`：

- `mask`：Topology 消费的二值 wire representation。
- `color_debug`
- `suppressed`
- `corridor`

不同版本输出数量不同，Wrapper 用 `None` 填充不存在的调试产物，不改变 mask。

## Topology Stage

输入：Scene、Wire 输出、context。

输出 `TopologyStageOutput.skeleton`。原算法继续原地更新：

- `scene.nets`
- snapping/junction/crossover/bridge 等 diagnostics

它不负责坐标翻转或 JSON 字段生成。

## Submission Stage

输入：完成 nets 的 Scene、context。

输出 `SubmissionStageOutput.data`。它直接调用原：

- `pcb.submission.export()`
- `pcb.submission.validate_strict()`

最终顶层严格为：

```json
{"components": {}, "pins": {}, "nets": {}}
```

任何非法官方 type、非法 bbox/point、失效 hyperGraph 引用或非法 edge 继续直接报错。

## Diagnostics 契约

现有顶层 diagnostics 字段全部保留。Stage 输出额外提供结构化容器，但 Pipeline 不删除、不重命名原算法写入的字段。计时、cache hit 等运行环境字段允许因缓存冷暖状态变化；它们不属于预测语义 parity。
