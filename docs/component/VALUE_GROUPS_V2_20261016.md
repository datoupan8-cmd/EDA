# Component 数值归属 V2

本轮比较对象为上一轮默认 `printed_value_groups_v1`。只开发和评测 Component，未运行 Pin、Wire、Topology、Net 阶段。无新增模型、额外 OCR 或依赖安装。

当前默认已启用 `ordered_value_groups_v2`，原 `printed_value_groups_v1` 仍可切回。

## 实测结果

| 方案 | 150 页宏平均 ComponentF1 | 正确器件数 | 内部 30 页宏平均 |
|---|---:|---:|---:|
| 上一轮默认 | 0.410919 | 2435 | 0.406923 |
| 单位与数字干扰处理 | 0.413450 | 2452 | 0.409005 |
| **加入阅读顺序软约束（新默认）** | **0.413544** | **2452** | **0.409542** |

相较上一轮默认：21 个器件由错变对、4 个由对变错，净增 17 个；16 页提高、2 页下降、132 页持平。上一轮记录的 25 个误改中修复了 8 个。检查通过 43 项。

150 页的位号、型号、类型、框和顺序均与上一轮相同；参照预测逐页核验与上一轮已保存结果完全一致。预测哈希、源文件哈希、逐器件变化与默认选择记录见 `runs/component_guarded_ordered_dev150_20261016_final/promotion_record.json`。

## 改动依据与实现

人工错误分析只查看 0001—0150 内、既有 `train120` 分组的页面（编号不能被 5 整除）。新的规则仅使用当前图像的 OCR 文本、文字框、器件框，不使用案例名称、GT 空值查表或记忆的器件参数。

1. **完整单位文字优先**：同一物理区域出现多个高置信度读法时，优先选择具有完整单位的合法原文。例如 `300Ω` 对 `3000`、`4.7uF` 对 `4.7u`。不补写 OCR 没有读出的字符，不换算单位。
2. **排除被占用的文字**：位号 OCR 的数字后缀不能再次被当成数值；`R1 10k` 已绑定 R1 后，同处的 `10k` 独立识别结果也不能再分给 R2。
3. **数字列负证据**：芯片、连接器框内或框边、具有同列邻居的纯数字，作为数值候选时排除。使用器件框与 OCR 几何关系完成，不调用引脚定位模块。
4. **阅读顺序软约束**：对位号后方/下方的数值给予更强支持；逆向排版仍能进入几何匹配，不硬性禁止。
5. **允许相邻文字框轻微重叠**：按文字中心行距、字号与左侧/中心对齐判断是否为两行，避免 OCR 框轻微重叠就丢弃有效的位号—数值组合。
6. **统一占用物理区域**：电容简写也进入同一张匹配矩阵，不在匹配后再次补填，避免同一处文字重复使用。

输出仍保留位号、型号、器件类型与框，只重新分配 R/C/L 的原文数值。外部文本及角色列表不变。

训练分组的人工检查中，也发现了少量图上可读数值而 target 的 value 为 null 的情况。这种差异会影响本地严格比较，但本轮没有为匹配空标注而删掉可见文字，也没有据此建立案例规则。

## 复现

```powershell
python tools/evaluate_component_local_joint.py `
  --dataset '公开数据集/200_train_cases' `
  --output runs/component_guarded_ordered_dev150_20261016_final `
  --variants baseline,value_groups,guarded_values,ordered_values --evaluate
```

- `baseline`：更早的 `contextual_values_v2`，保留用于历史分数对照。
- `value_groups`：本轮开始时的默认 `printed_value_groups_v1`。
- `guarded_values`：单位和数字干扰处理，不加入阅读顺序偏好。
- `ordered_values`：加入上述阅读顺序软约束。

所有案例与方案的预测先保存、计算哈希，再读取相应 GT。默认白名单为精确目录 0001—0150；缺失、重定向、重复编号会停止，0151—0200 与 10_GTcase 不参与。日志记录实际编号。

指标仍为 `evaluate_v2` 中 Component 专项代理评分，不是官方评分器结果。内部 30 页属于同一个开发白名单，也曾用于方案检查，不能作为独立官方验证。

## 文件

- 实现：`pcb/component/value_groups_v2.py`
- 专项检查：`tests/component/test_value_groups_v2.py`，通过现有模块测试发现机制接入。
- 配置：`configs/component/guarded_value_groups_v2.json`、`ordered_value_groups_v2.json`
- 最终对照：`runs/component_guarded_ordered_dev150_20261016_final/report.json`

## Pin L45 分支集成

器件代码在 `E:/EDA/EDA` 的 `feature/pin-l45-validated-20261012` 集成，默认配置选择 V2。只恢复器件相关文件，不恢复 GitHub Desktop 暂存中的 Pin、Wire 或网络改动；保留原暂存记录。Context.resources 不加入模型会话对象，新测试位于该分支现有的 `tests/component/` 目录。集成时另跑 Component 专项 150 页回归，核对与上轮保存的逐页预测是否一致；未运行其他模块评分。

集成验证：63 项器件相关检查通过，150 页三种对照共 450 份器件预测与上轮逐字节一致；V2 的本地宏平均 ComponentF1 仍为 0.413544、正确器件数 2452。摘要见 `PINL45_INTEGRATION_20261016.json`，本机完整回归位于 `runs/component_pinl45_integration_dev150_20261016/`。
