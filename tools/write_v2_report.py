"""Generate the final before/after report from committed JSON evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
metrics=json.loads((ROOT/'reports/v2_metrics.json').read_text(encoding='utf8'))
oracle=json.loads((ROOT/'reports/v2_oracle_ablation.json').read_text(encoding='utf8'))
full=json.loads((ROOT/'reports/v2_metrics_full_matrix_pre_rm2_filter.json').read_text(encoding='utf8'))
details=json.loads((ROOT/'runs/v2_final_validation/stage_details.json').read_text(encoding='utf8'))['cases']
b=metrics['stages']['v1_baseline'];v=metrics['stages']['v2_safe_text_sanitized_fixed']

def pct(x):return f'{x:.4f}'
def delta(x,y):return f'{y-x:+.4f}'
def row(name,x):
    return f"| {name} | {pct(x['ComponentF1'])} | {pct(x['PinF1'])} | {pct(x['NetHypergraphF1'])} | {pct(x['NetLineF1'])} | {x['weighted_diagnostic_score']:.3f} | {x['NetPairF1_singleton_resistant']:.4f} | {x['false_merge_count']:.2f} |"

one=next(x for x in details if x['case_id']=='0001');ob=one['v1_baseline'];ov=one['v2_safe_text_sanitized_fixed']
example_b=json.loads((ROOT/'examples/0001/baseline.json').read_text(encoding='utf8'));example_v=json.loads((ROOT/'examples/0001/v2.json').read_text(encoding='utf8'))
def edge_count(x):return sum(len(n['edges']) for n in x['nets'].values())
oracle_rows=[]
for name,x in oracle['oracles'].items():oracle_rows.append(row(name,x))
matrix_rows=[]
for name,x in sorted(full['stages'].items(),key=lambda kv:kv[1]['weighted_diagnostic_score'],reverse=True):matrix_rows.append(row(name,x))

text=f"""# V2 Before / After

> `OFFICIAL=false`。25 张 case 来自冻结清单 `splits/v2_dev.json`；参考由 Dataset300 CVJsonStd 派生。主入口在打开 annotation 之前完成全部图像预测，不能把本表当作正式榜单成绩。

## 最终保留版本

| stage | ComponentF1 | PinF1 | NetHypergraphF1 | NetLineF1 | weighted diagnostic | NetPairF1 | false merges/case |
|---|---:|---:|---:|---:|---:|---:|---:|
{row('V1 baseline',b)}
{row('V2 final: safe text + pin evidence + adaptive wire + fixed snap',v)}

变化：ComponentF1 {delta(b['ComponentF1'],v['ComponentF1'])}，PinF1 {delta(b['PinF1'],v['PinF1'])}，NetHypergraphF1 {delta(b['NetHypergraphF1'],v['NetHypergraphF1'])}，NetLineF1 {delta(b['NetLineF1'],v['NetLineF1'])}，加权诊断 {v['weighted_diagnostic_score']-b['weighted_diagnostic_score']:+.3f}。平均 false merge 从 {b['false_merge_count']:.2f} 降到 {v['false_merge_count']:.2f}；unattached terminal ratio 从 {b['unattached_pin_ratio']:.3f} 降到 {v['unattached_pin_ratio']:.3f}；每图导出的 physical edges 从 {b.get('exported_edges',b['edges']):.1f} 增至 {v.get('exported_edges',v['edges']):.1f}。

严格 PinF1 下降 {b['PinF1']-v['PinF1']:.4f}，原因是 V2 不再导出没有物理编号证据的复杂器件 terminal，并屏蔽与器件 pin 数量明显冲突的 178/179。探索版保留这些占位项时 weighted diagnostic 为 {full['stages']['v2_safe_text_fixed']['weighted_diagnostic_score']:.3f}，但不满足比赛输出真实性要求，因此没有选作默认。

## 0001 可复查案例

| 0001 | baseline | V2 final |
|---|---:|---:|
| component predictions | {ob['components']} | {ov['components']} |
| terminal candidates | {ob['pins']} | {ov['pins']} |
| exported pins | {ob['metrics']['counts']['pins_pred']} | {ov['metrics']['counts']['pins_pred']} |
| exported nets | {ob['metrics']['diagnostic_layers']['pred_nets']} | {ov['metrics']['diagnostic_layers']['pred_nets']} |
| physical edges | {edge_count(example_b)} | {edge_count(example_v)} |
| unattached terminals | {ob['unattached_pins']} | {ov['unattached_pins']} |
| false merge pairs | {ob['false_merge_count']} | {ov['false_merge_count']} |
| weighted diagnostic | {ob['metrics']['weighted_diagnostic_score']:.3f} | {ov['metrics']['weighted_diagnostic_score']:.3f} |

V2 用局部图像证据得到 `R1410k → R14 / 10k` 和 `R101k → R10 / 1k`。最终 `examples/0001/v2.json` 不含 `pin_RM2`、`pin_178`、`pin_179` 或 `pin_UNK*`；内部仍保留这些 terminal 参与诊断。

## 完整阶段矩阵

| stage | ComponentF1 | PinF1 | NetHypergraphF1 | NetLineF1 | weighted diagnostic | NetPairF1 | false merges/case |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(matrix_rows)}

结论：adaptive fixed raw 是 topology 的最佳纯 A/B；directional 略降，普通同名 signal label 合并进一步下降。完整 Hungarian component frontend、全局角色重分配和旋转 Pin 分支也下降，因此没有进入默认 pipeline。最后的 RM2/178/179 过滤在 `reports/v2_metrics.json` 中重新跑过全部 25 张。

## A–E Oracle Ablation

| oracle | ComponentF1 | PinF1 | NetHypergraphF1 | NetLineF1 | weighted diagnostic | NetPairF1 | false merges/case |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(oracle_rows)}

- A：当前 V1 全图预测；B：GT component key/bbox/type，其余预测；C：GT component 与 pin tip/base，pin number/name 仍预测；D：GT components+pins，只预测 wire/topology；E：完整合规 V2 图像预测。
- D 与 E 的差距量化了 component、tip 和 pin semantics 是首要瓶颈。D 的 NetPairF1 若仍低，则说明 wire/crossing/topology 仍有独立误差。

## 保留与拒绝

保留：安全 fused OCR、Pin 证据过滤、自适应 wire、触边 frame 删除、fixed snapping、CC/crossing/short bridge、分层 diagnostics、singleton-resistant evaluator。

默认关闭：directional snapping、ordinary flying-label merge。保留为实验代码：全局 Hungarian component/text frontend、四方向 Pin OCR。未接入：HAWP 和 Netlistify。

## 下一轮优先级

1. 训练 nano/small PCB component detector，并只在同一 25 张上确认 Component bbox recall/precision 和 ComponentF1 提升后接入。
2. 基于可靠 component body 训练/改进 terminal tip，再运行 C/D 类型 ablation；随后优化旋转 pin number/name OCR。
3. 对 false crossing/断线案例建立 junction 子集；规则达到瓶颈且 HAWP keypoint 在同一子集提升后，再接入 HAWP。
4. 扩充普通 flying-label 的角色分类与 endpoint association；现版本默认关闭。
5. Netlistify Transformer 仅在 learned wire-pair 数据已准备、规则 CC 仍不足时考虑。

继续以 Topology-Consistent Parsing 的 CC/snapping/crossover/short-bridge 思路作为拓扑主要参考。HAWP 下一步只值得做 junction 候选小实验；Netlistify 目前不值得接入主线。

## 验证状态

- 20 项结构测试通过，记录在 `reports/v2_test_results.txt`。
- 0001 的 PNG-only baseline/V2 JSON 与 overlay 已生成。
- Docker build 已真实尝试，但本机无 Docker executable，未通过；见 `reports/v2_docker_validation.json`。
"""
(ROOT/'reports/06_v2_before_after.md').write_text(text,encoding='utf8')
print('wrote reports/06_v2_before_after.md')
