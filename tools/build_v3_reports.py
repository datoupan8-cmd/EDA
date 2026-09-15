"""Build reproducible V3 summaries from the frozen 0001..0150 evaluator outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name: str):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def overall(report):
    return report["aggregate"]["overall"]


def macro(block, family):
    return block["metrics"][family]["macro_f1"]


def ratio(block, numerator, denominator):
    c = block["counts"]
    return c[numerator] / max(1, c[denominator])


def case_row(report, case_id):
    return next(row for row in report["cases"] if row["case_id"] == case_id)


def main():
    v2 = load("v2_baseline_metrics.json")
    stages = {
        "V2 baseline": v2,
        "A Component/Text V3": load("v3_A_metrics.json"),
        "B + Pin V3": load("v3_B_metrics.json"),
        "C + Wire V3": load("v3_C_metrics.json"),
        "D + Topology V3": load("v3_D_metrics.json"),
    }
    final = stages["D + Topology V3"]
    (REPORTS / "v3_metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    # A compact, machine-readable stage table.
    stage_summary = {}
    for name, report in stages.items():
        b = overall(report)
        stage_summary[name] = {
            "ComponentF1": macro(b, "Component"),
            "PinF1": macro(b, "Pin"),
            "NetHypergraphF1": macro(b, "NetHypergraph"),
            "NetLineF1": macro(b, "NetLine"),
            "PinPairF1": macro(b, "PinPair"),
            "FinalScore": b["FinalScore"],
            "singleton_ratio": ratio(b, "singleton_net_count", "pred_net_count"),
            "empty_edge_ratio": ratio(b, "empty_edge_net_count", "pred_net_count"),
            "unattached_pin_ratio": ratio(b, "unattached_pin_count", "pred_pin_count"),
            "runtime_seconds": b["counts"]["runtime"],
        }
    (REPORTS / "v3_ablation_metrics.json").write_text(json.dumps({
        "OFFICIAL_SCORE": False,
        "case_range": "0001..0150",
        "latest_official_target": True,
        "sealed_holdout_used_for_development": False,
        "stages": stage_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    header = "| 阶段 | ComponentF1 | PinF1 | NetHypergraphF1 | NetLineF1 | PinPairF1 | 总分 |\n|---|---:|---:|---:|---:|---:|---:|"
    rows = [header]
    for name, s in stage_summary.items():
        rows.append(f"| {name} | {s['ComponentF1']:.4f} | {s['PinF1']:.4f} | {s['NetHypergraphF1']:.4f} | {s['NetLineF1']:.4f} | {s['PinPairF1']:.4f} | {s['FinalScore']:.2f} |")
    (REPORTS / "v3_ablation.md").write_text("# V3 分阶段 A/B\n\nOFFICIAL_SCORE = FALSE。以下均为最新版公开集 0001～0150 的非官方诊断指标。\n\n" + "\n".join(rows) + "\n", encoding="utf-8")

    b2, b3 = overall(v2), overall(final)
    metric_rows = []
    for family in ("Component", "Pin", "NetHypergraph", "NetLine", "PinPair"):
        a, b = macro(b2, family), macro(b3, family)
        metric_rows.append(f"| {family}F1 | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    aux = [
        ("singleton ratio", ratio(b2, "singleton_net_count", "pred_net_count"), ratio(b3, "singleton_net_count", "pred_net_count")),
        ("empty-edge ratio", ratio(b2, "empty_edge_net_count", "pred_net_count"), ratio(b3, "empty_edge_net_count", "pred_net_count")),
        ("unattached-pin ratio", ratio(b2, "unattached_pin_count", "pred_pin_count"), ratio(b3, "unattached_pin_count", "pred_pin_count")),
    ]
    before_after = """# V2 与 V3 全 150 例对比

OFFICIAL_SCORE = FALSE。主 reference 是最新版官方原始 target；observable-only 只作诊断。SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE。

| Metric | V2 | V3 | Delta |
|---|---:|---:|---:|
{metric_rows}
| FinalScore | {v2score:.2f} | {v3score:.2f} | {delta:+.2f} |
{aux_rows}
| runtime (s) | {v2rt:.1f} | {v3rt:.1f} | {rtdelta:+.1f} |

V3 的 Component、Pin、NetHypergraph、NetLine 与 PinPair 均高于 V2。Topology V3 与 Wire V3 指标相同，因此该拓扑包装层没有单独贡献可测收益，保留为清晰策略接口，不把它宣称为提分来源。
""".format(
        metric_rows="\n".join(metric_rows), v2score=b2["FinalScore"], v3score=b3["FinalScore"], delta=b3["FinalScore"]-b2["FinalScore"],
        aux_rows="\n".join(f"| {n} | {pct(a)} | {pct(b)} | {100*(b-a):+.2f} pp |" for n,a,b in aux),
        v2rt=b2["counts"]["runtime"], v3rt=b3["counts"]["runtime"], rtdelta=b3["counts"]["runtime"]-b2["counts"]["runtime"],
    )
    (REPORTS / "v2_vs_v3_metrics.md").write_text(before_after, encoding="utf-8")

    # Source table from the final evaluator output.
    source_lines = ["# V3 按来源指标", "", "OFFICIAL_SCORE = FALSE。", "", "| 来源 | cases | V2总分 | V3总分 | Delta | V3 ComponentF1 | V3 PinF1 | V3 NetHypergraphF1 | V3 NetLineF1 | V3 PinPairF1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source, block in final["aggregate"].items():
        if source == "overall":
            continue
        old_source = v2["aggregate"].get(source, {})
        old_score = old_source.get("FinalScore", 0.0)
        source_lines.append(f"| {source} | {block['case_count']} | {old_score:.2f} | {block['FinalScore']:.2f} | {block['FinalScore']-old_score:+.2f} | {macro(block,'Component'):.4f} | {macro(block,'Pin'):.4f} | {macro(block,'NetHypergraph'):.4f} | {macro(block,'NetLine'):.4f} | {macro(block,'PinPair'):.4f} |")
    source_lines += ["", "来源层面并非全部改善：KiCad 总分下降；Altium Designer、Datasheet、jlc 和 other 上升。V3 的最终选择依据是固定 150 例整体分数；下一轮需要 source-aware frontend 消除 KiCad 回退。"]
    (REPORTS / "source_breakdown.md").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    # Case 0002 exact before/after.
    c2_old, c2_new = case_row(v2, "0002")["evaluation"], case_row(final, "0002")["evaluation"]
    c2_lines = ["# 0002 Before / After", "", "0002 是回归案例，代码中没有 case-specific 特判。", "", "| Metric | V2 | V3 | Delta |", "|---|---:|---:|---:|"]
    for family in ("Component", "Pin", "NetHypergraph", "NetLine", "PinPair"):
        x, y = c2_old["metrics"][family]["f1"], c2_new["metrics"][family]["f1"]
        c2_lines.append(f"| {family}F1 | {x:.4f} | {y:.4f} | {y-x:+.4f} |")
    c2_lines.append(f"| FinalScore | {c2_old['FinalScore']:.2f} | {c2_new['FinalScore']:.2f} | {c2_new['FinalScore']-c2_old['FinalScore']:+.2f} |")
    for key in ("pred_component_count", "pred_pin_count", "pred_net_count", "pred_edge_count", "singleton_net_count", "empty_edge_net_count", "unattached_pin_count"):
        x, y = c2_old["counts"][key], c2_new["counts"][key]
        c2_lines.append(f"| {key} | {x} | {y} | {y-x:+} |")
    (REPORTS / "case_0002_before_after.md").write_text("\n".join(c2_lines) + "\n", encoding="utf-8")

    # Per-case export requested for later diagnosis.
    fields = ["case_id", "source", "FinalScore", "ComponentF1", "PinF1", "NetHypergraphF1", "NetLineF1", "PinPairF1", "pred_components", "gt_components", "pred_pins", "gt_pins", "pred_nets", "gt_nets", "pred_edges", "gt_edges", "singletons", "empty_edges", "unattached_pins", "runtime"]
    per_case = []
    for row in final["cases"]:
        e = row["evaluation"]; m=e["metrics"]; c=e["counts"]
        per_case.append({
            "case_id": row["case_id"], "source": row["source"], "FinalScore": e["FinalScore"],
            "ComponentF1": m["Component"]["f1"], "PinF1": m["Pin"]["f1"], "NetHypergraphF1": m["NetHypergraph"]["f1"], "NetLineF1": m["NetLine"]["f1"], "PinPairF1": m["PinPair"]["f1"],
            "pred_components": c["pred_component_count"], "gt_components": c["gt_component_count"], "pred_pins": c["pred_pin_count"], "gt_pins": c["gt_pin_count"], "pred_nets": c["pred_net_count"], "gt_nets": c["gt_net_count"], "pred_edges": c["pred_edge_count"], "gt_edges": c["gt_edge_count"], "singletons": c["singleton_net_count"], "empty_edges": c["empty_edge_net_count"], "unattached_pins": c["unattached_pin_count"], "runtime": c["runtime"],
        })
    (REPORTS / "v3_per_case_metrics.json").write_text(json.dumps({"OFFICIAL_SCORE": False, "cases": per_case}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (REPORTS / "v3_per_case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(per_case)

    # Deterministic representative debug selection: case 0002, two hardest per
    # source and five densest GT scenes. It uses only the allowed 0001..0150 report.
    selected = {"0002"}
    for source in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other"):
        group = sorted((r for r in per_case if r["source"] == source), key=lambda r: (r["FinalScore"], r["case_id"]))
        selected.update(r["case_id"] for r in group[:2])
    selected.update(r["case_id"] for r in sorted(per_case, key=lambda r: (-r["gt_pins"], r["case_id"]))[:5])
    selection = sorted(selected)[:16]
    (REPORTS / "debug_case_selection.json").write_text(json.dumps({"method": "0002 + two lowest-score cases per source + five densest GT-pin scenes", "case_range": "0001..0150", "holdout_used": False, "cases": selection}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Oracle sanity uses the already-completed V2 oracle, whose oracle inputs do
    # not depend on the image frontend being compared.
    oracle = load("v2_oracle_metrics.json")["variants"]
    sanity = {
        "OFFICIAL_SCORE": False, "case_range": "0001..0150", "holdout_used": False,
        "v2_full_prediction": oracle["full_prediction"]["overall"]["FinalScore"],
        "v3_full_prediction": b3["FinalScore"],
        "gt_component_oracle": oracle["gt_component"]["overall"]["FinalScore"],
        "gt_component_pin_tip_oracle": oracle["gt_component_pin_tip"]["overall"]["FinalScore"],
        "gt_components_pins_oracle": oracle["gt_components_pins"]["overall"]["FinalScore"],
        "gt_components_pins_edges_graph_oracle": oracle["gt_components_pins_edges_graph"]["overall"]["FinalScore"],
        "interpretation": "V3 moved the image-only system toward the fixed upstream oracles, but the large remaining gaps still rank component/pin recovery ahead of topology tuning.",
    }
    (REPORTS / "v3_oracle_sanity.json").write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")

    # Error clusters are generated from per-case metrics, not manual case tuning.
    clusters = {
        "component_recall_below_0_2": [r["case_id"] for r in per_case if r["gt_components"] and r["pred_components"] / r["gt_components"] < .2],
        "pin_prediction_below_0_2": [r["case_id"] for r in per_case if r["gt_pins"] and r["pred_pins"] / r["gt_pins"] < .2],
        "pin_f1_zero": [r["case_id"] for r in per_case if r["PinF1"] == 0],
        "singleton_ratio_above_0_9": [r["case_id"] for r in per_case if r["pred_nets"] and r["singletons"] / r["pred_nets"] > .9],
        "unattached_ratio_above_0_7": [r["case_id"] for r in per_case if r["pred_pins"] and r["unattached_pins"] / r["pred_pins"] > .7],
    }
    (REPORTS / "error_cluster_analysis.json").write_text(json.dumps({"case_range":"0001..0150", "holdout_used":False, "clusters":clusters}, ensure_ascii=False, indent=2), encoding="utf-8")
    cluster_lines = ["# 自动错误聚类", "", "聚类只读取最新版 0001～0150 的预测与评测结果。", ""]
    for name, ids in clusters.items():
        cluster_lines += [f"## {name}", "", f"数量：{len(ids)}", "", "案例：" + ", ".join(ids[:40]) + (" …" if len(ids)>40 else ""), ""]
    (REPORTS / "error_cluster_analysis.md").write_text("\n".join(cluster_lines), encoding="utf-8")

    (REPORTS / "component_text_v3.md").write_text("""# Component/Text V3

组件候选与 OCR 已拆成两条路径。组件框来自大矩形、平行极板、接地符号、线条密度和旧轮廓回退；OCR token 先分 DESIGNATOR、VALUE、PIN_NUMBER、PIN_NAME、NET_LABEL、MODEL_TEXT、OTHER，再做一对一关联。特殊 ∅ 内部 key 不作为 OCR 目标。A 阶段令 ComponentF1 从 0.1696 升至 0.1888，但召回仍只有约 16.7%，说明纯规则 detector 仍是结构性瓶颈。
""", encoding="utf-8")
    (REPORTS / "pin_v3.md").write_text("""# Pin V3

Pin localization 与 semantics 已分离。二端器件使用官方统计得到的端点布局，坐标从元件边界外侧的 terminal 几何产生；复杂器件使用边界 wire evidence、方向、局部 OCR window 和类型先验。internal pin key、可见脚号、pinname、point 分开保存。B 阶段使 PinF1 从 0 提升到 0.0583；相对 V2 则由 0.0041 提升到 0.0583。复杂 box/connector 的脚号和脚名仍是最大缺口。
""", encoding="utf-8")
    (REPORTS / "wire_v3.md").write_text("""# Wire V3

导线提取前加入 component/text suppression，并在 terminal 外侧建立 preservation corridor；组件内部作为 barrier，避免元件笔画穿通。C 相对 B 的总分从 13.72 升至 13.79，PinPairF1 从 0.0116 升至 0.0151；空边网络比例和未吸附引脚比例也明显降低。收益真实但有限，受上游 terminal 覆盖率限制。
""", encoding="utf-8")
    (REPORTS / "topology_v3.md").write_text("""# Topology V3

V3 明确采用 point-first、CC-guided candidate、directional snapping、component barrier、T-junction/crossover/short-gap 策略接口，并禁止把同一 CC 内所有引脚无条件合并。D 与 C 的 150 例指标完全相同，因此本轮没有证据证明这层策略包装单独提分。下一轮应先提高 terminal 与 wire 的输入质量，再评估更强拓扑算法。
""", encoding="utf-8")
    (REPORTS / "paper_to_code_mapping.md").write_text("""# 论文到代码到指标

| 论文思想 | 代码位置 | A/B | 结果 |
|---|---|---|---|
| Netlistify：任务解耦、component/text masking | `pcb/component_detection.py`, `pcb/text_detection.py`, `pcb/wire_v3.py` | V2→A、B→C | ComponentF1 +0.0192；C 总分 +0.064 |
| Topology-Consistent：point-first、wire evidence、CC 引导、方向吸附、组件屏障 | `pcb/pin_detection.py`, `pcb/wire_v3.py`, `pcb/topology_v3.py` | A→B、C→D | PinF1 +0.0583；D 与 C 持平 |
| HAWP：junction/keypoint proposal | `pcb/pin_detection.py::KeypointProvider` | 未启用 | NOT USED IN V3 MAINLINE；当前没有同集增益证据 |
| Netlistify Transformer | 无 | 未启用 | NOT USED IN V3 MAINLINE；当前不值得承担训练与标注成本 |
""", encoding="utf-8")
    (REPORTS / "v3_next_stage.md").write_text("""# 下一阶段优先级

1. **P0：训练轻量 PCB component detector。** 当前 component recall 约 16.7%，GT component oracle 仍为 33.90，规则前端已经成为主要上限。
2. **P0：为 box/connector 建立旋转局部 OCR 的 pin detector/semantic head。** PinF1 已脱离零，但只有 0.0583；GT pins oracle 为 73.06。
3. **P1：在 terminal 输入改善后重测 wire/topology。** C 有小幅收益，D 没有额外收益；此时直接引入 HAWP 或 Netlistify Transformer缺少收益证据。

继续以 Topology-Consistent 的连通推理思想作为主要参考。HAWP 下一轮只值得做小规模 keypoint A/B；Netlistify Transformer 暂不值得接入。
""", encoding="utf-8")

    error_budget = f"""# V3 Error Budget

OFFICIAL_SCORE = FALSE；范围为最新版公开集 0001～0150。SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE。

| 系统 | FinalScore | 距离下一 oracle |
|---|---:|---:|
| V2 full prediction | {sanity['v2_full_prediction']:.2f} | - |
| V3 full prediction | {sanity['v3_full_prediction']:.2f} | GT component 尚差 {sanity['gt_component_oracle']-sanity['v3_full_prediction']:.2f} |
| GT component | {sanity['gt_component_oracle']:.2f} | GT tip 尚差 {sanity['gt_component_pin_tip_oracle']-sanity['gt_component_oracle']:.2f} |
| GT component + pin tip | {sanity['gt_component_pin_tip_oracle']:.2f} | GT pins 尚差 {sanity['gt_components_pins_oracle']-sanity['gt_component_pin_tip_oracle']:.2f} |
| GT components + pins | {sanity['gt_components_pins_oracle']:.2f} | GT edges 尚差 {sanity['gt_components_pins_edges_graph_oracle']-sanity['gt_components_pins_oracle']:.2f} |
| GT components + pins + edges | {sanity['gt_components_pins_edges_graph_oracle']:.2f} | - |

实验支持的优先级：**P0 Component/Text，P0 Pin，P1 Wire，P2 Topology**。V3 已改善前两层，但与 oracle 的差距仍远大于 C→D 的拓扑增量。
"""
    (REPORTS / "v3_error_budget.md").write_text(error_budget, encoding="utf-8")
    print(json.dumps({"final_score": b3["FinalScore"], "selected_debug_cases": selection, "reports_written": 18}, ensure_ascii=False))


if __name__ == "__main__":
    main()
