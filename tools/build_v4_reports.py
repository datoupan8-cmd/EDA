"""Build compact V4 A/B, per-case, source, and error reports from saved metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STAGES = ("A", "B", "C", "D", "E", "F", "BEST")


def load(stage):
    standard = REPORTS / ("v3_baseline_metrics.json" if stage == "A" else f"v4_{stage}_metrics.json")
    component = REPORTS / f"v4_{stage}_component_metrics.json"
    return json.loads(standard.read_text(encoding="utf-8")), json.loads(component.read_text(encoding="utf-8"))


def summary(stage, standard, component):
    overall = standard["aggregate"]["overall"]
    specialty = component["aggregate"]["overall"]
    dev = component["aggregate"]["dev_30_no_training_overlap"]
    metrics = overall["metrics"]
    counts = overall["counts"]
    return {
        "stage": stage,
        "success_count": standard.get("success_count", 150),
        "ComponentF1_macro": metrics["Component"]["macro_f1"],
        "PinF1_macro": metrics["Pin"]["macro_f1"],
        "NetHypergraphF1_macro": metrics["NetHypergraph"]["macro_f1"],
        "NetLineF1_macro": metrics["NetLine"]["macro_f1"],
        "PinPairF1_macro": metrics["PinPair"]["macro_f1"],
        "FinalScore": overall["FinalScore"],
        "SymbolTypeBBoxF1_micro": specialty["symbol_type_bbox"]["f1"],
        "BBoxF1_micro": specialty["bbox"]["f1"],
        "TypeAccuracy": specialty["type_accuracy"]["accuracy"],
        "DesignatorOCRRecall": specialty["designator_ocr"]["recall"],
        "DesignatorAssociationAccuracy": specialty["designator_association"]["recall"],
        "NameAccuracy": specialty["name_accuracy"]["accuracy"],
        "ValueAccuracy": specialty["value_accuracy"]["accuracy"],
        "FalsePositivePerImage": specialty["false_positive_per_image"],
        "DuplicatePairs": specialty["duplicate_component_pairs"],
        "Dev30_ComponentF1_macro": dev["strict_component"]["macro_f1"],
        "Dev30_SymbolTypeBBoxF1_micro": dev["symbol_type_bbox"]["f1"],
        "singleton_ratio": counts["singleton_net_count"] / counts["pred_net_count"] if counts["pred_net_count"] else 0,
        "empty_edge_ratio": counts["empty_edge_net_count"] / counts["pred_net_count"] if counts["pred_net_count"] else 0,
        "unattached_pin_ratio": counts["unattached_pin_count"] / counts["pred_pin_count"] if counts["pred_pin_count"] else 0,
    }


def main():
    loaded = {stage: load(stage) for stage in STAGES}
    summaries = [summary(stage, *loaded[stage]) for stage in STAGES]
    payload = {
        "OFFICIAL_SCORE": False,
        "case_range": [1, 150],
        "sealed_holdout_used": False,
        "detector_generalization_slice": "dev_30_no_training_overlap",
        "stages": summaries,
        "selected_mainline": "BEST = C + value; Name and geometry disabled after negative A/B",
    }
    (REPORTS / "v4_ablation_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["case_id", "source"]
    for stage in STAGES:
        fields += [f"{stage}_ComponentF1", f"{stage}_SymbolTypeBBoxF1", f"{stage}_FinalScore"]
    case_maps = {stage: {row["case_id"]: row for row in loaded[stage][1]["cases"]} for stage in STAGES}
    with (REPORTS / "v4_per_case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for number in range(1, 151):
            case_id = f"{number:04d}"; row = {"case_id": case_id, "source": case_maps["BEST"][case_id]["source"]}
            for stage in STAGES:
                ev = case_maps[stage][case_id]["evaluation"]
                row[f"{stage}_ComponentF1"] = ev["strict_component"]["f1"]
                row[f"{stage}_SymbolTypeBBoxF1"] = ev["symbol_type_bbox"]["f1"]
                row[f"{stage}_FinalScore"] = ev["FinalScore"]
            writer.writerow(row)

    labels = {
        "A": "V3 baseline", "B": "YOLO + greedy designator", "C": "+ global designator",
        "D": "+ Name", "E": "+ value", "F": "+ raw geometry fallback", "BEST": "C + value",
    }
    lines = ["# Component V4 A/B 实验", "", "OFFICIAL_SCORE = FALSE；只使用最新版官方 0001～0150。", "", "| 阶段 | 说明 | ComponentF1 | type+bbox F1 | PinF1 | NetHypergraphF1 | NetLineF1 | PinPairF1 | 总分 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['stage']} | {labels[row['stage']]} | {row['ComponentF1_macro']:.4f} | {row['SymbolTypeBBoxF1_micro']:.4f} | {row['PinF1_macro']:.4f} | {row['NetHypergraphF1_macro']:.4f} | {row['NetLineF1_macro']:.4f} | {row['PinPairF1_macro']:.4f} | {row['FinalScore']:.2f} |")
    lines += ["", "30 个没有参与 best.pt 训练的固定 dev 案例中，BEST 的 type+bbox F1 为 %.4f，严格 Component macro F1 为 %.4f。" % (summaries[-1]["Dev30_SymbolTypeBBoxF1_micro"], summaries[-1]["Dev30_ComponentF1_macro"]), "", "D 相比 C 下降，说明当前 Name 过滤仍会把 pin/signal/model token 错配到 box；F 的几何候选精度仅约 4%，造成大量误报。最终默认选择 BEST：保留 C 的全局位号匹配和 E 中有效的 value，关闭 Name 与几何回退。"]
    (REPORTS / "component_ablation.md").write_text("\n".join(lines), encoding="utf-8")

    best_standard, best_component = loaded["BEST"]
    source_lines = ["# V4 BEST 按来源诊断", "", "OFFICIAL_SCORE = FALSE。来源仅用于报告分组，推理代码不读取来源名。", "", "| 来源 | case | ComponentF1 | type+bbox F1 | PinF1 | NetHypergraphF1 | NetLineF1 | PinPairF1 | 总分 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other"):
        if source not in best_standard["aggregate"]: continue
        standard = best_standard["aggregate"][source]; specialty = best_component["aggregate"][source]
        source_lines.append(f"| {source} | {standard['case_count']} | {standard['metrics']['Component']['macro_f1']:.4f} | {specialty['symbol_type_bbox']['f1']:.4f} | {standard['metrics']['Pin']['macro_f1']:.4f} | {standard['metrics']['NetHypergraph']['macro_f1']:.4f} | {standard['metrics']['NetLine']['macro_f1']:.4f} | {standard['metrics']['PinPair']['macro_f1']:.4f} | {standard['FinalScore']:.2f} |")
    (REPORTS / "source_breakdown.md").write_text("\n".join(source_lines), encoding="utf-8")

    errors = best_component["aggregate"]["overall"]["error_classes"]
    error_lines = ["# Component V4 错误分类", "", "| 错误类型 | 数量 |", "|---|---:|"] + [f"| {name} | {count} |" for name, count in sorted(errors.items(), key=lambda row: -row[1])]
    error_lines += ["", "当前首要错误是 designator OCR 漏检，其次是 Detection Miss、Designator Association Error 和 Type Error。type+bbox 已明显提高，但只有约一半可观察位号进入 OCR token；被识别出的位号中约 58% 分到正确 symbol。", "", "下游回归原因：BEST 增加了许多真实框，同时也增加 unresolved/错位号元件。冻结的 terminal/topology 对更多框生成 pins 和连接，使 NetHypergraph precision 与 PinPair precision 下降。下一轮应先做 detector OOF 与局部位号 OCR，再按 YOLO body bbox 重新校准 terminal search；不能直接继续堆几何候选。"]
    (REPORTS / "component_error_analysis.md").write_text("\n".join(error_lines), encoding="utf-8")
    print(json.dumps({"stages": len(summaries), "selected": summaries[-1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

