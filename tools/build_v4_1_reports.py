"""Build compact V4.1 JSON, Markdown, and per-case CSV reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name: str):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def compact(block):
    return {
        "case_count": block["case_count"],
        "ComponentF1": block["strict_component"]["macro_f1"],
        "SymbolTypeBBoxF1": block["symbol_type_bbox"]["f1"],
        "BBoxF1": block["bbox"]["f1"],
        "TypeAccuracy": block["type_accuracy"]["accuracy"],
        "DesignatorOCRMicroF1": block["designator_ocr"]["f1"],
        "DesignatorOCRMacroF1": block["designator_ocr"]["macro_f1"],
        "DesignatorAssociationMicroAccuracy": block["designator_association"]["f1"],
        "DesignatorAssociationMacroAccuracy": block["designator_association"]["macro_f1"],
        "NameAccuracy": block["name_accuracy"]["accuracy"],
        "ValueAccuracy": block["value_accuracy"]["accuracy"],
        "PinF1": block["Pin"]["macro_f1"],
        "NetHypergraphF1": block["NetHypergraph"]["macro_f1"],
        "NetLineF1": block["NetLine"]["macro_f1"],
        "PinPairF1": block["PinPair"]["macro_f1"],
        "FinalScore": block["FinalScore_macro"],
        "FalsePositivePerImage": block["false_positive_per_image"],
        "DuplicateComponentPairs": block["duplicate_component_pairs"],
    }


def main():
    final = load("v4_1_hybrid_rapid144_best_150_component.json")
    old = load("v4_BEST_component_metrics.json")
    weight_only = load("v4_newweight_component_metrics.json")
    rows = {
        "V4_original": compact(old["aggregate"]["overall"]),
        "V4_new_weight_only": compact(weight_only["aggregate"]["overall"]),
        "V4_1_hybrid": compact(final["aggregate"]["overall"]),
        "V4_1_hybrid_dev30_no_training_overlap": compact(final["aggregate"]["dev_30_no_training_overlap"]),
    }
    summary = {
        "OFFICIAL_SCORE": False,
        "sealed_holdout_used": False,
        "main_run": "runs/v4_1_hybrid_rapid144_best_150",
        "contract_validation": "150/150",
        "variants": rows,
    }
    (REPORTS / "v4_1_metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["case_id", "source", "ComponentF1", "SymbolTypeBBoxF1", "BBoxF1", "DesignatorOCRF1", "DesignatorAssociationF1", "PinF1", "NetHypergraphF1", "NetLineF1", "PinPairF1", "FinalScore"]
    with (REPORTS / "v4_1_per_case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in final["cases"]:
            ev = row["evaluation"]
            writer.writerow({
                "case_id": row["case_id"], "source": row["source"],
                "ComponentF1": ev["strict_component"]["f1"],
                "SymbolTypeBBoxF1": ev["symbol_type_bbox"]["f1"],
                "BBoxF1": ev["bbox"]["f1"],
                "DesignatorOCRF1": ev["designator_ocr"]["f1"],
                "DesignatorAssociationF1": ev["designator_association"]["f1"],
                "PinF1": ev["downstream"]["Pin"]["f1"],
                "NetHypergraphF1": ev["downstream"]["NetHypergraph"]["f1"],
                "NetLineF1": ev["downstream"]["NetLine"]["f1"],
                "PinPairF1": ev["downstream"]["PinPair"]["f1"],
                "FinalScore": ev["FinalScore"],
            })

    source_lines = [
        "# Component V4.1 分来源指标", "",
        "以下均为本地非官方诊断结果。", "",
        "| 来源 | 案例 | Component F1 | Symbol F1 | Pin F1 | Net F1 | Line F1 | PinPair F1 | 总分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other"):
        block = final["aggregate"].get(source)
        if not block:
            continue
        source_lines.append(
            f"| {source} | {block['case_count']} | {block['strict_component']['macro_f1']:.4f} | {block['symbol_type_bbox']['macro_f1']:.4f} | "
            f"{block['Pin']['macro_f1']:.4f} | {block['NetHypergraph']['macro_f1']:.4f} | {block['NetLine']['macro_f1']:.4f} | "
            f"{block['PinPair']['macro_f1']:.4f} | {block['FinalScore_macro']:.2f} |"
        )
    source_lines.extend(["", "SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE", ""])
    (REPORTS / "source_breakdown_v4_1.md").write_text("\n".join(source_lines), encoding="utf-8")

    labels = [
        ("ComponentF1", "ComponentF1"), ("Symbol type+bbox F1", "SymbolTypeBBoxF1"),
        ("BBox F1", "BBoxF1"), ("位号 OCR micro F1", "DesignatorOCRMicroF1"),
        ("位号关联 micro 准确率", "DesignatorAssociationMicroAccuracy"), ("Value 准确率", "ValueAccuracy"),
        ("PinF1", "PinF1"), ("NetHypergraphF1", "NetHypergraphF1"),
        ("NetLineF1", "NetLineF1"), ("PinPairF1", "PinPairF1"), ("总分", "FinalScore"),
    ]
    before, after = rows["V4_original"], rows["V4_1_hybrid"]
    lines = [
        "# Component V4 与 V4.1 对比", "",
        "对比使用最新版官方 0001 至 0150。YOLO 权重训练过其中 120 例，因此完整 150 例只属于开发诊断；无训练重叠结论以 dev30 为准。", "",
        "| 指标 | 原 V4 | V4.1 | 变化 |", "|---|---:|---:|---:|",
    ]
    for label, key in labels:
        lines.append(f"| {label} | {before[key]:.4f} | {after[key]:.4f} | {after[key]-before[key]:+.4f} |")
    lines.extend(["", "`OFFICIAL_SCORE = FALSE`", "", "`SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE`", ""])
    (REPORTS / "v4_vs_v4_1_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
