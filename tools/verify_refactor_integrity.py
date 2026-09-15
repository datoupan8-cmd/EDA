"""Verify that the modular refactor did not modify algorithm source or scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "reports/refactor_baseline_manifest.json").read_text(encoding="utf-8"))
    source_rows = []
    for name, before in baseline["source_sha256"].items():
        after = sha256(root / name)
        source_rows.append({
            "file": name,
            "before_sha256": before,
            "after_sha256": after,
            "identical": before == after,
            "expected_change": name == "main.py",
        })
    legacy_metrics = json.loads((root / "reports/v4_1_hybrid_rapid144_best_150_metrics.json").read_text(encoding="utf-8"))
    modular_metrics = json.loads((root / "reports/refactor_modular_metrics.json").read_text(encoding="utf-8"))
    legacy_overall = legacy_metrics["aggregate"]["overall"]
    modular_overall = modular_metrics["aggregate"]["overall"]
    metric_names = ("Component", "Pin", "NetHypergraph", "NetLine", "PinPair")
    metric_rows = {
        name: {
            "legacy_macro_f1": legacy_overall["metrics"][name]["macro_f1"],
            "modular_macro_f1": modular_overall["metrics"][name]["macro_f1"],
            "identical": legacy_overall["metrics"][name]["macro_f1"] == modular_overall["metrics"][name]["macro_f1"],
        }
        for name in metric_names
    }
    parity = json.loads((root / "reports/refactor_parity_report.json").read_text(encoding="utf-8"))
    payload = {
        "behavior_preserving": (
            parity["all_equal"]
            and parity.get("diagnostics_semantic_all_equal", False)
            and all(row["identical"] for row in metric_rows.values())
            and legacy_overall["FinalScore"] == modular_overall["FinalScore"]
            and all(row["identical"] or row["expected_change"] for row in source_rows)
        ),
        "prediction_parity": {
            "case_count": parity["equal_count"],
            "different_count": parity["different_count"],
            "all_equal": parity["all_equal"],
            "diagnostics_semantic_equal_count": parity.get("diagnostics_semantic_equal_count", 0),
            "diagnostics_semantic_all_equal": parity.get("diagnostics_semantic_all_equal", False),
        },
        "metric_parity": metric_rows,
        "final_score": {
            "legacy": legacy_overall["FinalScore"],
            "modular": modular_overall["FinalScore"],
            "identical": legacy_overall["FinalScore"] == modular_overall["FinalScore"],
            "official": False,
        },
        "source_integrity": source_rows,
        "weights": {
            "component_yolo11n_baseline_best.pt": sha256(root / "models/component_yolo11n_baseline_best.pt"),
            "component_yolo11n_continue_v2_best.pt": sha256(root / "models/component_yolo11n_continue_v2_best.pt"),
        },
        "sealed_holdout_used": False,
    }
    output = root / "reports/refactor_validation_summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"behavior_preserving": payload["behavior_preserving"], **payload["prediction_parity"]}, indent=2))
    return int(not payload["behavior_preserving"])


if __name__ == "__main__":
    raise SystemExit(main())
