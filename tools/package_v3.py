"""Create the curated V3 delivery archive and verify its contents."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "PCB_Competition_Solution_V3.zip"
PREFIX = "PCB_Competition_Solution_V3"

TOP_LEVEL = {
    "README.md", "requirements.txt", "main.py", "evaluate_v2.py",
    "oracle_experiments.py", "LICENSE",
}
REPORT_NAMES = {
    "dataset_version_diff.md", "dataset_version_diff.json", "dataset_rule_changes.md",
    "component_text_v3.md", "pin_v3.md", "wire_v3.md", "topology_v3.md",
    "error_cluster_analysis.md", "error_cluster_analysis.json",
    "paper_to_code_mapping.md", "v2_vs_v3_metrics.md", "case_0002_before_after.md",
    "source_breakdown.md", "v3_next_stage.md", "v3_metrics.json",
    "v3_A_metrics.json", "v3_B_metrics.json", "v3_C_metrics.json", "v3_D_metrics.json",
    "v3_ablation.md", "v3_ablation_metrics.json", "v3_final_validation.json",
    "v3_oracle_sanity.json", "v3_error_budget.md", "v3_per_case_metrics.json",
    "v3_per_case_metrics.csv", "debug_case_selection.json", "v3_test_results.txt",
    "v2_baseline_metrics.json", "v2_oracle_metrics.json", "v2_error_budget.md",
    "v2_baseline_summary.md",
    "v3_delivery_manifest.json",
}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if len(parts) == 1:
        return path.name in TOP_LEVEL
    if parts[0] == "pcb":
        return path.suffix in {".py", ".json"}
    if parts[0] == "tests":
        return path.suffix == ".py"
    if parts[0] == "tools":
        return path.name in {
            "validate_predictions.py", "inspect_dev_case.py", "build_v3_priors.py",
            "build_v3_reports.py", "package_v3.py",
        }
    if parts[0] == "reports":
        return path.name in REPORT_NAMES
    if parts[:2] == ("runs", "v3_D_topology"):
        return path.name in {"result.json", "diagnostics.json", "batch_summary.json"}
    if parts[:2] == ("runs", "v3_debug"):
        return path.suffix in {".json", ".png"}
    return False


def main():
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and include(p))
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, (Path(PREFIX) / path.relative_to(ROOT)).as_posix())
    with zipfile.ZipFile(ARCHIVE) as zf:
        bad = zf.testzip()
        names = set(zf.namelist())
        result_count = sum(name.startswith(f"{PREFIX}/runs/v3_D_topology/") and name.endswith("/result.json") for name in names)
        required = {
            f"{PREFIX}/README.md", f"{PREFIX}/main.py", f"{PREFIX}/requirements.txt",
            f"{PREFIX}/reports/v3_metrics.json", f"{PREFIX}/reports/v3_final_validation.json",
        }
        missing = sorted(required - names)
    manifest = {
        "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "file_count": len(files), "result_json_count": result_count,
        "crc_error": bad, "missing_required": missing,
        "includes_final_run": "runs/v3_D_topology",
        "includes_debug_run": "runs/v3_debug",
        "excluded": ["all OCR caches", "V2/dev/smoke runs", "raw A/B/C runs; their complete metrics are included"],
    }
    (ROOT / "reports" / "v3_delivery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if bad or missing or result_count != 150:
        raise SystemExit(json.dumps(manifest, ensure_ascii=False))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
