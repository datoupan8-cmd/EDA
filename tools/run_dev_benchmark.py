"""Run the frozen modular pipeline and local evaluators on cases 0001..0150."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n>", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path ending in 200_train_cases")
    parser.add_argument("--name", default="local_dev150", help="Safe run/report name")
    parser.add_argument("--device", default="auto", help="YOLO device: auto, cpu, or CUDA index")
    parser.add_argument("--ocr-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-inference", action="store_true", help="Evaluate an existing runs/<name>")
    args = parser.parse_args()

    dataset = args.dataset_root.expanduser().resolve()
    if dataset.name != "200_train_cases":
        raise SystemExit("--dataset-root must point directly to the official 200_train_cases folder")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise SystemExit("--name may contain only letters, numbers, dot, underscore, and hyphen")
    missing = [f"{number:04d}" for number in range(1, 151) if not (dataset / f"{number:04d}").is_dir()]
    if missing:
        raise SystemExit(f"Development dataset is incomplete; missing {missing[:10]}")

    prediction_dir = ROOT / "runs" / args.name
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    if not args.skip_inference:
        run([
            sys.executable, "main.py", "--input_dir", str(dataset), "--case_start", "1", "--case_end", "150",
            "--output_dir", str(prediction_dir), "--orchestrator", "modular",
            "--pipeline_config", "configs/current.json", "--weights", "models/component_yolo11n_continue_v2_best.pt",
            "--device", args.device, "--ocr_backend", "hybrid", "--ocr_device", args.ocr_device,
        ])

    metrics_path = reports / f"{args.name}_metrics.json"
    component_path = reports / f"{args.name}_component_metrics.json"
    validation_path = reports / f"{args.name}_validation.json"
    run([sys.executable, "evaluate_v2.py", "--prediction_dir", str(prediction_dir), "--target_root", str(dataset), "--case_start", "1", "--case_end", "150", "--output", str(metrics_path)])
    run([sys.executable, "evaluate_component_v4.py", "--prediction_dir", str(prediction_dir), "--target_root", str(dataset), "--case_start", "1", "--case_end", "150", "--output", str(component_path)])
    run([sys.executable, "tools/validate_v4_predictions.py", "--prediction_dir", str(prediction_dir), "--target_root", str(dataset), "--output", str(validation_path)])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    overall = metrics["aggregate"]["overall"]
    print("\nLocal non-official result")
    print(json.dumps({
        "success_count": metrics["success_count"],
        "failure_count": metrics["failure_count"],
        "ComponentF1": overall["metrics"]["Component"]["macro_f1"],
        "PinF1": overall["metrics"]["Pin"]["macro_f1"],
        "NetHypergraphF1": overall["metrics"]["NetHypergraph"]["macro_f1"],
        "NetLineF1": overall["metrics"]["NetLine"]["macro_f1"],
        "PinPairF1": overall["metrics"]["PinPair"]["macro_f1"],
        "FinalScore": overall["FinalScore"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
