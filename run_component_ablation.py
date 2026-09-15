"""Reproduce Component V4 stages B-F/BEST on allowed cases 0001-0150."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", type=Path, default=Path("models/component_yolo11n_continue_v2_best.pt"))
    parser.add_argument("--ocr-backend", choices=["rapidocr", "easyocr_tiled", "hybrid", "selective_local"], default="hybrid")
    parser.add_argument("--ocr-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--text-rules", choices=["v4", "v4_1"], default="v4_1")
    parser.add_argument("--case_start", type=int, default=1)
    parser.add_argument("--case_end", type=int, default=150)
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.case_start <= args.case_end <= 150):
        raise SystemExit("Only cases 0001..0150 are allowed")
    if args.target_root.name != "200_train_cases":
        raise SystemExit("target_root must be exactly 200_train_cases")
    run_names = {"B": "v4_B_yolo_only", "C": "v4_C_designator", "D": "v4_D_name", "E": "v4_E_value", "F": "v4_F_geometry", "BEST": "v4_BEST"}
    for stage, run_name in run_names.items():
        prediction_dir = Path("runs") / run_name
        if not args.skip_inference:
            run([
                sys.executable, "main.py", "--input_dir", str(args.target_root),
                "--case_start", str(args.case_start), "--case_end", str(args.case_end),
                "--output_dir", str(prediction_dir), "--pipeline", "v4_component",
                "--component_stage", stage, "--device", args.device,
                "--weights", str(args.weights), "--cache_dir", "runs/ocr_cache_v4_1",
                "--yolo_cache_dir", "runs/yolo_cache",
                "--ocr_backend", args.ocr_backend, "--ocr_device", args.ocr_device,
                "--text_rules", args.text_rules,
            ])
        run([sys.executable, "evaluate_v2.py", "--prediction_dir", str(prediction_dir), "--target_root", str(args.target_root), "--case_start", str(args.case_start), "--case_end", str(args.case_end), "--output", f"reports/v4_{stage}_metrics.json"])
        run([sys.executable, "evaluate_component_v4.py", "--prediction_dir", str(prediction_dir), "--target_root", str(args.target_root), "--case_start", str(args.case_start), "--case_end", str(args.case_end), "--output", f"reports/v4_{stage}_component_metrics.json"])
    if args.case_start == 1 and args.case_end == 150:
        run([sys.executable, "tools/build_v4_reports.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
