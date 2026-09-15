"""Compare OCR strategies on the fixed 30-case detector dev split."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEV_CASE_IDS = ",".join(f"{case_id:04d}" for case_id in range(5, 151, 5))


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ocr-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--weights", type=Path, default=Path("models/component_yolo11n_continue_v2_best.pt"))
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    if args.target_root.name != "200_train_cases":
        raise SystemExit("target_root must be exactly 200_train_cases")
    variants = ("rapidocr", "easyocr_tiled", "hybrid", "selective_local")
    for backend in variants:
        output_dir = Path("runs") / f"ocr_ablation_{backend}_dev30"
        if not args.skip_inference:
            run([
                sys.executable, "main.py", "--input_dir", str(args.target_root),
                "--case_ids", DEV_CASE_IDS, "--output_dir", str(output_dir),
                "--pipeline", "v4_component", "--component_stage", "BEST",
                "--weights", str(args.weights), "--device", args.device,
                "--ocr_backend", backend, "--ocr_device", args.ocr_device,
                "--text_rules", "v4_1",
            ])
        # The evaluator sees only result directories that exist.  It reports
        # the absent non-dev cases as missing, while aggregate.overall remains
        # the intended 30-case result.
        run([
            sys.executable, "evaluate_component_v4.py", "--prediction_dir", str(output_dir),
            "--target_root", str(args.target_root), "--case_ids", DEV_CASE_IDS,
            "--output", f"reports/ocr_ablation_{backend}_dev30_component.json",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
