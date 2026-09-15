"""Optional deterministic continuation training for the component detector."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(dataset_yaml: Path) -> dict:
    manifest = json.loads((dataset_yaml.parent / "manifest.json").read_text(encoding="utf-8"))
    policy, split = manifest.get("policy", {}), manifest.get("split", {})
    used = split.get("train_case_ids", []) + split.get("dev_case_ids", [])
    if policy.get("allowed_case_range") != [1, 150] or policy.get("reserved_cases_accessed") is not False:
        raise PermissionError("Dataset manifest violates the 0001-0150 boundary")
    if len(set(used)) != 150 or set(used) != set(range(1, 151)):
        raise ValueError("Manifest must account for each allowed case exactly once")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--model", default=str(ROOT / "models" / "component_yolo11n_continue_v2_best.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--multi-scale", type=float, default=.15)
    parser.add_argument("--name", default="component_yolo11n_continue")
    args = parser.parse_args()
    manifest = verify_manifest(args.dataset_yaml)
    os.environ["YOLO_CONFIG_DIR"] = str((ROOT / "runs" / "ultralytics_config").resolve())
    from ultralytics import YOLO
    model = YOLO(args.model)
    dataset_names = list(manifest["class_names"])
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    if len(model_names) == len(dataset_names) and model_names != dataset_names:
        raise ValueError("Checkpoint class order differs from dataset class order")
    result = model.train(
        data=str(args.dataset_yaml.resolve()), epochs=args.epochs, imgsz=args.image_size,
        batch=args.batch, device=args.device, workers=args.workers,
        project=str((ROOT / "runs" / "component_detector").resolve()), name=args.name,
        exist_ok=False, pretrained=True, optimizer="auto", patience=args.patience,
        seed=args.seed, deterministic=True, cache=False, multi_scale=args.multi_scale,
        plots=True, verbose=True,
    )
    print(json.dumps({"status": "complete", "save_dir": str(result.save_dir), "sealed_holdout_used": False}))


if __name__ == "__main__":
    main()
