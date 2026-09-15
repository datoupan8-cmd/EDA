"""Capture immutable source and prediction hashes before modular refactoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CORE_FILES = (
    "main.py",
    "pcb/schema.py",
    "pcb/vision_v4.py",
    "pcb/component_detection_v4.py",
    "pcb/component_detector_yolo.py",
    "pcb/component_proposal_fusion.py",
    "pcb/component_text_assignment.py",
    "pcb/text_detection.py",
    "pcb/pin_detection.py",
    "pcb/pin_semantics.py",
    "pcb/wire.py",
    "pcb/wire_v2.py",
    "pcb/wire_v3.py",
    "pcb/topology.py",
    "pcb/topology_v2.py",
    "pcb/topology_v3.py",
    "pcb/submission.py",
    "pcb/coordinates.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v4_1_hybrid_rapid144_best_150"))
    parser.add_argument("--output", type=Path, default=Path("reports/refactor_baseline_manifest.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = {name: sha256(root / name) for name in CORE_FILES}
    predictions = {}
    for result in sorted((root / args.run_dir).glob("*/result.json")):
        predictions[result.parent.name] = sha256(result)
    payload = {
        "baseline_run": args.run_dir.as_posix(),
        "source_sha256": source,
        "prediction_count": len(predictions),
        "prediction_sha256": predictions,
        "sealed_holdout_used": False,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(f"prediction_count={len(predictions)}")
    return 0 if len(predictions) == 150 else 1


if __name__ == "__main__":
    raise SystemExit(main())
