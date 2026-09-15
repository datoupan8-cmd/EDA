"""Migrate verified V4.1 caches to versioned OCR and weight-hash keys."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcb.component_detector_yolo import YoloConfig
from pcb.io import read_image


WEIGHT = ROOT / "models" / "component_yolo11n_continue_v2_best.pt"
WEIGHT_SHA256 = hashlib.sha256(WEIGHT.read_bytes()).hexdigest()
EXPECTED_WEIGHT_NAMES = {WEIGHT.name, "best.pt"}


def main():
    dataset = Path(sys.argv[1])
    old_rapid_root = ROOT / "runs" / "ocr_cache_easyocr"
    new_rapid_root = ROOT / "runs" / "ocr_cache_v4_1_rapid144"
    old_yolo_root = ROOT / "runs" / "yolo_cache"
    new_yolo_root = ROOT / "runs" / "yolo_cache_v4_1"
    new_rapid_root.mkdir(parents=True, exist_ok=True)
    new_yolo_root.mkdir(parents=True, exist_ok=True)
    counts = {"rapid_migrated": 0, "rapid_missing": [], "yolo_migrated": 0, "yolo_missing": [], "rejected": []}
    config = asdict(YoloConfig())
    old_yolo_signature = json.dumps({"weights_size": WEIGHT.stat().st_size, "config": config}, sort_keys=True).encode()
    new_yolo_signature = json.dumps({"weights_sha256": WEIGHT_SHA256, "config": config}, sort_keys=True).encode()
    old_rapid_signature = b"rapidocr-1.4.4-s2-d2048-v1"
    new_rapid_signature = b"rapidocr-full-v4.1-package-1.4.4-d2048-cls0"
    for case_id in range(1, 151):
        folder = dataset / f"{case_id:04d}"
        image_path = next(folder.glob("*.png"))
        image = read_image(image_path)
        old_key = hashlib.sha256(image.tobytes() + old_rapid_signature).hexdigest()
        new_key = hashlib.sha256(image.tobytes() + new_rapid_signature).hexdigest()
        source, destination = old_rapid_root / f"{old_key}.json", new_rapid_root / f"{new_key}.json"
        if source.exists():
            rows = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(rows, list) and all(isinstance(row, dict) and {"text", "bbox", "score"} <= row.keys() for row in rows):
                destination.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                counts["rapid_migrated"] += 1
            else:
                counts["rejected"].append({"case": case_id, "kind": "rapid", "reason": "invalid payload"})
        elif not destination.exists():
            counts["rapid_missing"].append(case_id)

        old_key = hashlib.sha256(image.tobytes() + old_yolo_signature).hexdigest()
        new_key = hashlib.sha256(image.tobytes() + new_yolo_signature).hexdigest()
        source, destination = old_yolo_root / f"{old_key}.json", new_yolo_root / f"{new_key}.json"
        if source.exists():
            payload = json.loads(source.read_text(encoding="utf-8"))
            diagnostics = payload.get("diagnostics", {})
            weight_path = Path(str(diagnostics.get("weights", "")))
            valid = weight_path.name in EXPECTED_WEIGHT_NAMES and int(diagnostics.get("tile_size", -1)) == config["tile_size"] and int(diagnostics.get("overlap", -1)) == config["overlap"]
            # ``best.pt`` is accepted only when its parent path identifies the
            # teammate V2 continuation run; this prevents importing V1 caches.
            if weight_path.name == "best.pt":
                valid = valid and "component_yolo11n_continue_20260914_170933" in weight_path.as_posix()
            if valid:
                diagnostics["weights"] = str(WEIGHT.resolve())
                diagnostics["weights_sha256"] = WEIGHT_SHA256
                payload["diagnostics"] = diagnostics
                destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                counts["yolo_migrated"] += 1
            else:
                counts["rejected"].append({"case": case_id, "kind": "yolo", "reason": "weight/config mismatch", "weights": str(weight_path)})
        elif not destination.exists():
            counts["yolo_missing"].append(case_id)
    output = ROOT / "reports" / "v4_1_cache_migration.json"
    output.write_text(json.dumps({**counts, "weight_sha256": WEIGHT_SHA256, "sealed_holdout_used": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
