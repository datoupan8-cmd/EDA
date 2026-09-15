"""Batch strict contract validation with a hard guard at case 0150."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcb.io import read_image, write_json
from pcb.submission import validate_strict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for number in range(1, 151):
        case_id = f"{number:04d}"
        try:
            image_path = next((args.target_root / case_id).glob("*.png"))
            image = read_image(image_path)
            prediction = json.loads((args.prediction_dir / case_id / "result.json").read_text(encoding="utf-8"))
            validate_strict(prediction, (image.shape[1], image.shape[0]))
            rows.append({"case_id": case_id, "valid": True})
        except Exception as exc:
            rows.append({"case_id": case_id, "valid": False, "error": repr(exc)})
    result = {"case_start": 1, "case_end": 150, "sealed_holdout_used": False, "valid_count": sum(row["valid"] for row in rows), "invalid_count": sum(not row["valid"] for row in rows), "cases": rows}
    write_json(args.output, result)
    print(json.dumps({"valid_count": result["valid_count"], "invalid_count": result["invalid_count"]}))
    return int(result["invalid_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
