"""Build allowed Component-only statistics from latest official cases 0001-0150."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pcb/v4_component_priors.json"))
    args = parser.parse_args()
    if args.target_root.name != "200_train_cases":
        raise SystemExit("target_root must be exactly 200_train_cases")
    combinations, dimensions, pin_counts = Counter(), defaultdict(list), defaultdict(list)
    for number in range(1, 151):
        case = args.target_root / f"{number:04d}"
        target = json.loads(next(case.glob("*_target.json")).read_text(encoding="utf-8"))
        for key, component in target["components"].items():
            match = re.match(r"([A-Za-z]+)", key)
            prefix = match.group(1).upper() if match else "INTERNAL_OR_SPECIAL"
            component_type = component["type"]
            combinations[(prefix, component_type)] += 1
            box = component["bbox"]
            dimensions[component_type].append((abs(float(box[2]) - float(box[0])), abs(float(box[3]) - float(box[1]))))
            pin_counts[component_type].append(len(target["pins"].get(key, {})))
    output = {
        "case_range": [1, 150], "sealed_holdout_used": False,
        "prefix_type_counts": [
            {"prefix": prefix, "type": component_type, "count": count}
            for (prefix, component_type), count in sorted(combinations.items(), key=lambda row: (-row[1], row[0]))
        ],
        "type_statistics": {
            component_type: {
                "count": len(rows),
                "median_width": sorted(row[0] for row in rows)[len(rows) // 2],
                "median_height": sorted(row[1] for row in rows)[len(rows) // 2],
                "median_pin_count": sorted(pin_counts[component_type])[len(rows) // 2],
            }
            for component_type, rows in sorted(dimensions.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"types": len(output["type_statistics"]), "pairs": len(output["prefix_type_counts"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

