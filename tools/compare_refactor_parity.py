"""Compare legacy and modular prediction trees without consulting holdout data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right, "reason": "type"}
    if isinstance(left, dict):
        if list(left) != list(right):
            return {"path": path, "left_keys": list(left), "right_keys": list(right), "reason": "keys_or_order"}
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right), "reason": "length"}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
    elif left != right:
        return {"path": path, "left": left, "right": right, "reason": "value"}
    return None


VOLATILE_DIAGNOSTIC_KEYS = {"seconds", "cache_hit", "cached_tile_count", "weights"}


def normalized_diagnostics(value):
    """Drop runtime/cache provenance while retaining every algorithm decision."""
    if isinstance(value, dict):
        return {
            key: normalized_diagnostics(value[key])
            for key in sorted(value)
            if key not in VOLATILE_DIAGNOSTIC_KEYS
        }
    if isinstance(value, list):
        return [normalized_diagnostics(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--case-start", type=int, default=1)
    parser.add_argument("--case-end", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.case_start <= args.case_end < 151):
        raise SystemExit("Holdout guard: parity accepts only cases 0001..0150")
    cases = []
    for number in range(args.case_start, args.case_end + 1):
        case = f"{number:04d}"
        left_path = args.baseline / case / "result.json"
        right_path = args.candidate / case / "result.json"
        try:
            left = json.loads(left_path.read_text(encoding="utf-8"))
            right = json.loads(right_path.read_text(encoding="utf-8"))
            difference = first_difference(left, right)
            left_diag_path = args.baseline / case / "diagnostics.json"
            right_diag_path = args.candidate / case / "diagnostics.json"
            diagnostics_difference = None
            if left_diag_path.exists() and right_diag_path.exists():
                left_diag = normalized_diagnostics(json.loads(left_diag_path.read_text(encoding="utf-8")))
                right_diag = normalized_diagnostics(json.loads(right_diag_path.read_text(encoding="utf-8")))
                diagnostics_difference = first_difference(left_diag, right_diag)
            cases.append({
                "case_id": case,
                "equal": difference is None,
                "diagnostics_semantic_equal": diagnostics_difference is None,
                "baseline_sha256": digest(left_path),
                "candidate_sha256": digest(right_path),
                "first_difference": difference,
                "diagnostics_first_difference": diagnostics_difference,
            })
        except Exception as exc:
            cases.append({"case_id": case, "equal": False, "diagnostics_semantic_equal": False, "error": repr(exc)})
    equal_count = sum(row["equal"] for row in cases)
    diagnostics_equal_count = sum(row["diagnostics_semantic_equal"] for row in cases)
    payload = {
        "case_start": args.case_start,
        "case_end": args.case_end,
        "sealed_holdout_used": False,
        "comparison": "strict parsed JSON including dictionary insertion order",
        "equal_count": equal_count,
        "different_count": len(cases) - equal_count,
        "all_equal": equal_count == len(cases),
        "diagnostics_semantic_equal_count": diagnostics_equal_count,
        "diagnostics_semantic_different_count": len(cases) - diagnostics_equal_count,
        "diagnostics_semantic_all_equal": diagnostics_equal_count == len(cases),
        "diagnostics_ignored_runtime_keys": sorted(VOLATILE_DIAGNOSTIC_KEYS),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("equal_count", "different_count", "all_equal", "diagnostics_semantic_equal_count", "diagnostics_semantic_all_equal")}, indent=2))
    return int(not payload["all_equal"])


if __name__ == "__main__":
    raise SystemExit(main())
