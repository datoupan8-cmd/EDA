"""A/B the current full text mask against no text mask for strict box matches.

The experiment replays saved inputs and does not change the production pin
localizer.  It measures localization only; GT pin semantics are never used by
candidate generation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_v2 import bbox_error, component_key_map, prf  # noqa: E402
from pcb.coordinates import target_bbox_to_opencv, target_to_opencv  # noqa: E402
from pcb.data_policy import assert_allowed_case_id  # noqa: E402
from pcb.io import read_image, write_json  # noqa: E402
from tools.analyze_box_terminal_stage_trace import _match, trace_boundary_wire  # noqa: E402
from tools.analyze_pin_terminal_ownership import infer_side  # noqa: E402


VARIANTS = ("current_full_text_mask", "no_text_mask")
RADII = (5.0, 10.0, 20.0)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_box_pairs(predicted: dict[str, Any], target: dict[str, Any]) -> list[tuple[str, str]]:
    mapping = component_key_map(predicted, target)
    output = []
    for pred_key, gt_key in mapping.items():
        pred, gt = predicted[pred_key], target[gt_key]
        if pred.get("type") != "box" or gt.get("type") != "box":
            continue
        if bbox_error(pred.get("bbox") or (), gt.get("bbox") or ()) > 20:
            continue
        if pred.get("Name") != gt.get("Name"):
            continue
        output.append((pred_key, gt_key))
    return output


def _gt_pins(target: dict[str, Any], component: str, height: int) -> list[dict[str, Any]]:
    gt = target["components"][component]
    box = tuple(target_bbox_to_opencv(gt["bbox"], height))
    output = []
    for pin_key, pin in (target.get("pins") or {}).get(component, {}).items():
        point = tuple(target_to_opencv(pin["point"]["x"], pin["point"]["y"], height))
        output.append({
            "id": f"{component}/{pin_key}",
            "point": point,
            "side": infer_side(point, box),
        })
    return output


def _metric(tp: int, pred: int, gt: int) -> dict[str, Any]:
    return prf(tp, pred, gt)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    totals = {
        variant: {
            "pred": 0,
            "gt": 0,
            "tp": {radius: 0 for radius in RADII},
            "component_count": 0,
            "exact_count": 0,
            "absolute_count_error": 0,
        }
        for variant in VARIANTS
    }
    per_case = []
    for number in range(1, 151):
        assert_allowed_case_id(number)
        case = f"{number:04d}"
        folder = args.target_root / case
        image = read_image(next(folder.glob("*.png")))
        target = _load(next(folder.glob("*_target*.json")))
        predicted = _load(args.prediction_dir / case / "result.json")
        diagnostics = _load(args.prediction_dir / case / "diagnostics.json")
        text_bboxes = [
            row["bbox"] for row in diagnostics.get("text_roles") or []
            if isinstance(row, dict) and isinstance(row.get("bbox"), (list, tuple))
        ]
        case_totals = {variant: {"pred": 0, "gt": 0, "tp20": 0} for variant in VARIANTS}
        for pred_key, gt_key in _strict_box_pairs(predicted.get("components") or {}, target.get("components") or {}):
            box = tuple(target_bbox_to_opencv(predicted["components"][pred_key]["bbox"], image.shape[0]))
            trace = trace_boundary_wire(image, box, text_bboxes)
            pins = _gt_pins(target, gt_key, image.shape[0])
            candidates_by_variant = {
                "current_full_text_mask": trace["final_candidates"],
                "no_text_mask": trace["raw_final_candidates"],
            }
            for variant, candidates in candidates_by_variant.items():
                block = totals[variant]
                block["component_count"] += 1
                block["pred"] += len(candidates)
                block["gt"] += len(pins)
                block["absolute_count_error"] += abs(len(candidates) - len(pins))
                block["exact_count"] += len(candidates) == len(pins)
                case_totals[variant]["pred"] += len(candidates)
                case_totals[variant]["gt"] += len(pins)
                for radius in RADII:
                    matched = _match(pins, candidates, radius=radius)
                    block["tp"][radius] += len(matched)
                    if radius == 20.0:
                        case_totals[variant]["tp20"] += len(matched)
        per_case.append({"case_id": case, **{
            f"{variant}_{field}": value
            for variant, values in case_totals.items()
            for field, value in values.items()
        }})

    summary = {}
    for variant, block in totals.items():
        result = {
            "component_count": block["component_count"],
            "pred_terminal_count": block["pred"],
            "gt_terminal_count": block["gt"],
            "pin_count_mae": block["absolute_count_error"] / max(1, block["component_count"]),
            "exact_pin_count_accuracy": block["exact_count"] / max(1, block["component_count"]),
        }
        for radius in RADII:
            result[f"terminal_{int(radius)}px"] = _metric(block["tp"][radius], block["pred"], block["gt"])
        summary[variant] = result
    return {
        "metadata": {
            "OFFICIAL_SCORE": False,
            "ORACLE": True,
            "DEPLOYABLE": False,
            "case_start": 1,
            "case_end": 150,
            "sealed_holdout_used": False,
            "scope": "strictly matched box components",
            "formal_inference_modified": False,
        },
        "summary": summary,
        "per_case": per_case,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# P0.1 Box Text-mask Counterfactual",
        "",
        "> ORACLE = TRUE；DEPLOYABLE = FALSE。只比较候选生成，不修改正式Pin Localization。",
        "",
        "| 方案 | Pred | GT | P@5 | R@5 | F1@5 | R@10 | R@20 | Count MAE | Exact Count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary[variant]
        m5, m10, m20 = row["terminal_5px"], row["terminal_10px"], row["terminal_20px"]
        lines.append(
            f"| {variant} | {row['pred_terminal_count']} | {row['gt_terminal_count']} | "
            f"{m5['precision']:.4f} | {m5['recall']:.4f} | {m5['f1']:.4f} | "
            f"{m10['recall']:.4f} | {m20['recall']:.4f} | {row['pin_count_mae']:.3f} | "
            f"{row['exact_pin_count_accuracy']:.2%} |"
        )
    lines.extend([
        "",
        "无文字屏蔽只用于上限诊断。只有在Recall提升且Precision/计数误差可接受时，才值得设计基于真实线连续性的局部像素恢复；不能直接把正式text mask关闭。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=ROOT / "reports" / "pin_p0_1_box_text_mask_ablation.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "reports" / "pin_p0_1_box_text_mask_ablation.md")
    args = parser.parse_args(argv)
    if args.target_root.name != "200_train_cases":
        raise ValueError("target-root must be exactly 200_train_cases")
    payload = analyze(args)
    write_json(args.json_output, payload)
    _write_markdown(args.markdown_output, payload)
    print(json.dumps({"json": str(args.json_output.resolve()), "summary": payload["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
