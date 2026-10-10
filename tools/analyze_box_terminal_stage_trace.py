"""Trace every V3 boundary-wire stage for matched-box terminal misses.

The tool replays the unchanged rule-based boundary scan from saved V6/P6.1
inputs.  It is an offline diagnostic only: no inference module or configuration
is modified, and targets are read only for cases 0001..0150.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_v2 import linear_sum_assignment, source_from_name  # noqa: E402
from pcb.coordinates import target_bbox_to_opencv, target_to_opencv  # noqa: E402
from pcb.data_policy import assert_allowed_case_id  # noqa: E402
from pcb.io import read_image, write_json  # noqa: E402
from pcb.pin_detection import _groups  # noqa: E402
from tools.analyze_pin_terminal_ownership import infer_side  # noqa: E402


CLASSIFICATIONS = (
    "NO_RAW_BOUNDARY_SUPPORT",
    "REMOVED_BY_TEXT_MASK",
    "BELOW_MINIMUM_SUPPORT",
    "EXCLUDED_BY_MARGIN",
    "COLLAPSED_BY_GROUPING",
    "COLLAPSED_BY_DEDUP",
    "FINAL_CANDIDATE_WRONG_SIDE",
    "FINAL_CANDIDATE_OFFSET_GT20",
    "TRACE_CONTRADICTION_FINAL_WITHIN20",
)
INVALID_COST = 1_000_000.0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tangent(point: tuple[float, float], side: str) -> float:
    return float(point[1] if side in ("left", "right") else point[0])


def _candidate(side: str, coordinate: float, box: tuple[int, int, int, int], pin_length: float, width: int, height: int, support: int) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    if side == "left":
        base, tip = (float(x1), coordinate), (max(0.0, x1 - pin_length), coordinate)
    elif side == "right":
        base, tip = (float(x2), coordinate), (min(float(width - 1), x2 + pin_length), coordinate)
    elif side == "top":
        base, tip = (coordinate, float(y1)), (coordinate, max(0.0, y1 - pin_length))
    else:
        base, tip = (coordinate, float(y2)), (coordinate, min(float(height - 1), y2 + pin_length))
    return {
        "tip": tip,
        "base": base,
        "side": side,
        "method": "boundary_wire_support",
        "wire_support_score": support,
        "coordinate": coordinate,
    }


def _deduplicate(candidates: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    unique = []
    for item in candidates:
        old = next((candidate for candidate in unique if item["side"] == candidate["side"] and math.dist(item["tip"], candidate["tip"]) < 7 * scale), None)
        if old is not None:
            if item["wire_support_score"] > old["wire_support_score"]:
                old.update(item)
        else:
            unique.append(dict(item))
    return unique


def _scan_rows(
    ink: np.ndarray,
    box: tuple[int, int, int, int],
    reach: int,
    minimum: int,
    margin: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, tuple[int, int]]]:
    """Return accepted coordinates, boundary-contact coordinates and ranges."""
    height, width = ink.shape
    x1, y1, x2, y2 = box
    accepted = {side: [] for side in ("left", "right", "top", "bottom")}
    contacts = {side: [] for side in accepted}
    y_start, y_stop = max(0, y1 + margin), min(height, y2 - margin + 1)
    x_start, x_stop = max(0, x1 + margin), min(width, x2 - margin + 1)
    for y in range(y_start, y_stop):
        left = ink[y, max(0, x1 - reach):min(width, x1 + 2)]
        right = ink[y, max(0, x2 - 1):min(width, x2 + reach + 1)]
        left_contact = bool(len(left) and left[-min(2, len(left)):].any())
        right_contact = bool(len(right) and right[:min(2, len(right))].any())
        if left_contact:
            contacts["left"].append(y)
            if left.sum() >= minimum:
                accepted["left"].append(y)
        if right_contact:
            contacts["right"].append(y)
            if right.sum() >= minimum:
                accepted["right"].append(y)
    for x in range(x_start, x_stop):
        top = ink[max(0, y1 - reach):min(height, y1 + 2), x]
        bottom = ink[max(0, y2 - 1):min(height, y2 + reach + 1), x]
        top_contact = bool(len(top) and top[-min(2, len(top)):].any())
        bottom_contact = bool(len(bottom) and bottom[:min(2, len(bottom))].any())
        if top_contact:
            contacts["top"].append(x)
            if top.sum() >= minimum:
                accepted["top"].append(x)
        if bottom_contact:
            contacts["bottom"].append(x)
            if bottom.sum() >= minimum:
                accepted["bottom"].append(x)
    return accepted, contacts, {
        "left": (y_start, y_stop), "right": (y_start, y_stop),
        "top": (x_start, x_stop), "bottom": (x_start, x_stop),
    }


def trace_boundary_wire(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    text_bboxes: list[list[float]],
) -> dict[str, Any]:
    """Replay V3 exactly while retaining intermediate support stages."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw_ink = (gray < 220) | ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 250))
    text_mask = np.zeros((height, width), np.uint8)
    for raw in text_bboxes:
        a, b, c, d = map(lambda value: int(round(value)), raw)
        cv2.rectangle(text_mask, (max(0, a), max(0, b)), (min(width - 1, c), min(height - 1, d)), 1, -1)
    masked_ink = raw_ink.copy()
    masked_ink[text_mask > 0] = False
    box = tuple(map(lambda value: int(round(value)), bbox))
    scale = max(0.55, min(2.2, max(height, width) / 1200))
    reach = max(8, int(round(20 * scale)))
    minimum = max(3, int(round(4 * scale)))
    margin = max(3, int(round(4 * scale)))
    pin_length = max(6.0, 13.0 * scale)
    raw_accepted, raw_contacts, ranges = _scan_rows(raw_ink, box, reach, minimum, margin)
    masked_accepted, masked_contacts, _ = _scan_rows(masked_ink, box, reach, minimum, margin)

    raw_groups = {side: _groups(values, max(2, int(round(3 * scale)))) for side, values in raw_accepted.items()}
    masked_groups = {side: _groups(values, max(2, int(round(3 * scale)))) for side, values in masked_accepted.items()}
    raw_candidates = []
    pre_dedup = []
    for side in ("left", "right", "top", "bottom"):
        for group in raw_groups[side]:
            raw_candidates.append(_candidate(side, float(np.median(group)), box, pin_length, width, height, len(group)))
        for group in masked_groups[side]:
            pre_dedup.append(_candidate(side, float(np.median(group)), box, pin_length, width, height, len(group)))
    raw_final = _deduplicate(raw_candidates, scale)
    final = _deduplicate(pre_dedup, scale)
    return {
        "scale": scale,
        "reach": reach,
        "minimum": minimum,
        "margin": margin,
        "pin_length": pin_length,
        "box": box,
        "scan_ranges": ranges,
        "raw_contacts": raw_contacts,
        "masked_contacts": masked_contacts,
        "raw_accepted": raw_accepted,
        "masked_accepted": masked_accepted,
        "raw_candidates": raw_candidates,
        "raw_final_candidates": raw_final,
        "pre_dedup_candidates": pre_dedup,
        "final_candidates": final,
    }


def _match(
    pins: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    radius: float | None = None,
    tangent_tolerance: float | None = None,
) -> dict[str, tuple[dict[str, Any], float]]:
    if not pins or not candidates:
        return {}
    limit = radius if radius is not None else float(tangent_tolerance)
    costs = [[INVALID_COST] * (len(candidates) + len(pins)) for _ in pins]
    for i, pin in enumerate(pins):
        for j, candidate in enumerate(candidates):
            if pin["side"] != candidate["side"]:
                continue
            distance = (
                math.dist(pin["point"], candidate["tip"])
                if radius is not None
                else abs(_tangent(pin["point"], pin["side"]) - _tangent(candidate["tip"], candidate["side"]))
            )
            if distance <= limit:
                costs[i][j] = distance
        costs[i][len(candidates) + i] = limit + 1.0
    rows, columns = linear_sum_assignment(costs)
    output = {}
    for i, j in zip(rows, columns):
        i, j = int(i), int(j)
        if j < len(candidates) and costs[i][j] <= limit:
            output[pins[i]["id"]] = (candidates[j], float(costs[i][j]))
    return output


def _near_coordinate(values: list[int], value: float, tolerance: float) -> bool:
    return any(abs(float(item) - value) <= tolerance for item in values)


def _intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return left[0] <= right[2] and right[0] <= left[2] and left[1] <= right[3] and right[1] <= left[3]


def _masking_tokens(
    text_rows: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    side: str,
    tangent: float,
    reach: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = box
    if side == "left":
        corridor = (x1 - reach, tangent - tolerance, x1 + 2, tangent + tolerance)
    elif side == "right":
        corridor = (x2 - 1, tangent - tolerance, x2 + reach, tangent + tolerance)
    elif side == "top":
        corridor = (tangent - tolerance, y1 - reach, tangent + tolerance, y1 + 2)
    else:
        corridor = (tangent - tolerance, y2 - 1, tangent + tolerance, y2 + reach)
    output = []
    for row in text_rows:
        bbox = row.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and _intersects(corridor, tuple(map(float, bbox))):
            output.append({
                "text": str(row.get("text") or ""),
                "role": str(row.get("role") or ""),
                "confidence": row.get("confidence"),
                "bbox": list(map(float, bbox)),
            })
    return output


def _trace_parity(final: list[dict[str, Any]], events: list[dict[str, Any]]) -> bool:
    left = sorted((row["side"], round(row["tip"][0], 6), round(row["tip"][1], 6)) for row in final)
    right = sorted((str(row.get("side") or ""), round(float(row["tip"][0]), 6), round(float(row["tip"][1]), 6)) for row in events)
    return left == right


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    ownership = _load(args.ownership_json)
    if ownership["metadata"].get("case_start") != 1 or ownership["metadata"].get("case_end") != 150:
        raise ValueError("Ownership input must cover exactly cases 0001..0150")
    targets = [
        record for record in ownership["records"].values()
        if record["original_bucket"] == "A1_TERMINAL_MISS"
        and record["ownership_class"] == "TRUE_TERMINAL_MISS"
        and record["gt_component_type"] == "box"
    ]
    if len(targets) != 1259:
        raise ValueError(f"Expected 1259 matched-box misses, got {len(targets)}")
    target_ids = {row["gt_pin_id"] for row in targets}
    targets_by_case_component: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        targets_by_case_component[(row["case_id"], row["gt_component"])].append(row)

    output = []
    component_parity = Counter()
    traced_components = 0
    for number in range(1, 151):
        assert_allowed_case_id(number)
        case = f"{number:04d}"
        case_keys = [key for key in targets_by_case_component if key[0] == case]
        if not case_keys:
            continue
        folder = args.target_root / case
        image_path = next(folder.glob("*.png"))
        target = _load(next(folder.glob("*_target*.json")))
        predicted = _load(args.prediction_dir / case / "result.json")
        diagnostics = _load(args.prediction_dir / case / "diagnostics.json")
        image = read_image(image_path)
        height = image.shape[0]
        text_rows = [row for row in diagnostics.get("text_roles") or [] if isinstance(row, dict) and isinstance(row.get("bbox"), (list, tuple))]
        text_bboxes = [row["bbox"] for row in text_rows]
        events_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in diagnostics.get("pin_events") or []:
            if isinstance(event, dict) and isinstance(event.get("tip"), (list, tuple)):
                events_by_component[str(event.get("component") or "")].append(event)

        for _, gt_component in case_keys:
            requested = targets_by_case_component[(case, gt_component)]
            pred_component = requested[0].get("pred_component") or gt_component
            # A1 uses the evaluator's identity/GND mapping.  Ordinary box keys
            # are therefore identical; retain an explicit fallback for older audit files.
            if pred_component not in (predicted.get("components") or {}):
                pred_component = gt_component
            pred = (predicted.get("components") or {}).get(pred_component)
            gt = (target.get("components") or {}).get(gt_component)
            if pred is None or gt is None:
                raise ValueError(f"{case}/{gt_component}: missing matched component")
            pred_box = tuple(target_bbox_to_opencv(pred["bbox"], height))
            gt_box = tuple(target_bbox_to_opencv(gt["bbox"], height))
            trace = trace_boundary_wire(image, pred_box, text_bboxes)
            existing_events = [row for row in events_by_component.get(pred_component, []) if row.get("method") == "boundary_wire_support"]
            parity = _trace_parity(trace["final_candidates"], existing_events)
            component_parity["pass" if parity else "fail"] += 1
            traced_components += 1

            gt_pins = []
            for pin_key, pin in (target.get("pins") or {}).get(gt_component, {}).items():
                point = tuple(target_to_opencv(pin["point"]["x"], pin["point"]["y"], height))
                gt_pins.append({
                    "id": f"{case}/{gt_component}/{pin_key}",
                    "pin_key": pin_key,
                    "point": point,
                    "side": infer_side(point, gt_box),
                })
            tolerance = max(4.0, 3.0 * trace["scale"])
            raw_match = _match(gt_pins, trace["raw_candidates"], tangent_tolerance=tolerance)
            grouped_match = _match(gt_pins, trace["pre_dedup_candidates"], tangent_tolerance=tolerance)
            final_tangent_match = _match(gt_pins, trace["final_candidates"], tangent_tolerance=tolerance)
            final_20_match = _match(gt_pins, trace["final_candidates"], radius=20.0)

            for pin in gt_pins:
                if pin["id"] not in target_ids:
                    continue
                side = pin["side"]
                tangent = _tangent(pin["point"], side)
                scan_start, scan_stop = trace["scan_ranges"][side]
                raw_contact = _near_coordinate(trace["raw_contacts"][side], tangent, tolerance)
                raw_accepted = _near_coordinate(trace["raw_accepted"][side], tangent, tolerance)
                masked_accepted = _near_coordinate(trace["masked_accepted"][side], tangent, tolerance)
                masking_tokens = _masking_tokens(
                    text_rows, trace["box"], side, tangent, trace["reach"], tolerance
                )
                nearest_any = min(
                    trace["final_candidates"],
                    key=lambda row: math.dist(pin["point"], row["tip"]),
                    default=None,
                )
                nearest_any_distance = math.dist(pin["point"], nearest_any["tip"]) if nearest_any else None
                if pin["id"] in final_20_match:
                    classification = "TRACE_CONTRADICTION_FINAL_WITHIN20"
                elif pin["id"] in final_tangent_match:
                    classification = "FINAL_CANDIDATE_OFFSET_GT20"
                elif nearest_any is not None and nearest_any_distance <= 40.0 and nearest_any["side"] != side:
                    classification = "FINAL_CANDIDATE_WRONG_SIDE"
                elif pin["id"] in grouped_match:
                    classification = "COLLAPSED_BY_DEDUP"
                elif not (scan_start <= tangent < scan_stop):
                    classification = "EXCLUDED_BY_MARGIN"
                elif raw_accepted and not masked_accepted:
                    classification = "REMOVED_BY_TEXT_MASK"
                elif masked_accepted:
                    classification = "COLLAPSED_BY_GROUPING"
                elif raw_contact:
                    classification = "BELOW_MINIMUM_SUPPORT"
                else:
                    classification = "NO_RAW_BOUNDARY_SUPPORT"
                final_candidate = final_tangent_match.get(pin["id"], (None, None))[0]
                output.append({
                    "gt_pin_id": pin["id"],
                    "case_id": case,
                    "source": source_from_name(image_path.name),
                    "component": gt_component,
                    "pin_key": pin["pin_key"],
                    "side": side,
                    "classification": classification,
                    "component_strict_tp": bool(requested[0].get("component_strict_tp")),
                    "trace_parity": parity,
                    "scale": trace["scale"],
                    "reach": trace["reach"],
                    "minimum": trace["minimum"],
                    "margin": trace["margin"],
                    "pin_length": trace["pin_length"],
                    "tangent_tolerance": tolerance,
                    "raw_contact_near": raw_contact,
                    "raw_accepted_near": raw_accepted,
                    "masked_accepted_near": masked_accepted,
                    "masking_tokens": masking_tokens,
                    "raw_group_matched": pin["id"] in raw_match,
                    "pre_dedup_matched": pin["id"] in grouped_match,
                    "final_tangent_matched": pin["id"] in final_tangent_match,
                    "final_within20": pin["id"] in final_20_match,
                    "final_candidate_distance": (
                        math.dist(pin["point"], final_candidate["tip"]) if final_candidate else None
                    ),
                    "nearest_final_any_side_distance": nearest_any_distance,
                })

    counts = Counter(row["classification"] for row in output)
    masking_roles = Counter()
    masking_texts = Counter()
    for row in output:
        if row["classification"] != "REMOVED_BY_TEXT_MASK":
            continue
        for token in row["masking_tokens"]:
            masking_roles[token["role"] or "(unknown)"] += 1
            masking_texts[token["text"] or "(empty)"] += 1
    if len(output) != 1259:
        raise AssertionError(f"Expected 1259 output records, got {len(output)}")
    by_case = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[row["case_id"]].append(row)
    for case, rows in sorted(grouped.items()):
        row_counts = Counter(row["classification"] for row in rows)
        by_case.append({
            "case_id": case,
            "target_count": len(rows),
            **{name: row_counts[name] for name in CLASSIFICATIONS},
        })
    return {
        "metadata": {
            "OFFICIAL_SCORE": False,
            "case_start": 1,
            "case_end": 150,
            "sealed_holdout_used": False,
            "prediction_dir": str(args.prediction_dir.resolve()),
            "target_root": str(args.target_root.resolve()),
            "formal_inference_modified": False,
            "scope": "A1 TRUE_TERMINAL_MISS box pins",
        },
        "summary": {
            "target_pin_count": len(output),
            "traced_component_count": traced_components,
            "trace_parity": dict(component_parity),
            "classification_counts": {name: counts[name] for name in CLASSIFICATIONS},
            "strict_component_target_count": sum(row["component_strict_tp"] for row in output),
            "strict_component_classification_counts": {
                name: sum(row["component_strict_tp"] and row["classification"] == name for row in output)
                for name in CLASSIFICATIONS
            },
            "raw_group_survival": sum(row["raw_group_matched"] for row in output),
            "pre_dedup_survival": sum(row["pre_dedup_matched"] for row in output),
            "final_tangent_survival": sum(row["final_tangent_matched"] for row in output),
            "final_within20": sum(row["final_within20"] for row in output),
            "text_mask_support_removed": counts["REMOVED_BY_TEXT_MASK"],
            "text_mask_direct_unmasked_group_recoverable": sum(
                row["classification"] == "REMOVED_BY_TEXT_MASK" and row["raw_group_matched"]
                for row in output
            ),
            "text_mask_removed_but_unmasked_group_ambiguous": sum(
                row["classification"] == "REMOVED_BY_TEXT_MASK" and not row["raw_group_matched"]
                for row in output
            ),
            "masking_token_roles": dict(masking_roles.most_common()),
            "top_masking_token_texts": dict(masking_texts.most_common(30)),
        },
        "by_case": by_case,
        "records": {row["gt_pin_id"]: row for row in output},
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    total = summary["target_pin_count"]
    lines = [
        "# P0.0 Matched-box Terminal Stage Trace",
        "",
        "> 本地非官方诊断。只分析0001～0150中A1/TRUE_TERMINAL_MISS的1259个box Pin；未修改正式算法。",
        "",
        "## 重放一致性",
        "",
        f"- 追踪Component：{summary['traced_component_count']}",
        f"- 与现有pin_events逐坐标一致：{summary['trace_parity'].get('pass', 0)}",
        f"- 不一致：{summary['trace_parity'].get('fail', 0)}",
        "",
        "## 分类结果",
        "",
        "| 分类 | 数量 | 比例 |",
        "|---|---:|---:|",
    ]
    for name in CLASSIFICATIONS:
        count = summary["classification_counts"][name]
        lines.append(f"| {name} | {count} | {count / total:.2%} |")
    strict_total = summary["strict_component_target_count"]
    lines.extend([
        "",
        f"其中严格Component TP范围为{strict_total}个Pin；其分类如下：",
        "",
        "| 分类 | 严格Component数量 | 比例 |",
        "|---|---:|---:|",
    ])
    for name in CLASSIFICATIONS:
        count = summary["strict_component_classification_counts"][name]
        lines.append(f"| {name} | {count} | {count / max(1, strict_total):.2%} |")
    lines.extend([
        "",
        "## 阶段存活",
        "",
        f"- raw group可一对一对齐：{summary['raw_group_survival']} / {total}",
        f"- 文字屏蔽后、dedup前可对齐：{summary['pre_dedup_survival']} / {total}",
        f"- 最终candidate切向可对齐：{summary['final_tangent_survival']} / {total}",
        f"- 最终candidate在20px内：{summary['final_within20']} / {total}",
        f"- 文字屏蔽移除局部支持：{summary['text_mask_support_removed']} / {total}",
        f"- 不屏蔽时可直接形成独立group候选：{summary['text_mask_direct_unmasked_group_recoverable']} / {total}",
        f"- 不屏蔽时仍受grouping歧义影响：{summary['text_mask_removed_but_unmasked_group_ambiguous']} / {total}",
        "",
        "## 覆盖Pin出口的OCR角色",
        "",
    ])
    for role, count in summary["masking_token_roles"].items():
        lines.append(f"- {role}：{count}")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "该工具重放V3规则式边界扫描，不涉及Pin检测模型、训练或NMS。`FINAL_CANDIDATE_OFFSET_GT20`表示候选沿边位置存在，但最终tip距离超过20px；`NO_RAW_BOUNDARY_SUPPORT`才表示当前边界扫描没有形成可用原始支持。文字屏蔽桶还分成可直接恢复独立group和仍受grouping歧义影响两类，不能把全部桶数量当作关闭text mask后的可恢复量。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ownership-json", type=Path, default=ROOT / "reports" / "pin_terminal_ownership_audit.json")
    parser.add_argument("--json-output", type=Path, default=ROOT / "reports" / "pin_p0_0_box_stage_trace.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "reports" / "pin_p0_0_box_stage_trace.md")
    parser.add_argument("--per-case-output", type=Path, default=ROOT / "reports" / "pin_p0_0_box_stage_trace_per_case.csv")
    args = parser.parse_args(argv)
    if args.target_root.name != "200_train_cases":
        raise ValueError("target-root must be exactly 200_train_cases")
    payload = analyze(args)
    write_json(args.json_output, payload)
    _write_markdown(args.markdown_output, payload)
    _write_csv(args.per_case_output, payload["by_case"])
    print(json.dumps({
        "json": str(args.json_output.resolve()),
        "markdown": str(args.markdown_output.resolve()),
        "summary": payload["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
