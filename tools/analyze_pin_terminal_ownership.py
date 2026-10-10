"""Audit whether missed GT pins have a plausible, uniquely owned raw terminal.

This is an offline ORACLE-FREE ownership diagnostic over public development
cases 0001..0150.  It never changes inference output.  GT is used only by this
analysis process to classify existing raw terminal events.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_v2 import component_key_map, linear_sum_assignment, source_from_name  # noqa: E402
from pcb.coordinates import target_bbox_to_opencv, target_to_opencv  # noqa: E402
from pcb.data_policy import assert_allowed_case_id  # noqa: E402
from pcb.io import read_image, write_json  # noqa: E402


RADII = (5.0, 10.0, 20.0)
UNMATCHED_COST = 21.0
INVALID_COST = 1_000_000.0
OWNERSHIP_CLASSES = (
    "CORRECT_OWNER_TERMINAL",
    "UNRESOLVED_OWNER_TERMINAL",
    "WRONG_KNOWN_OWNER_TERMINAL",
    "NEARBY_BUT_GEOMETRY_INVALID",
    "TRUE_TERMINAL_MISS",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _bbox_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def _bbox_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def infer_side(point: tuple[float, float], box: tuple[float, float, float, float]) -> str:
    """Infer the nearest component side in OpenCV coordinates."""
    x, y = point
    distances = {
        "left": abs(x - box[0]),
        "right": abs(x - box[2]),
        "top": abs(y - box[1]),
        "bottom": abs(y - box[3]),
    }
    return min(distances, key=distances.get)


def _geometry_component_map(
    predicted: dict[str, Any],
    target: dict[str, Any],
    known_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """One-to-one diagnostic mapping for currently unmatched components.

    This mapping is deliberately geometry/type based and is used only to
    classify ownership evidence.  It is not a deployable component resolver.
    """
    pred_keys = [key for key in predicted if key not in known_map]
    used_gt = set(known_map.values())
    gt_keys = [key for key in target if key not in used_gt]
    if not pred_keys or not gt_keys:
        return {}, {}
    costs = [[INVALID_COST for _ in gt_keys] for _ in pred_keys]
    evidence: dict[tuple[int, int], dict[str, float]] = {}
    for i, pred_key in enumerate(pred_keys):
        pred = predicted[pred_key]
        pred_box = tuple(map(float, pred.get("bbox") or ()))
        if len(pred_box) != 4:
            continue
        for j, gt_key in enumerate(gt_keys):
            gt = target[gt_key]
            if pred.get("type") != gt.get("type"):
                continue
            gt_box = tuple(map(float, gt.get("bbox") or ()))
            if len(gt_box) != 4:
                continue
            iou = _bbox_iou(pred_box, gt_box)
            center_distance = math.dist(_center(pred_box), _center(gt_box))
            box_error = _bbox_error(pred_box, gt_box)
            diagonal = max(1.0, math.hypot(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]))
            size_ratio = ((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])) / max(
                1.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            )
            admissible = (
                box_error <= 40.0
                or iou >= 0.10
                or (center_distance <= max(20.0, 0.50 * diagonal) and 0.20 <= size_ratio <= 5.0)
            )
            if not admissible:
                continue
            cost = box_error + 0.25 * center_distance + 20.0 * (1.0 - iou)
            costs[i][j] = cost
            evidence[(i, j)] = {
                "bbox_iou": iou,
                "bbox_error": box_error,
                "center_distance": center_distance,
                "cost": cost,
            }
    rows, columns = linear_sum_assignment(costs)
    mapping: dict[str, str] = {}
    details: dict[str, dict[str, float]] = {}
    for i, j in zip(rows, columns):
        i, j = int(i), int(j)
        if costs[i][j] >= INVALID_COST:
            continue
        mapping[pred_keys[i]] = gt_keys[j]
        details[pred_keys[i]] = evidence[(i, j)]
    return mapping, details


def match_pins_one_to_one(
    gt_pins: list[dict[str, Any]],
    events: list[dict[str, Any]],
    radius: float,
) -> dict[str, tuple[int, float]]:
    """Hungarian matching with dummy columns and hard side/radius gates."""
    if not gt_pins or not events:
        return {}
    columns = len(events) + len(gt_pins)
    costs = [[INVALID_COST] * columns for _ in gt_pins]
    for i, pin in enumerate(gt_pins):
        for j, event in enumerate(events):
            distance = math.dist(pin["point"], event["tip"])
            if distance <= radius and pin["side"] == event.get("side"):
                costs[i][j] = distance
        costs[i][len(events) + i] = radius + 1.0
    rows, assigned_columns = linear_sum_assignment(costs)
    matches: dict[str, tuple[int, float]] = {}
    for i, j in zip(rows, assigned_columns):
        i, j = int(i), int(j)
        if j >= len(events) or costs[i][j] > radius:
            continue
        matches[gt_pins[i]["id"]] = (j, float(costs[i][j]))
    return matches


def _events(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, raw in enumerate(diagnostics.get("pin_events") or []):
        if not isinstance(raw, dict) or not isinstance(raw.get("tip"), (list, tuple)):
            continue
        row = dict(raw)
        row["event_index"] = index
        row["tip"] = tuple(map(float, row["tip"]))
        row["component"] = str(row.get("component") or "")
        row["side"] = str(row.get("side") or "")
        result.append(row)
    return result


def _gt_pins(target: dict[str, Any], image_height: int) -> list[dict[str, Any]]:
    result = []
    components = target.get("components") or {}
    for component_key, pins in (target.get("pins") or {}).items():
        component = components.get(component_key)
        if component is None:
            continue
        box = tuple(target_bbox_to_opencv(component["bbox"], image_height))
        for pin_key, pin in pins.items():
            point = tuple(target_to_opencv(pin["point"]["x"], pin["point"]["y"], image_height))
            result.append({
                "id": f"{component_key}/{pin_key}",
                "component": component_key,
                "component_type": component.get("type", ""),
                "pin_key": pin_key,
                "point": point,
                "side": infer_side(point, box),
            })
    return result


def _aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = []
    for value, items in sorted(groups.items()):
        counts = Counter(item["ownership_class"] for item in items)
        output.append({
            key: value,
            "missed_pin_count": len(items),
            **{name: counts[name] for name in OWNERSHIP_CLASSES},
            "constrained_coverage_5": sum(item["matched_5"] for item in items) / len(items),
            "constrained_coverage_10": sum(item["matched_10"] for item in items) / len(items),
            "constrained_coverage_20": sum(item["matched_20"] for item in items) / len(items),
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    attribution = _load(args.attribution_json)
    if attribution["metadata"].get("case_start") != 1 or attribution["metadata"].get("case_end") != 150:
        raise ValueError("Attribution input must cover exactly cases 0001..0150")
    prediction_dir = args.prediction_dir or Path(attribution["metadata"]["prediction_dir"])
    records_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in attribution["records"].values():
        if record.get("primary_bucket") in {"A0_COMPONENT_MISS", "A1_TERMINAL_MISS"}:
            records_by_case[record["case_id"]].append(record)

    output_records = []
    component_mapping_summary = Counter()
    for number in range(1, 151):
        assert_allowed_case_id(number)
        case = f"{number:04d}"
        folder = args.target_root / case
        image_path = next(folder.glob("*.png"))
        target_path = next(folder.glob("*_target*.json"))
        image = read_image(image_path)
        target = _load(target_path)
        predicted = _load(prediction_dir / case / "result.json")
        diagnostics = _load(prediction_dir / case / "diagnostics.json")
        events = _events(diagnostics)
        gt_pins = _gt_pins(target, image.shape[0])

        known_map = component_key_map(predicted.get("components") or {}, target.get("components") or {})
        geometry_map, geometry_details = _geometry_component_map(
            predicted.get("components") or {}, target.get("components") or {}, known_map
        )
        owner_to_gt = {**known_map, **geometry_map}
        for owner in geometry_map:
            component_mapping_summary[
                "unresolved" if owner.startswith("UNRESOLVED_") else "wrong_known"
            ] += 1

        pins_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        events_by_gt_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pin in gt_pins:
            pins_by_component[pin["component"]].append(pin)
        for event in events:
            gt_component = owner_to_gt.get(event["component"])
            if gt_component:
                events_by_gt_component[gt_component].append(event)

        matches_by_radius: dict[float, dict[str, tuple[dict[str, Any], float]]] = {}
        for radius in RADII:
            all_matches: dict[str, tuple[dict[str, Any], float]] = {}
            for component_key, component_pins in pins_by_component.items():
                component_events = events_by_gt_component.get(component_key, [])
                matched = match_pins_one_to_one(component_pins, component_events, radius)
                for pin_id, (event_index, distance) in matched.items():
                    all_matches[pin_id] = (component_events[event_index], distance)
            matches_by_radius[radius] = all_matches

        case_records = records_by_case.get(case, [])
        for original in case_records:
            pin_id = f"{original['gt_component']}/{original['gt_pin_key']}"
            match_20 = matches_by_radius[20.0].get(pin_id)
            global_candidates = [
                (math.dist(tuple(original["gt_point_opencv"]), event["tip"]), event)
                for event in events
            ]
            nearest_distance, nearest_event = min(global_candidates, default=(None, None), key=lambda item: item[0] if item[0] is not None else math.inf)
            if match_20:
                event, distance = match_20
                owner = event["component"]
                if owner in known_map and known_map[owner] == original["gt_component"]:
                    ownership_class = "CORRECT_OWNER_TERMINAL"
                elif owner.startswith("UNRESOLVED_"):
                    ownership_class = "UNRESOLVED_OWNER_TERMINAL"
                else:
                    ownership_class = "WRONG_KNOWN_OWNER_TERMINAL"
                geometry = geometry_details.get(owner)
            elif nearest_distance is not None and nearest_distance <= 20.0:
                event, distance, owner, geometry = nearest_event, nearest_distance, nearest_event["component"], None
                ownership_class = "NEARBY_BUT_GEOMETRY_INVALID"
            else:
                event, distance, owner, geometry = nearest_event, nearest_distance, (nearest_event or {}).get("component"), None
                ownership_class = "TRUE_TERMINAL_MISS"
            output_records.append({
                "gt_pin_id": original["gt_pin_id"],
                "case_id": case,
                "source": original.get("source") or source_from_name(image_path.name),
                "gt_component": original["gt_component"],
                "gt_component_type": original["gt_component_type"],
                "gt_pin_key": original["gt_pin_key"],
                "pred_component": original.get("pred_component"),
                "component_strict_tp": bool(original.get("component_strict_tp")),
                "original_bucket": original["primary_bucket"],
                "ownership_class": ownership_class,
                "matched_5": pin_id in matches_by_radius[5.0],
                "matched_10": pin_id in matches_by_radius[10.0],
                "matched_20": match_20 is not None,
                "distance": distance,
                "event_index": event.get("event_index") if event else None,
                "event_owner": owner,
                "event_side": event.get("side") if event else None,
                "event_method": event.get("method") if event else None,
                "gt_side": infer_side(tuple(original["gt_point_opencv"]), tuple(target_bbox_to_opencv(
                    target["components"][original["gt_component"]]["bbox"], image.shape[0]
                ))),
                "geometry_component_match": geometry,
            })

    counts = Counter(row["ownership_class"] for row in output_records)
    by_original_bucket = {}
    for bucket in ("A0_COMPONENT_MISS", "A1_TERMINAL_MISS"):
        subset = [row for row in output_records if row["original_bucket"] == bucket]
        bucket_counts = Counter(row["ownership_class"] for row in subset)
        by_original_bucket[bucket] = {
            "count": len(subset),
            "ownership_counts": {name: bucket_counts[name] for name in OWNERSHIP_CLASSES},
            "coverage_5": sum(row["matched_5"] for row in subset) / max(1, len(subset)),
            "coverage_10": sum(row["matched_10"] for row in subset) / max(1, len(subset)),
            "coverage_20": sum(row["matched_20"] for row in subset) / max(1, len(subset)),
        }
    payload = {
        "metadata": {
            "OFFICIAL_SCORE": False,
            "ORACLE": False,
            "case_start": 1,
            "case_end": 150,
            "sealed_holdout_used": False,
            "prediction_dir": str(prediction_dir.resolve()),
            "target_root": str(args.target_root.resolve()),
            "primary_radius_px": 20.0,
            "matching": "per-component one-to-one Hungarian with side/radius gates",
            "geometry_component_mapping": "diagnostic type+bbox Hungarian for unmatched components",
        },
        "summary": {
            "analyzed_missed_pins": len(output_records),
            "ownership_counts": {name: counts[name] for name in OWNERSHIP_CLASSES},
            "coverage_5": sum(row["matched_5"] for row in output_records) / len(output_records),
            "coverage_10": sum(row["matched_10"] for row in output_records) / len(output_records),
            "coverage_20": sum(row["matched_20"] for row in output_records) / len(output_records),
            "component_geometry_mappings": dict(component_mapping_summary),
            "by_original_bucket": by_original_bucket,
        },
        "by_component_type": _aggregate(output_records, "gt_component_type"),
        "by_case": _aggregate(output_records, "case_id"),
        "records": {row["gt_pin_id"]: row for row in output_records},
    }
    return payload


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    total = summary["analyzed_missed_pins"]
    lines = [
        "# Pin Terminal Ownership 约束化审计",
        "",
        "> 本报告为本地非官方诊断。仅使用0001～0150；未读取sealed holdout。",
        "",
        "## 总体结果",
        "",
        "| 分类 | 数量 | 比例 |",
        "|---|---:|---:|",
    ]
    for name in OWNERSHIP_CLASSES:
        count = summary["ownership_counts"][name]
        lines.append(f"| {name} | {count} | {count / max(1, total):.2%} |")
    lines.extend([
        "",
        "## 一对一约束覆盖率",
        "",
        f"- 5px：{summary['coverage_5']:.2%}",
        f"- 10px：{summary['coverage_10']:.2%}",
        f"- 20px：{summary['coverage_20']:.2%}",
        "",
        "这些覆盖率要求terminal与GT Component具有可接受的type/bbox对应、side一致，并经过一对一Hungarian分配。它们不能与旧的全图最近邻覆盖率直接等同。",
        "",
        "## 原A0/A1分解",
        "",
    ])
    for bucket, block in summary["by_original_bucket"].items():
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append(f"- 数量：{block['count']}")
        lines.append(f"- Coverage@5/10/20：{block['coverage_5']:.2%} / {block['coverage_10']:.2%} / {block['coverage_20']:.2%}")
        for name in OWNERSHIP_CLASSES:
            lines.append(f"- {name}：{block['ownership_counts'][name]}")
        lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- `UNRESOLVED_OWNER_TERMINAL`和`WRONG_KNOWN_OWNER_TERMINAL`依赖离线GT几何映射，只用于误差预算。",
        "- `NEARBY_BUT_GEOMETRY_INVALID`表示20px内虽有terminal，但其owner/type/bbox/side证据不足。",
        "- `TRUE_TERMINAL_MISS`表示在当前约束下没有20px内可唯一分配的terminal；它比旧的160px全局最近邻更接近Localization漏检，但仍是诊断结论。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--attribution-json", type=Path, default=ROOT / "reports" / "pin_recall_attribution.json")
    parser.add_argument("--json-output", type=Path, default=ROOT / "reports" / "pin_terminal_ownership_audit.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "reports" / "pin_terminal_ownership_audit.md")
    parser.add_argument("--per-case-output", type=Path, default=ROOT / "reports" / "pin_terminal_ownership_per_case.csv")
    parser.add_argument("--per-type-output", type=Path, default=ROOT / "reports" / "pin_terminal_ownership_per_type.csv")
    args = parser.parse_args(argv)
    if args.target_root.name != "200_train_cases":
        raise ValueError("target-root must be exactly 200_train_cases")
    payload = analyze(args)
    write_json(args.json_output, payload)
    _write_markdown(args.markdown_output, payload)
    _write_csv(args.per_case_output, payload["by_case"])
    _write_csv(args.per_type_output, payload["by_component_type"])
    print(json.dumps({
        "json": str(args.json_output.resolve()),
        "markdown": str(args.markdown_output.resolve()),
        "summary": payload["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
