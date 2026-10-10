"""P0 audit for the unchanged Pin Localization V3 stage.

This is an evaluation-only entry point.  It runs Text -> Component V4 ->
Pin Localization V3, stops before Pin Semantics, and then reads the official
target for coordinate-only terminal matching.  Cases outside 0001..0150 are
rejected by the shared data policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_v2 import bbox_error, component_key_map, linear_sum_assignment, source_from_name
from pcb.component.v4 import ComponentStageV4
from pcb.component_detector_yolo import YoloComponentDetector
from pcb.coordinates import opencv_bbox_to_target
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.data_policy import assert_allowed_case_id, discover_allowed_cases
from pcb.io import read_image, write_json
from pcb.official_target import read_official_target, target_to_opencv
from pcb.official_types import to_official_type
from pcb.ocr_backends import HybridOCR, TiledEasyOCR
from pcb.pin.localization.v3 import PinLocalizationStageV3
from pcb.pin_detection import TWO_TERMINAL
from pcb.text.stage import CurrentTextStage
from pcb.vision import OCR


OFFICIAL_SCORE = False
THRESHOLDS = (5.0, 10.0, 20.0)


def branch_for_type(component_type: str) -> str:
    """Return the exact branch selected by terminal_candidates_v3()."""
    component_type = to_official_type(component_type)
    if component_type in TWO_TERMINAL:
        return "two_terminal"
    if component_type == "crystal_4pin":
        return "crystal_4pin"
    if component_type == "gnd":
        return "gnd"
    return "boundary_wire"


def parse_case_ids(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        assert_allowed_case_id(value)
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--case-ids did not contain a case number")
    return values


def terminal_coordinates(terminals: Iterable[dict[str, Any]]) -> list[tuple[float, float]]:
    return [(float(item["tip"][0]), float(item["tip"][1])) for item in terminals]


def coordinate_assignment(
    terminals: list[dict[str, Any]],
    gt_points: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Minimum-distance one-to-one assignment using coordinates only."""
    if not terminals or not gt_points:
        return []
    pred_points = terminal_coordinates(terminals)
    distance = np.empty((len(pred_points), len(gt_points)), dtype=float)
    for pred_index, pred_point in enumerate(pred_points):
        for gt_index, gt_point in enumerate(gt_points):
            distance[pred_index, gt_index] = math.dist(pred_point, gt_point)
    rows, columns = linear_sum_assignment(distance)
    matches: list[dict[str, Any]] = []
    for pred_index, gt_index in zip(rows.tolist(), columns.tolist()):
        pred_x, pred_y = pred_points[pred_index]
        gt_x, gt_y = gt_points[gt_index]
        matches.append({
            "pred_index": int(pred_index),
            "gt_index": int(gt_index),
            "distance": float(distance[pred_index, gt_index]),
            "dx": pred_x - gt_x,
            "dy": pred_y - gt_y,
            "side": str(terminals[pred_index].get("side") or "unknown"),
        })
    return matches


def predicted_component_payload(component, image_height: int) -> dict[str, Any]:
    """Mirror the Component fields used by the existing submission exporter."""
    display = component.name or component.observable_designator or component.net_label or component.key
    return {
        "Name": display,
        "type": to_official_type(component.type),
        "value": component.value,
        "bbox": list(opencv_bbox_to_target(component.bbox, image_height)),
    }


def strict_component_pairs(
    predicted: dict[str, dict[str, Any]],
    target: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    """Reuse evaluate_v2's mapping and its Component true-positive filters."""
    mapping = component_key_map(predicted, target)
    pairs: list[tuple[str, str]] = []
    for predicted_key, target_key in mapping.items():
        if target_key not in target:
            continue
        pred_item = predicted[predicted_key]
        gt_item = target[target_key]
        if pred_item.get("type") != gt_item.get("type"):
            continue
        if bbox_error(pred_item.get("bbox", ()), gt_item.get("bbox", ())) > 20:
            continue
        if pred_item.get("type") == "box" and pred_item.get("Name") != gt_item.get("Name"):
            continue
        if pred_item.get("type") in {"r", "c"} and pred_item.get("value") != gt_item.get("value"):
            continue
        pairs.append((predicted_key, target_key))
    return pairs


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator:
        return float(numerator) / float(denominator)
    return 1.0 if float(numerator) == 0 else 0.0


def _prf(true_positive: int, predicted: int, target: int) -> tuple[float, float, float]:
    precision = _safe_ratio(true_positive, predicted)
    recall = _safe_ratio(true_positive, target)
    if precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 1.0 if predicted == 0 and target == 0 else 0.0
    return precision, recall, f1


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


@dataclass
class AuditAccumulator:
    predicted_components: int = 0
    target_components: int = 0
    matched_components: int = 0
    predicted_terminals: int = 0
    target_terminals: int = 0
    matched: dict[float, int] = field(default_factory=lambda: {value: 0 for value in THRESHOLDS})
    count_errors: list[int] = field(default_factory=list)
    exact_count: int = 0
    over_count: int = 0
    under_count: int = 0
    offsets: list[tuple[float, float, str]] = field(default_factory=list)

    def add_inventory(
        self,
        predicted_terminals: int,
        target_terminals: int,
        *,
        predicted_components: int = 0,
        target_components: int = 0,
    ) -> None:
        self.predicted_terminals += int(predicted_terminals)
        self.target_terminals += int(target_terminals)
        self.predicted_components += int(predicted_components)
        self.target_components += int(target_components)

    def add_component_pair(
        self,
        predicted_count: int,
        target_count: int,
        assignments: list[dict[str, Any]],
        *,
        include_inventory: bool = False,
    ) -> None:
        self.matched_components += 1
        difference = int(predicted_count) - int(target_count)
        self.count_errors.append(abs(difference))
        self.exact_count += int(difference == 0)
        self.over_count += int(difference > 0)
        self.under_count += int(difference < 0)
        if include_inventory:
            self.add_inventory(predicted_count, target_count, predicted_components=1, target_components=1)
        for threshold in THRESHOLDS:
            self.matched[threshold] += sum(item["distance"] <= threshold for item in assignments)
        for item in assignments:
            if item["distance"] <= 20.0:
                self.offsets.append((float(item["dx"]), float(item["dy"]), str(item["side"])))

    def merge(self, other: "AuditAccumulator") -> None:
        self.predicted_components += other.predicted_components
        self.target_components += other.target_components
        self.matched_components += other.matched_components
        self.predicted_terminals += other.predicted_terminals
        self.target_terminals += other.target_terminals
        for threshold in THRESHOLDS:
            self.matched[threshold] += other.matched[threshold]
        self.count_errors.extend(other.count_errors)
        self.exact_count += other.exact_count
        self.over_count += other.over_count
        self.under_count += other.under_count
        self.offsets.extend(other.offsets)

    def metric_dict(self) -> dict[str, Any]:
        precision_5, recall_5, f1_5 = _prf(
            self.matched[5.0], self.predicted_terminals, self.target_terminals
        )
        precision_10, recall_10, f1_10 = _prf(
            self.matched[10.0], self.predicted_terminals, self.target_terminals
        )
        precision_20, recall_20, f1_20 = _prf(
            self.matched[20.0], self.predicted_terminals, self.target_terminals
        )
        count_samples = len(self.count_errors)
        return {
            "terminal_precision_5": precision_5,
            "terminal_recall_5": recall_5,
            "terminal_f1_5": f1_5,
            "terminal_precision_10": precision_10,
            "terminal_recall_10": recall_10,
            "terminal_f1_10": f1_10,
            "terminal_precision_20": precision_20,
            "terminal_recall_20": recall_20,
            "terminal_f1_20": f1_20,
            "pin_count_mae": _mean([float(value) for value in self.count_errors]),
            "exact_pin_count_accuracy": (
                _safe_ratio(self.exact_count, count_samples) if count_samples else None
            ),
            "over_count_components": self.over_count,
            "under_count_components": self.under_count,
            "count_evaluated_components": count_samples,
            "predicted_component_count": self.predicted_components,
            "target_component_count": self.target_components,
            "matched_component_count": self.matched_components,
            "predicted_terminal_count": self.predicted_terminals,
            "target_terminal_count": self.target_terminals,
            "matched_terminal_5": self.matched[5.0],
            "matched_terminal_10": self.matched[10.0],
            "matched_terminal_20": self.matched[20.0],
        }

    def offset_dict(self) -> dict[str, Any]:
        dx = [item[0] for item in self.offsets]
        dy = [item[1] for item in self.offsets]

        def summarize(rows: list[tuple[float, float, str]]) -> dict[str, Any]:
            x_values = [row[0] for row in rows]
            y_values = [row[1] for row in rows]
            return {
                "count": len(rows),
                "mean_dx": _mean(x_values),
                "mean_dy": _mean(y_values),
                "mean_abs_dx": _mean([abs(value) for value in x_values]),
                "mean_abs_dy": _mean([abs(value) for value in y_values]),
                "median_dx": _median(x_values),
                "median_dy": _median(y_values),
                "median_abs_dx": _median([abs(value) for value in x_values]),
                "median_abs_dy": _median([abs(value) for value in y_values]),
            }

        result = summarize(self.offsets)
        result["by_predicted_side"] = {
            side: summarize([row for row in self.offsets if row[2] == side])
            for side in ("left", "right", "top", "bottom", "unknown")
            if any(row[2] == side for row in self.offsets)
        }
        return result


def _target_points(target_scene, component_key: str, image_height: int) -> list[tuple[float, float]]:
    return [
        target_to_opencv(pin.point.x, pin.point.y, image_height)
        for pin in target_scene.pins.get(component_key, {}).values()
    ]


def _dominant_error(errors: Counter[str]) -> str:
    if not errors:
        return "NONE"
    return max(errors.items(), key=lambda item: (item[1], item[0]))[0]


def audit_case(case_files, text_stage, component_stage, localization_stage, config, ocr, detector):
    started = time.perf_counter()
    image = read_image(case_files.image_path)
    height, width = image.shape[:2]
    context = PipelineContext(
        case_files.image_path.name,
        width,
        height,
        config,
        ocr,
        detector,
    )

    text_output = text_stage.run(image, context)
    component_output = component_stage.run(image, text_output, context)
    localization_output = localization_stage.run(image, component_output, context)

    # Match exporter overwrite behavior if duplicate keys occur.
    predicted_components: dict[str, Any] = {}
    predicted_payload: dict[str, dict[str, Any]] = {}
    terminals_by_component: dict[str, list[dict[str, Any]]] = {}
    for component, terminals in localization_output.terminals:
        predicted_components[component.key] = component
        predicted_payload[component.key] = predicted_component_payload(component, height)
        terminals_by_component[component.key] = terminals

    # Targets enter only here, after the inference stages have finished.
    target_scene = read_official_target(case_files.target_path)
    target_payload = target_scene.raw.get("components") or {}
    matched_pairs = strict_component_pairs(predicted_payload, target_payload)
    matched_predicted_keys = {pair[0] for pair in matched_pairs}
    matched_target_keys = {pair[1] for pair in matched_pairs}

    overall = AuditAccumulator()
    conditional = AuditAccumulator()
    by_branch: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    by_type: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    by_branch_conditional: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    by_type_conditional: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)

    for key, component in predicted_components.items():
        terminal_count = len(terminals_by_component.get(key, ()))
        component_type = to_official_type(component.type)
        branch = branch_for_type(component_type)
        overall.add_inventory(terminal_count, 0, predicted_components=1)
        by_type[component_type].add_inventory(terminal_count, 0, predicted_components=1)
        by_branch[branch].add_inventory(terminal_count, 0, predicted_components=1)

    for key, target_component in target_scene.components.items():
        terminal_count = len(target_scene.pins.get(key, {}))
        component_type = to_official_type(target_component.type)
        branch = branch_for_type(component_type)
        overall.add_inventory(0, terminal_count, target_components=1)
        by_type[component_type].add_inventory(0, terminal_count, target_components=1)
        by_branch[branch].add_inventory(0, terminal_count, target_components=1)

    errors: Counter[str] = Counter()
    errors["COMPONENT_MISS"] = len(target_scene.components) - len(matched_target_keys)
    for predicted_key, target_key in matched_pairs:
        component = predicted_components[predicted_key]
        predicted_terminals = terminals_by_component.get(predicted_key, [])
        gt_points = _target_points(target_scene, target_key, height)
        assignments = coordinate_assignment(predicted_terminals, gt_points)
        component_type = to_official_type(component.type)
        branch = branch_for_type(component_type)

        overall.add_component_pair(len(predicted_terminals), len(gt_points), assignments)
        conditional.add_component_pair(
            len(predicted_terminals), len(gt_points), assignments, include_inventory=True
        )
        by_type[component_type].add_component_pair(
            len(predicted_terminals), len(gt_points), assignments
        )
        by_branch[branch].add_component_pair(
            len(predicted_terminals), len(gt_points), assignments
        )
        by_type_conditional[component_type].add_component_pair(
            len(predicted_terminals), len(gt_points), assignments, include_inventory=True
        )
        by_branch_conditional[branch].add_component_pair(
            len(predicted_terminals), len(gt_points), assignments, include_inventory=True
        )

        difference = len(predicted_terminals) - len(gt_points)
        if difference > 0:
            errors["OVERCOUNT"] += 1
        elif difference < 0:
            errors["UNDERCOUNT"] += 1
        near_5 = sum(item["distance"] <= 5.0 for item in assignments)
        near_20 = sum(item["distance"] <= 20.0 for item in assignments)
        errors["OFFSET"] += max(0, near_20 - near_5)
        errors["MISS"] += max(0, len(gt_points) - near_20)
        errors["FALSE_POSITIVE"] += max(0, len(predicted_terminals) - near_20)

    source = source_from_name(case_files.image_path.name)
    metrics = overall.metric_dict()
    conditional_metrics = conditional.metric_dict()
    offsets = conditional.offset_dict()
    case_row = {
        "case_id": f"{case_files.case_id:04d}",
        "source": source,
        "gt_component_count": len(target_scene.components),
        "pred_component_count": len(predicted_components),
        "matched_component_count": len(matched_pairs),
        "gt_terminal_count": overall.target_terminals,
        "pred_terminal_count": overall.predicted_terminals,
        "matched_terminal_5": overall.matched[5.0],
        "matched_terminal_10": overall.matched[10.0],
        "matched_terminal_20": overall.matched[20.0],
        "precision_5": metrics["terminal_precision_5"],
        "recall_5": metrics["terminal_recall_5"],
        "recall_10": metrics["terminal_recall_10"],
        "recall_20": metrics["terminal_recall_20"],
        "conditional_recall_5": conditional_metrics["terminal_recall_5"],
        "pin_count_mae": conditional_metrics["pin_count_mae"],
        "exact_pin_count_accuracy": conditional_metrics["exact_pin_count_accuracy"],
        "mean_abs_dx_20": offsets["mean_abs_dx"],
        "mean_abs_dy_20": offsets["mean_abs_dy"],
        "dominant_error_type": _dominant_error(errors),
        "runtime_seconds": time.perf_counter() - started,
    }
    return {
        "case_row": case_row,
        "overall": overall,
        "conditional": conditional,
        "by_branch": by_branch,
        "by_type": by_type,
        "by_branch_conditional": by_branch_conditional,
        "by_type_conditional": by_type_conditional,
        "errors": errors,
        "matched_pairs": len(matched_pairs),
        "text_count": len(text_output.texts),
    }


def _merge_map(destination: defaultdict[str, AuditAccumulator], source) -> None:
    for key, accumulator in source.items():
        destination[key].merge(accumulator)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    return "" if value is None else value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def per_type_rows(
    by_type: dict[str, AuditAccumulator],
    by_type_conditional: dict[str, AuditAccumulator],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component_type, accumulator in sorted(by_type.items()):
        metrics = accumulator.metric_dict()
        conditional = by_type_conditional.get(component_type, AuditAccumulator())
        conditional_metrics = conditional.metric_dict()
        offsets = conditional.offset_dict()
        rows.append({
            "component_type": component_type,
            "component_count": accumulator.target_components,
            "matched_component_count": accumulator.matched_components,
            "gt_terminal_count": accumulator.target_terminals,
            "pred_terminal_count": accumulator.predicted_terminals,
            "precision_5": metrics["terminal_precision_5"],
            "recall_5": metrics["terminal_recall_5"],
            "f1_5": metrics["terminal_f1_5"],
            "recall_10": metrics["terminal_recall_10"],
            "recall_20": metrics["terminal_recall_20"],
            "conditional_precision_5": conditional_metrics["terminal_precision_5"],
            "conditional_recall_5": conditional_metrics["terminal_recall_5"],
            "conditional_f1_5": conditional_metrics["terminal_f1_5"],
            "conditional_recall_10": conditional_metrics["terminal_recall_10"],
            "conditional_recall_20": conditional_metrics["terminal_recall_20"],
            "pin_count_mae": conditional_metrics["pin_count_mae"],
            "exact_count_accuracy": conditional_metrics["exact_pin_count_accuracy"],
            "mean_abs_dx_20": offsets["mean_abs_dx"],
            "mean_abs_dy_20": offsets["mean_abs_dy"],
        })
    return rows


def _round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return value


def _worst_group(groups: dict[str, AuditAccumulator], minimum_target_terminals: int) -> tuple[str, dict[str, Any]] | None:
    candidates = [
        (key, value.metric_dict())
        for key, value in groups.items()
        if value.target_terminals >= minimum_target_terminals
    ]
    if not candidates:
        candidates = [(key, value.metric_dict()) for key, value in groups.items() if value.target_terminals]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row[1]["terminal_recall_5"], -row[1]["target_terminal_count"]))


def diagnosis(overall: dict[str, Any], conditional: dict[str, Any]) -> tuple[str, str]:
    recall_5 = conditional["terminal_recall_5"]
    recall_20 = conditional["terminal_recall_20"]
    near_gain = recall_20 - recall_5
    if recall_20 < 0.50 and near_gain >= 0.15:
        return "两者都有", "20px内命中明显增加，但20px Recall仍低于0.50。"
    if recall_20 < 0.50:
        return "candidate漏检", "扩大到20px后Recall仍低，主要损失无法用小范围坐标偏移解释。"
    if near_gain >= 0.20:
        return "坐标偏移", "5px到20px的Recall增幅明显，候选通常存在于GT附近。"
    if recall_5 >= 0.70:
        return "暂无法判断", "Localization本身已较高，最终Pin误差可能更多来自Semantics。"
    return "两者都有", "5px严格命中不足，且20px提升未形成单一主导原因。"


def render_summary(
    metadata: dict[str, Any],
    overall: dict[str, Any],
    conditional: dict[str, Any],
    offsets: dict[str, Any],
    by_branch: dict[str, AuditAccumulator],
    by_type: dict[str, AuditAccumulator],
    by_branch_conditional: dict[str, AuditAccumulator],
    by_type_conditional: dict[str, AuditAccumulator],
    errors: Counter[str],
) -> str:
    minimum_type = max(10, int(round(conditional["target_terminal_count"] * 0.01)))
    worst_branch = _worst_group(by_branch_conditional, 1)
    worst_type = _worst_group(by_type_conditional, minimum_type)
    problem, evidence = diagnosis(overall, conditional)
    offset_text = (
        "没有20px内可配对样本"
        if offsets["count"] == 0
        else f"mean dx={offsets['mean_dx']:.3f}px，mean dy={offsets['mean_dy']:.3f}px，"
             f"mean |dx|={offsets['mean_abs_dx']:.3f}px，mean |dy|={offsets['mean_abs_dy']:.3f}px"
    )
    matched_fraction = _safe_ratio(
        conditional["matched_component_count"], overall["target_component_count"]
    )
    eligible_terminal_fraction = _safe_ratio(
        conditional["target_terminal_count"], overall["target_terminal_count"]
    )
    boundary_metrics = by_branch_conditional.get("boundary_wire", AuditAccumulator()).metric_dict()
    boundary_offsets = by_branch_conditional.get("boundary_wire", AuditAccumulator()).offset_dict()
    boundary_side = boundary_offsets.get("by_predicted_side", {})
    boundary_direction = ", ".join(
        f"{side}({row['mean_dx']:.2f},{row['mean_dy']:.2f})"
        for side, row in boundary_side.items()
        if row.get("count")
    ) or "N/A"
    return "\n".join([
        "# Pin Localization P0专项审计",
        "",
        f"- 配置：Component V4 + Pin Localization V3；未运行Pin Semantics、Wire或Topology。",
        f"- 测试范围：{metadata['case_range']}，成功 {metadata['success_count']}/{metadata['requested_case_count']}。",
        f"- OFFICIAL_SCORE：FALSE。sealed holdout used：false。",
        f"- Terminal Precision@5：{overall['terminal_precision_5']:.6f}",
        f"- Terminal Recall@5：{overall['terminal_recall_5']:.6f}",
        f"- Terminal F1@5：{overall['terminal_f1_5']:.6f}",
        f"- Recall@10 / @20：{overall['terminal_recall_10']:.6f} / {overall['terminal_recall_20']:.6f}",
        f"- Conditional Recall@5 / @10 / @20：{conditional['terminal_recall_5']:.6f} / {conditional['terminal_recall_10']:.6f} / {conditional['terminal_recall_20']:.6f}",
        f"- Component严格匹配率：{matched_fraction:.6f}；匹配Component覆盖GT terminal比例：{eligible_terminal_fraction:.6f}",
        f"- Pin Count MAE：{conditional['pin_count_mae'] if conditional['pin_count_mae'] is not None else 'N/A'}",
        f"- Exact Pin Count Accuracy：{conditional['exact_pin_count_accuracy']:.6f}",
        f"- 20px offset：{offset_text}",
        f"- 最差算法分支：{worst_branch[0] if worst_branch else 'N/A'}",
        f"- boundary_wire条件Recall@5 / @20：{boundary_metrics['terminal_recall_5']:.6f} / {boundary_metrics['terminal_recall_20']:.6f}；Pred/GT terminal={boundary_metrics['predicted_terminal_count']}/{boundary_metrics['target_terminal_count']}",
        f"- boundary_wire按预测side的mean(dx,dy)：{boundary_direction}",
        f"- 最差主要器件类型：{worst_type[0] if worst_type else 'N/A'}（主要类型阈值：至少{minimum_type}个GT terminal）",
        f"- 主要问题判断：{problem}。{evidence}",
        f"- 错误计数：{dict(errors)}",
        "",
        "20px结果仅用于偏移诊断；5px仍是本报告的严格Localization口径。",
    ]) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-ids", help="Comma-separated cases within 0001..0150; omit for all 150")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "current.json")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "component_yolo11n_continue_v2_best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ocr-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ocr-model-dir", type=Path, default=PROJECT_ROOT / "models" / "easyocr")
    parser.add_argument("--ocr-cache-dir", type=Path, default=PROJECT_ROOT / "runs" / "ocr_cache_v4_1")
    parser.add_argument("--yolo-cache-dir", type=Path, default=PROJECT_ROOT / "runs" / "yolo_cache")
    parser.add_argument("--allow-ocr-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--output-prefix", default="pin_p0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if dataset_root.name != "200_train_cases":
        raise SystemExit("dataset-root must be exactly the 200_train_cases directory")
    selected_ids = parse_case_ids(args.case_ids)
    all_cases = discover_allowed_cases(dataset_root)
    selected = all_cases if selected_ids is None else [row for row in all_cases if row.case_id in selected_ids]
    if selected_ids is not None and [row.case_id for row in selected] != selected_ids:
        lookup = {row.case_id: row for row in selected}
        selected = [lookup[value] for value in selected_ids]

    config = PipelineConfig.load(args.config)
    if config.component != "v4" or config.pin_localization != "v3":
        raise SystemExit("P0 requires Component V4 and Pin Localization V3")
    tiled = TiledEasyOCR(
        args.ocr_cache_dir,
        args.ocr_model_dir,
        device=args.ocr_device,
        allow_download=args.allow_ocr_download,
    )
    ocr = HybridOCR(tiled, OCR(args.ocr_cache_dir))
    detector = YoloComponentDetector(args.weights, args.device, cache_dir=args.yolo_cache_dir)
    text_stage = CurrentTextStage()
    component_stage = ComponentStageV4()
    localization_stage = PinLocalizationStageV3()

    overall = AuditAccumulator()
    conditional = AuditAccumulator()
    by_branch: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    by_type: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    by_branch_conditional: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    by_type_conditional: defaultdict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    error_counts: Counter[str] = Counter()
    case_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for index, case_files in enumerate(selected, 1):
        case_id = f"{case_files.case_id:04d}"
        try:
            result = audit_case(
                case_files,
                text_stage,
                component_stage,
                localization_stage,
                config,
                ocr,
                detector,
            )
            case_rows.append(result["case_row"])
            overall.merge(result["overall"])
            conditional.merge(result["conditional"])
            _merge_map(by_branch, result["by_branch"])
            _merge_map(by_type, result["by_type"])
            _merge_map(by_branch_conditional, result["by_branch_conditional"])
            _merge_map(by_type_conditional, result["by_type_conditional"])
            error_counts.update(result["errors"])
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(selected)}",
                        "case_id": case_id,
                        "matched_components": result["matched_pairs"],
                        "terminal_recall_5": result["case_row"]["recall_5"],
                        "status": "ok",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            failures.append({"case_id": case_id, "error": repr(exc)})
            print(json.dumps({"case_id": case_id, "status": "failed", "error": repr(exc)}, ensure_ascii=False), flush=True)

    metrics = overall.metric_dict()
    conditional_metrics = conditional.metric_dict()
    offsets = conditional.offset_dict()
    case_range = ",".join(f"{value:04d}" for value in selected_ids) if selected_ids is not None else "0001-0150"
    metadata = {
        "case_range": case_range,
        "requested_case_count": len(selected),
        "success_count": len(case_rows),
        "failure_count": len(failures),
        "dataset_root": str(dataset_root),
        "pipeline_config": config.as_dict(),
        "OFFICIAL_SCORE": OFFICIAL_SCORE,
        "sealed_holdout_used": False,
        "target_read_boundary": "tools/analyze_pin_localization.py only, after inference stages",
        "terminal_source": "PinLocalizationStageV3 raw terminal output",
        "matching": "coordinate-only one-to-one Hungarian within evaluate_v2-matched components",
        "runtime_seconds": time.perf_counter() - started,
    }
    output = {
        "metadata": metadata,
        "overall": metrics,
        "conditional_on_matched_component": conditional_metrics,
        "offset_near_20px": offsets,
        "by_algorithm_branch": {
            key: {
                "end_to_end": value.metric_dict(),
                "conditional_on_matched_component": by_branch_conditional.get(key, AuditAccumulator()).metric_dict(),
                "offset_near_20px": by_branch_conditional.get(key, AuditAccumulator()).offset_dict(),
            }
            for key, value in sorted(by_branch.items())
        },
        "by_component_type": {
            key: {
                "end_to_end": value.metric_dict(),
                "conditional_on_matched_component": by_type_conditional.get(key, AuditAccumulator()).metric_dict(),
                "offset_near_20px": by_type_conditional.get(key, AuditAccumulator()).offset_dict(),
            }
            for key, value in sorted(by_type.items())
        },
        "counts": {
            "total_gt_components": overall.target_components,
            "predicted_components": overall.predicted_components,
            "matched_components": overall.matched_components,
            "total_gt_terminals": overall.target_terminals,
            "total_pred_terminals": overall.predicted_terminals,
            "matched_terminals_5": overall.matched[5.0],
            "matched_terminals_10": overall.matched[10.0],
            "matched_terminals_20": overall.matched[20.0],
        },
        "error_counts": dict(error_counts),
        "failures": failures,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    json_path = output_dir / f"{prefix}_localization.json"
    type_path = output_dir / f"{prefix}_per_type.csv"
    case_path = output_dir / f"{prefix}_per_case.csv"
    summary_path = output_dir / f"{prefix}_error_summary.md"
    write_json(json_path, _round_nested(output))

    type_rows = per_type_rows(by_type, by_type_conditional)
    write_csv(type_path, type_rows, [
        "component_type", "component_count", "matched_component_count",
        "gt_terminal_count", "pred_terminal_count", "precision_5", "recall_5", "f1_5",
        "recall_10", "recall_20", "conditional_precision_5", "conditional_recall_5",
        "conditional_f1_5", "conditional_recall_10", "conditional_recall_20",
        "pin_count_mae", "exact_count_accuracy",
        "mean_abs_dx_20", "mean_abs_dy_20",
    ])
    write_csv(case_path, case_rows, [
        "case_id", "source", "gt_component_count", "pred_component_count",
        "matched_component_count", "gt_terminal_count", "pred_terminal_count",
        "matched_terminal_5", "matched_terminal_10", "matched_terminal_20",
        "precision_5", "recall_5", "recall_10", "recall_20",
        "conditional_recall_5", "pin_count_mae", "exact_pin_count_accuracy",
        "mean_abs_dx_20", "mean_abs_dy_20", "dominant_error_type", "runtime_seconds",
    ])
    summary_path.write_text(
        render_summary(
            metadata,
            metrics,
            conditional_metrics,
            offsets,
            by_branch,
            by_type,
            by_branch_conditional,
            by_type_conditional,
            error_counts,
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "outputs": [str(json_path), str(type_path), str(case_path), str(summary_path)],
        "success_count": len(case_rows),
        "failure_count": len(failures),
        "overall": metrics,
        "conditional": conditional_metrics,
    }, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
