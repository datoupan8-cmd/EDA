"""Component-specialty evaluator for V4. All results are non-official diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from pcb.assignment import maximum_weight_assignment
from pcb.component_proposal_fusion import bbox_iou
from pcb.io import write_json
from evaluate_v2 import evaluate_case, source_from_name

OFFICIAL_SCORE = False
HOLDOUT_START = 151
INTERNAL_KEY = re.compile(r"^∅\d+$")


def bbox_error(left, right):
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def metric(tp, pred, gt):
    precision = tp / pred if pred else (1.0 if gt == 0 else 0.0)
    recall = tp / gt if gt else (1.0 if pred == 0 else 0.0)
    f1 = 2 * tp / (pred + gt) if pred + gt else 1.0
    return {"tp": tp, "pred": pred, "gt": gt, "precision": precision, "recall": recall, "f1": f1}


def matched_pairs(gt_components, pred_components, require_type=True, threshold=20.0):
    gt_items, pred_items = list(gt_components.items()), list(pred_components.items())
    weights = []
    for _, gt in gt_items:
        row = []
        for _, pred in pred_items:
            valid = bbox_error(gt["bbox"], pred["bbox"]) <= threshold and (not require_type or gt["type"] == pred["type"])
            row.append((1.0 - bbox_error(gt["bbox"], pred["bbox"]) / (threshold + 1.0)) if valid else 0.0)
        weights.append(row)
    return [(i, j) for i, j in maximum_weight_assignment(weights) if weights[i][j] > 0], gt_items, pred_items


def normalized_tokens(diagnostics):
    rows = diagnostics.get("text_roles", [])
    return {re.sub(r"\s+", "", str(row.get("text", ""))).upper() for row in rows}


def observable_keys(components):
    return {key for key, component in components.items() if not INTERNAL_KEY.fullmatch(key) and component.get("type") != "gnd"}


def component_case(prediction, target, diagnostics):
    type_pairs, gt_items, pred_items = matched_pairs(target["components"], prediction["components"], True)
    bbox_pairs, _, _ = matched_pairs(target["components"], prediction["components"], False)
    type_bbox = metric(len(type_pairs), len(pred_items), len(gt_items))
    bbox = metric(len(bbox_pairs), len(pred_items), len(gt_items))
    type_correct = sum(gt_items[i][1]["type"] == pred_items[j][1]["type"] for i, j in bbox_pairs)
    type_accuracy = type_correct / len(bbox_pairs) if bbox_pairs else 0.0

    tokens = normalized_tokens(diagnostics)
    observable = observable_keys(target["components"])
    recognized = {key for key in observable if key.upper() in tokens}
    associated = {
        key for key in recognized
        if key in prediction["components"]
        and bbox_error(target["components"][key]["bbox"], prediction["components"][key]["bbox"]) <= 20
        and target["components"][key]["type"] == prediction["components"][key]["type"]
    }
    key_hits = observable.intersection(prediction["components"])
    name_total = name_hits = value_total = value_hits = 0
    for key in target["components"].keys() & prediction["components"].keys():
        gt, pred = target["components"][key], prediction["components"][key]
        if gt.get("Name") is not None:
            name_total += 1; name_hits += pred.get("Name") == gt.get("Name")
        if gt.get("value") is not None:
            value_total += 1; value_hits += pred.get("value") == gt.get("value")

    pred_boxes = [item[1]["bbox"] for item in pred_items]
    duplicate_pairs = sum(bbox_iou(pred_boxes[i], pred_boxes[j]) >= .5 for i in range(len(pred_boxes)) for j in range(i + 1, len(pred_boxes)))
    provenance = {row.get("component"): row for row in diagnostics.get("component_provenance", [])}
    geometry_keys = {key for key, row in provenance.items() if row.get("proposal_source", "").endswith("_geometry")}
    geometry_hits = 0
    for key in geometry_keys:
        pred = prediction["components"].get(key)
        if pred is None: continue
        if any(gt["type"] == pred["type"] and bbox_error(gt["bbox"], pred["bbox"]) <= 20 for gt in target["components"].values()):
            geometry_hits += 1

    errors = Counter()
    matched_gt = {i for i, _ in type_pairs}
    bbox_gt = {i for i, _ in bbox_pairs}
    for index, (key, gt) in enumerate(gt_items):
        if index not in bbox_gt:
            close = min((bbox_error(gt["bbox"], pred["bbox"]) for _, pred in pred_items), default=math.inf)
            errors["BBox Error" if close <= 40 else "Detection Miss"] += 1
        elif index not in matched_gt:
            errors["Type Error"] += 1
        if key in observable and key.upper() not in tokens:
            errors["Designator OCR Error"] += 1
        elif key in recognized and key not in associated:
            errors["Designator Association Error"] += 1
        if key in prediction["components"]:
            pred = prediction["components"][key]
            if gt.get("Name") is not None and pred.get("Name") != gt.get("Name"):
                errors["Name Error"] += 1
            if gt.get("value") is not None and pred.get("value") != gt.get("value"):
                errors["Value Error"] += 1
    errors["False Positive"] += max(0, len(pred_items) - len(type_pairs))
    errors["Duplicate Prediction"] += duplicate_pairs

    full = evaluate_case(prediction, target, diagnostics)
    return {
        "strict_component": full["metrics"]["Component"],
        "downstream": {name: full["metrics"][name] for name in ("Pin", "NetHypergraph", "NetLine", "PinPair")},
        "FinalScore": full["FinalScore"],
        "symbol_type_bbox": type_bbox,
        "bbox": bbox,
        "type_accuracy": {"correct": type_correct, "matched_bbox": len(bbox_pairs), "accuracy": type_accuracy},
        "designator_ocr": metric(len(recognized), len(observable), len(observable)),
        "designator_association": metric(len(associated), len(recognized), len(recognized)),
        "component_key_recall": metric(len(key_hits), len(pred_items), len(observable)),
        "name_accuracy": {"correct": name_hits, "total": name_total, "accuracy": name_hits / name_total if name_total else 1.0},
        "value_accuracy": {"correct": value_hits, "total": value_total, "accuracy": value_hits / value_total if value_total else 1.0},
        "duplicate_component_pairs": duplicate_pairs,
        "false_positive_count": max(0, len(pred_items) - len(type_pairs)),
        "geometry_fallback": metric(geometry_hits, len(geometry_keys), len(gt_items)),
        "errors": dict(errors),
    }


def aggregate(rows):
    groups = {"overall": rows, "dev_30_no_training_overlap": [r for r in rows if int(r["case_id"]) % 5 == 0], "train_120_in_sample": [r for r in rows if int(r["case_id"]) % 5 != 0]}
    for source in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other"):
        groups[source] = [r for r in rows if r["source"] == source]
    output = {}
    for name, subset in groups.items():
        if not subset: continue
        block = {"case_count": len(subset)}
        for family in ("strict_component", "symbol_type_bbox", "bbox", "designator_ocr", "designator_association", "component_key_recall", "geometry_fallback"):
            values = [r["evaluation"][family] for r in subset]
            block[family] = metric(sum(v["tp"] for v in values), sum(v["pred"] for v in values), sum(v["gt"] for v in values))
            block[family]["macro_f1"] = sum(v["f1"] for v in values) / len(values)
        for family in ("Pin", "NetHypergraph", "NetLine", "PinPair"):
            values = [r["evaluation"]["downstream"][family] for r in subset]
            block[family] = metric(sum(v["tp"] for v in values), sum(v["pred"] for v in values), sum(v["gt"] for v in values))
            block[family]["macro_f1"] = sum(v["f1"] for v in values) / len(values)
        for family in ("type_accuracy", "name_accuracy", "value_accuracy"):
            values = [r["evaluation"][family] for r in subset]
            denominator_key = "matched_bbox" if family == "type_accuracy" else "total"
            correct = sum(v["correct"] for v in values); total = sum(v[denominator_key] for v in values)
            block[family] = {"correct": correct, denominator_key: total, "accuracy": correct / total if total else 1.0}
        block["duplicate_component_pairs"] = sum(r["evaluation"]["duplicate_component_pairs"] for r in subset)
        block["false_positive_per_image"] = sum(r["evaluation"]["false_positive_count"] for r in subset) / len(subset)
        block["error_classes"] = dict(sum((Counter(r["evaluation"]["errors"]) for r in subset), Counter()))
        block["FinalScore_macro"] = sum(r["evaluation"]["FinalScore"] for r in subset) / len(subset)
        output[name] = block
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--case_start", type=int, default=1)
    parser.add_argument("--case_end", type=int, default=150)
    parser.add_argument("--case_ids", help="Optional comma-separated subset within 0001..0150")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case_ids:
        numbers = [int(value.strip()) for value in args.case_ids.split(",") if value.strip()]
        if not numbers or len(numbers) != len(set(numbers)) or any(number < 1 or number >= HOLDOUT_START for number in numbers):
            raise SystemExit("case_ids must be unique cases within 0001..0150")
    elif not (1 <= args.case_start <= args.case_end < HOLDOUT_START):
        raise SystemExit("Holdout guard: only 0001..0150 may be evaluated")
    else:
        numbers = list(range(args.case_start, args.case_end + 1))
    rows, failures = [], []
    for number in numbers:
        case_id = f"{number:04d}"
        try:
            case = args.target_root / case_id
            target_path = next(case.glob("*_target.json")); image_path = next(case.glob("*.png"))
            prediction = json.loads((args.prediction_dir / case_id / "result.json").read_text(encoding="utf-8"))
            diagnostics_path = args.prediction_dir / case_id / "diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {}
            target = json.loads(target_path.read_text(encoding="utf-8"))
            rows.append({"case_id": case_id, "source": source_from_name(image_path.name), "evaluation": component_case(prediction, target, diagnostics)})
        except Exception as exc:
            failures.append({"case_id": case_id, "error": repr(exc)})
    result = {
        "OFFICIAL_SCORE": OFFICIAL_SCORE, "case_start": min(numbers), "case_end": max(numbers),
        "case_ids": [f"{number:04d}" for number in numbers],
        "sealed_holdout_used": False, "success_count": len(rows), "failure_count": len(failures),
        "training_overlap_note": "best.pt trained on non-multiples of 5; use dev_30_no_training_overlap for generalization",
        "aggregate": aggregate(rows), "cases": rows, "failures": failures,
    }
    write_json(args.output, result)
    print(json.dumps({"success_count": len(rows), "failure_count": len(failures), "overall": result["aggregate"].get("overall"), "dev": result["aggregate"].get("dev_30_no_training_overlap")}, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
