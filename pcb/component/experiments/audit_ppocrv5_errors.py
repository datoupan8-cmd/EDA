"""Read-only error audit of saved PP-OCRv5 predictions; exactly dev0001..0150."""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluate_v2 import bbox_error, component_key_map, component_counts
from tools.team_access_guard import Guard

CASE_IDS = tuple(f"{n:04d}" for n in range(1, 151))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/component_ppocrv5_full_ab_dev150_20261013")
    parser.add_argument("--ocr-run", type=Path, default=ROOT / "runs/component_alternative_frontends_dev150_20261013")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    guard = Guard()
    root = args.dataset.resolve()
    if root.name != "200_train_cases":
        raise ValueError("Require the explicit development dataset root")
    report = load(args.run / "report.json")
    if tuple(report["ACTUAL_CASE_IDS"]) != CASE_IDS or tuple(row["case_id"] for row in report["cases"]) != CASE_IDS:
        raise ValueError("Require exactly 0001..0150, without duplicates or omissions")
    if report["failure_count"] or report["sealed_holdout_used"]:
        raise ValueError("Require a successful development-only run")
    sources, targets = [], []
    seal = load(args.run / "predictions_sealed_before_GT.json")["prediction_sha256"]
    for cid in CASE_IDS:
        folder = root / cid
        if not folder.is_dir() or folder.resolve() != folder or folder.name != cid:
            raise ValueError(f"Missing or redirected development case {cid}")
        gt_files = list(folder.glob("*_target*.json"))
        if len(gt_files) != 1 or gt_files[0].resolve().parent != folder:
            raise ValueError(f"{cid}: require one canonical target")
        prediction = args.run / cid / "yolo_ppocrv5/result.json"
        ocr = args.ocr_run / cid / "ocr_v5.json"
        if not prediction.is_file() or not ocr.is_file():
            raise ValueError(f"Missing saved prediction or OCR for {cid}")
        if hashlib.sha256(prediction.read_bytes()).hexdigest() != seal[f"{cid}/yolo_ppocrv5/result.json"]:
            raise ValueError(f"Saved prediction changed for {cid}")
        sources.append((cid, prediction, ocr))
        targets.append(gt_files[0])
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Use a new output directory outside the read-only dataset")
    print(json.dumps({"event": "audit_start", "ACTUAL_CASE_IDS": CASE_IDS,
                      "sealed_holdout_used": False, "algorithm_or_threshold_modified": False}), flush=True)
    reasons, fields, identities, placeholders = Counter(), Counter(), Counter(), Counter()
    identity_by_type = defaultdict(Counter)
    by_type, examples = defaultdict(Counter), defaultdict(list)
    rows = []
    for (cid, prediction, ocr_path), target in zip(sources, targets):
        pred = load(prediction)["components"]
        ocr = load(ocr_path)
        tokens = {re.sub(r"\s+", "", text["text"]).upper() for text in ocr}
        with guard.evaluating(target, [prediction]):
            gt = load(target)["components"]
        mapping = component_key_map(pred, gt)
        reverse = {g: p for p, g in mapping.items()}
        current = Counter()
        for key, expected in gt.items():
            candidate = pred.get(reverse.get(key))
            if candidate is None:
                nearby = [p for p in pred.values() if bbox_error(p["bbox"], expected["bbox"]) <= 20]
                same = [p for p in nearby if p["type"] == expected["type"]]
                reason = "no_correct_key_but_type_and_bbox_present" if same else (
                    "no_correct_key_but_other_type_bbox_present" if nearby else "no_correct_key_or_nearby_bbox")
                if same:
                    kind = "GT_placeholder_key" if re.fullmatch(r"∅\d+", key) else (
                        "exact_key_string_in_global_OCR" if re.sub(r"\s+", "", key).upper() in tokens
                        else "exact_key_string_absent_from_global_OCR")
                    identities[kind] += 1
                    identity_by_type[expected["type"]][kind] += 1
                    if any(pk.casefold() == key.casefold() and p["type"] == expected["type"]
                           and bbox_error(p["bbox"], expected["bbox"]) <= 20 for pk, p in pred.items()):
                        identities["case_only_key_difference"] += 1
            elif candidate["type"] != expected["type"]:
                reason = "correct_key_wrong_type"
            elif bbox_error(candidate["bbox"], expected["bbox"]) > 20:
                reason = "correct_key_and_type_wrong_bbox"
            elif expected["type"] == "box" and candidate.get("Name") != expected.get("Name"):
                reason = "correct_key_type_bbox_wrong_model_name"
            elif expected["type"] in ("r", "c") and candidate.get("value") != expected.get("value"):
                reason = "correct_key_type_bbox_missing_value" if candidate.get("value") is None else (
                    "correct_key_type_bbox_wrong_value")
            else:
                reason = "correct"
            if candidate is not None and candidate["type"] == expected["type"] and bbox_error(candidate["bbox"], expected["bbox"]) <= 20:
                if expected["type"] == "box":
                    fields["model_name_correct" if candidate.get("Name") == expected.get("Name") else "model_name_wrong"] += 1
                if expected["type"] in ("r", "c"):
                    fields["value_correct" if candidate.get("value") == expected.get("value") else "value_wrong"] += 1
                    if candidate.get("value") != expected.get("value"):
                        fields["value_wrong_target_is_null" if expected.get("value") is None
                               else "value_wrong_target_is_not_null"] += 1
            reasons[reason] += 1
            current[reason] += 1
            by_type[expected["type"]][reason] += 1
            if re.fullmatch(r"∅\d+", key):
                placeholders[reason] += 1
            if reason != "correct" and len(examples[reason]) < 8:
                examples[reason].append({"case_id": cid, "key": key, "target": expected, "prediction": candidate})
        tp, _, _ = component_counts(pred, gt, mapping)
        if tp != current["correct"]:
            raise AssertionError(f"Audit classifications do not reproduce scoring TP for {cid}")
        rows.append({"case_id": cid, "reasons": dict(current), "tp": tp, "gt": len(gt), "pred": len(pred)})
    metric = report["aggregate"]["yolo_ppocrv5"]["overall"]["metrics"]["Component"]
    if reasons["correct"] != metric["tp"] or sum(reasons.values()) != metric["gt"]:
        raise AssertionError("Audit differs from the saved full-pipeline score")
    output.mkdir(parents=True)
    result = {"ACTUAL_CASE_IDS": CASE_IDS, "sealed_holdout_used": False, "OFFICIAL_SCORE": False,
              "source_run": str(args.run.resolve()), "GT_opens": guard.target_opens,
              "algorithm_or_threshold_modified": False,
              "gt": sum(reasons.values()), "strict_tp": reasons["correct"],
              "reasons": dict(reasons), "same_geometry_missing_identity": dict(identities),
              "identity_mismatches_by_type": {k: dict(v) for k, v in identity_by_type.items()},
              "GT_placeholder_key_reasons": dict(placeholders),
              "fields_when_key_type_bbox_are_correct": dict(fields),
              "by_type": {k: dict(v) for k, v in by_type.items()}, "examples": dict(examples), "cases": rows,
              "limitations": ["Reasons are a first-failure partition in the existing scorer; bbox errors can also reflect swapped identities.",
                               "A nearby proposal is not a one-to-one geometry match and is not detector AP.",
                               "Global OCR string presence does not prove correct local assignment or visible GT identifiers.",
                               "Placeholder GT keys are reported separately and are not invented from the image."]}
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "GT_opens": guard.target_opens, "reasons": dict(reasons),
                      "same_geometry_missing_identity": dict(identities),
                      "fields": dict(fields), "GT_placeholder_key_reasons": dict(placeholders),
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
