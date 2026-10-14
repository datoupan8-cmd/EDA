"""Explain Component errors in already saved predictions; never read holdouts."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluate_v2 import component_key_map, bbox_error
from tools.team_access_guard import Guard

CASE_IDS = tuple(f"{n:04d}" for n in range(1, 151))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset.name != "200_train_cases":
        raise ValueError("Require the explicit development dataset root")
    report = json.loads((args.baseline / "report.json").read_text(encoding="utf-8"))
    actual = tuple(row["case_id"] for row in report["cases"])
    if actual != CASE_IDS:
        raise ValueError("Require saved predictions for exactly 0001..0150, without duplicates")
    inputs = []
    for cid in CASE_IDS:
        folder = args.dataset / cid
        if not folder.is_dir() or folder.resolve().parent != args.dataset.resolve():
            raise ValueError(f"Missing or redirected development case {cid}")
        targets = list(folder.glob("*_target*.json"))
        prediction = args.baseline / cid / "result.json"
        scene = args.baseline / cid / "scene.json"
        if len(targets) != 1 or not prediction.is_file() or not scene.is_file():
            raise ValueError(f"{cid}: require one target and saved prediction/scene")
        if targets[0].resolve().parent != folder.resolve():
            raise ValueError(f"Redirected target for {cid}")
        inputs.append((cid, targets[0], prediction, scene))
    print(json.dumps({"event": "audit_start", "ACTUAL_CASE_IDS": actual}), flush=True)
    guard = Guard()
    reasons, unmatched = Counter(), Counter()
    by_type, examples = defaultdict(Counter), defaultdict(list)
    for cid, target, prediction, scene_path in inputs:
        pred = json.loads(prediction.read_text(encoding="utf-8"))
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        with guard.evaluating(target, [prediction]):
            gt = json.loads(target.read_text(encoding="utf-8"))
        mapping = component_key_map(pred["components"], gt["components"])
        reverse = {g: p for p, g in mapping.items()}
        for key, expected in gt["components"].items():
            candidate = pred["components"].get(reverse.get(key))
            if candidate is None:
                close = [p for p in pred["components"].values()
                         if bbox_error(p["bbox"], expected["bbox"]) <= 20]
                same = [p for p in close if p["type"] == expected["type"]]
                reason = ("detected_wrong_identity" if same else
                          "detected_wrong_type_identity" if close else "missing_geometry")
                if same:
                    present = any(re.sub(r"\s+", "", t["text"]).upper().startswith(key.upper())
                                  for t in scene["texts"])
                    reason += ":OCR_present" if present else ":OCR_absent"
            elif candidate["type"] != expected["type"]:
                reason = "wrong_type"
            elif bbox_error(candidate["bbox"], expected["bbox"]) > 20:
                reason = "wrong_bbox"
            elif expected["type"] == "box" and candidate.get("Name") != expected.get("Name"):
                reason = "wrong_model_name"
            elif expected["type"] in ("r", "c") and candidate.get("value") != expected.get("value"):
                reason = "wrong_value_missing" if candidate.get("value") is None else "wrong_value_text"
            else:
                reason = "correct"
            reasons[reason] += 1
            by_type[expected["type"]][reason] += 1
            if reason != "correct" and len(examples[reason]) < 12:
                example = {"case": cid, "key": key, "gt": expected, "pred": candidate}
                if reason == "wrong_model_name":
                    a, b, c, d = expected["bbox"]
                    box = a, scene["height"] - d, c, scene["height"] - b
                    example["near_texts"] = [t["text"] for t in scene["texts"]
                        if box[0] - 40 <= sum(t["bbox"][::2]) / 2 <= box[2] + 40
                        and box[1] - 40 <= sum(t["bbox"][1::2]) / 2 <= box[3] + 40]
                examples[reason].append(example)
        for key, component in pred["components"].items():
            if key not in mapping:
                kind = "placeholder" if key.startswith("UNRESOLVED_") else component["type"]
                unmatched[kind] += 1
    if reasons["correct"] != report["aggregate"]["overall"]["metrics"]["Component"]["tp"]:
        raise ValueError("Current annotations/predictions differ from the recorded baseline")
    output = {"ACTUAL_CASE_IDS": actual, "OFFICIAL_SCORE": False, "sealed_holdout_used": False,
              "gt_components": sum(reasons.values()), "reasons": dict(reasons),
              "by_type": {t: dict(c) for t, c in by_type.items()},
              "pred_unmatched": dict(unmatched), "examples": dict(examples),
              "GT_opens": guard.target_opens}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reasons": dict(reasons), "pred_unmatched": dict(unmatched),
                      "report": str(args.output / "report.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
