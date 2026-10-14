"""Factorial PNG evaluation of independent detector and OCR alternatives on dev150."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.run_team_pipeline import build_runtime, select_inputs
from tools.team_access_guard import Guard
from pcb.core.context import PipelineContext
from pcb.core.interfaces import TextStageOutput
from pcb.core.team_config import load_team_config
from pcb.component.alternative_rcnn import RcnnDetector, assemble_rcnn_component_output
from pcb.component.alternative_ocr import PPOcrV5Backend, MODEL_HASHES
from pcb.schema import Scene
from pcb.io import read_image
from pcb.submission import export, validate_strict
from evaluate_v2 import component_counts, component_key_map, prf, bbox_error
from pcb.assignment import maximum_weight_assignment


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def geometry_counts(prediction, target):
    """Diagnostic only: exact type and <=20px bbox error, ignoring text identity."""
    pred, gt = list(prediction.values()), list(target.values())
    weights = [[float(p["type"] == g["type"] and bbox_error(p["bbox"], g["bbox"]) <= 20)
                for g in gt] for p in pred]
    tp = sum(weights[i][j] > 0 for i, j in maximum_weight_assignment(weights))
    return prf(tp, len(pred), len(gt))


def aggregate(records, modes):
    output = {}
    for mode in modes:
        values = [row["evaluation"][mode]["component"] for row in records]
        geometries = [row["evaluation"][mode]["geometry"] for row in records]
        output[mode] = {**prf(sum(v["tp"] for v in values), sum(v["pred"] for v in values), sum(v["gt"] for v in values)),
                        "macro_f1": sum(v["f1"] for v in values) / len(values),
                        "geometry_macro_f1": sum(v["f1"] for v in geometries) / len(geometries),
                        "geometry_micro": prf(sum(v["tp"] for v in geometries),
                                              sum(v["pred"] for v in geometries), sum(v["gt"] for v in geometries))}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/component_fasterrcnn/final.pth")
    parser.add_argument("--cases", help="Unique development IDs; default all 0001..0150")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    guard = Guard()
    inputs = select_inputs(None, args.dataset, args.cases)
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    for cid, image, folder in inputs:
        if folder.resolve() != folder or folder.name != cid or image.resolve().parent != folder:
            raise ValueError("Case aliases or redirects are prohibited")
    args.output.mkdir(parents=True)
    source_files = sorted(set(list((ROOT / "pcb/component").rglob("*.py")) +
                              list((ROOT / "configs").rglob("*.json")) +
                              [ROOT / "evaluate_v2.py", ROOT / "assets/cpntLibrary.json",
                               ROOT / "pcb/submission.py", ROOT / "tools/run_team_pipeline.py", Path(__file__).resolve()]))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    config = load_team_config(ROOT / "configs/component/team_contextual_values_v2.json")
    pipeline, old_ocr, yolo, device = build_runtime(config, args.device, args.device, ROOT / "runs/team_cache")
    rcnn = RcnnDetector(args.checkpoint, device=device, cache_dir=ROOT / "runs/component_alternative_cache/rcnn")
    new_ocr = PPOcrV5Backend(ROOT / "runs/component_alternative_cache/ocr_v5")
    modes = ("baseline_v2", "rcnn_legacy_ocr", "yolo_ppocrv5", "rcnn_ppocrv5")
    print(json.dumps({"event": "start", "ACTUAL_CASE_IDS": [cid for cid, _, _ in inputs],
                      "training_case_ids": rcnn.training_ids, "OFFICIAL_SCORE": False,
                      "sealed_holdout_used": False, "modes": modes}), flush=True)
    write(args.output / "frozen_manifest.json", {"ACTUAL_CASE_IDS": [cid for cid, _, _ in inputs],
        "rcnn_weights_sha256": rcnn.weight_sha256, "yolo_weights_sha256": yolo.weight_sha256,
        "ocr_v5_models_sha256": MODEL_HASHES, "source_sha256": hashes, "GT_opens": guard.target_opens})
    records, start = [], time.perf_counter()
    for cid, path, _ in inputs:
        case_start = time.perf_counter()
        with guard.inferring():
            image = read_image(path)
            height, width = image.shape[:2]
            context = PipelineContext(path.name, width, height, config, old_ocr, yolo)
            old_text = pipeline.text.run(image, context)
            baseline = pipeline.component.run(image, old_text, context)
            new_text = new_ocr.recognize(image)
            proposals, detector_diagnostics = rcnn.detect(image)
            results = {"baseline_v2": baseline,
                       "rcnn_legacy_ocr": assemble_rcnn_component_output(proposals, old_text.texts, detector_diagnostics),
                       "yolo_ppocrv5": pipeline.component.run(image, TextStageOutput(new_text), context),
                       "rcnn_ppocrv5": assemble_rcnn_component_output(proposals, new_text, detector_diagnostics)}
            for mode, result in results.items():
                scene = Scene(width, height, result.components, result.texts, diagnostics=result.diagnostics)
                prediction = export(scene)
                validate_strict(prediction, (width, height))
                write(args.output / cid / mode / "result.json", prediction)
                write(args.output / cid / mode / "diagnostics.json", result.diagnostics)
            write(args.output / cid / "ocr_v5.json", [asdict(token) for token in new_text])
        record = {"case_id": cid, "image_name": path.name, "status": "ok",
                  "seconds": time.perf_counter() - case_start,
                  "split": "development_check30" if int(cid) % 5 == 0 else "training_fit120",
                  "component_counts": {mode: len(result.components) for mode, result in results.items()},
                  "text_counts": {"old": len(old_text.texts), "new": len(new_text)}}
        records.append(record)
        write(args.output / "progress.json", {"cases": records, "GT_opens": guard.target_opens})
        print(json.dumps({"event": "case_complete", **record}), flush=True)
    saved = [args.output / cid / mode / "result.json" for cid, _, _ in inputs for mode in modes]
    if guard.target_opens or not all(path.is_file() for path in saved):
        raise AssertionError("All four arms must be saved before any evaluation target opens")
    for record, (cid, _, folder) in zip(records, inputs):
        targets = list(folder.glob("*_target*.json"))
        if len(targets) != 1 or targets[0].resolve().parent != folder:
            raise ValueError(f"{cid}: require one allowed evaluation target")
        with guard.evaluating(targets[0], saved):
            target = json.loads(targets[0].read_text(encoding="utf-8"))
        record["evaluation"] = {}
        for mode in modes:
            prediction = json.loads((args.output / cid / mode / "result.json").read_text(encoding="utf-8"))
            counts = component_counts(prediction["components"], target["components"],
                                      component_key_map(prediction["components"], target["components"]))
            record["evaluation"][mode] = {"component": prf(*counts),
                                          "geometry": geometry_counts(prediction["components"], target["components"])}
    groups = {"all_development150": records}
    for split in ("development_check30", "training_fit120"):
        subset = [row for row in records if row["split"] == split]
        if subset:
            groups[split] = subset
    aggregates = {name: aggregate(rows, modes) for name, rows in groups.items()}
    paired = {}
    for mode in modes:
        values = [row["evaluation"][mode]["component"]["f1"] -
                  row["evaluation"]["baseline_v2"]["component"]["f1"] for row in records]
        paired[mode] = {"improved": sum(v > 1e-12 for v in values), "regressed": sum(v < -1e-12 for v in values),
                        "unchanged": sum(abs(v) <= 1e-12 for v in values)}
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}:
        raise ValueError("Frozen algorithm/configs changed during the experiment")
    report = {"ACTUAL_CASE_IDS": [cid for cid, _, _ in inputs], "OFFICIAL_SCORE": False,
              "sealed_holdout_used": False, "Component_only": True,
              "mode": "independent_detector_ocr_factorial_png", "source_sha256": hashes,
              "rcnn_weights_sha256": rcnn.weight_sha256, "yolo_weights_sha256": yolo.weight_sha256,
              "ocr_v5_models_sha256": MODEL_HASHES, "training_case_ids": rcnn.training_ids,
              "training_fit_scores_are_not_generalization_estimates": True,
              "all_predictions_saved_before_GT": True, "success_count": len(records),
              "failure_count": 0, "GT_opens": guard.target_opens, "seconds": time.perf_counter() - start,
              "aggregate": aggregates, "paired_cases": paired, "cases": records}
    write(args.output / "report.json", report)
    print(json.dumps({"event": "complete", "aggregate": aggregates, "paired_cases": paired}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
