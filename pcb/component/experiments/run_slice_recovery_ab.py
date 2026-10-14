"""Actual PNG slice-recovery ablation on the fixed development whitelist."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.run_team_pipeline import build_runtime, select_inputs, project_path
from tools.team_access_guard import Guard
from pcb.core.context import PipelineContext
from pcb.core.team_config import load_team_config
from pcb.core.interfaces import ComponentStageOutput
from pcb.component.latest.stage import ComponentStageV4
from pcb.component.contextual_values import refine_contextual_values
from pcb.component.slice_recovery import recover_slice_components
from pcb.component.text_enrichment import enrich_fields
from pcb.schema import Scene
from pcb.io import read_image
from pcb.submission import export, validate_strict
from evaluate_v2 import component_counts, component_key_map, prf


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", help="Unique development IDs; default 0001..0150")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs/team_cache")
    args = parser.parse_args()
    guard = Guard()
    inputs = select_inputs(None, args.dataset, args.cases)
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    files = sorted(list((ROOT / "pcb/component").rglob("*.py")) + [ROOT / "evaluate_v2.py",
                   ROOT / "pcb/submission.py", ROOT / "assets/cpntLibrary.json"])
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    config = load_team_config(ROOT / "configs/component/team_text_enriched_v1.json")
    pipeline, ocr, detector, device = build_runtime(config, args.device, args.device, project_path(args.cache_dir))
    baseline_stage = ComponentStageV4()
    print(json.dumps({"event": "start", "ACTUAL_CASE_IDS": [cid for cid, _, _ in inputs],
                      "mode": "slice_recovery_png_ablation", "device": device,
                      "OFFICIAL_SCORE": False, "sealed_holdout_used": False}), flush=True)
    records = []
    for cid, image_path, _ in inputs:
        start = time.perf_counter()
        with guard.inferring():
            image = read_image(image_path)
            height, width = image.shape[:2]
            context = PipelineContext(image_path.name, width, height, config, ocr, detector)
            text = pipeline.text.run(image, context)
            original = baseline_stage.run(image, text, context)
            components, diagnostics = enrich_fields(original.components, original.roles, original.diagnostics)
            current = ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)
            contextual = refine_contextual_values(original, conflicts=True, pin_boxes=True)
            results = {"baseline_v1": current, "contextual_values": contextual,
                       "slice_only": recover_slice_components(image, current, detector),
                       "both": recover_slice_components(image, contextual, detector)}
            for mode, result in results.items():
                if result.components[:len(current.components)] != results[
                        "contextual_values" if mode == "both" else "baseline_v1" if mode == "slice_only" else mode].components:
                    raise AssertionError("Slice recovery must preserve every existing component")
                scene = Scene(width, height, result.components, result.texts, diagnostics=result.diagnostics)
                prediction = export(scene)
                validate_strict(prediction, (width, height))
                write(args.output / cid / mode / "result.json", prediction)
                write(args.output / cid / mode / "diagnostics.json", result.diagnostics)
            record = {"case_id": cid, "image_name": image_path.name, "status": "ok",
                      "seconds": time.perf_counter() - start,
                      "component_counts": {mode: len(result.components) for mode, result in results.items()}}
            records.append(record)
        write(args.output / "progress.json", {"ACTUAL_CASE_IDS": [cid for cid, _, _ in inputs],
                                              "cases": records, "GT_opens": guard.target_opens})
        print(json.dumps({"event": "case_complete", **record}), flush=True)
    all_saved = [args.output / cid / mode / "result.json" for cid, _, _ in inputs for mode in results]
    if guard.target_opens or not all(path.is_file() for path in all_saved):
        raise AssertionError("All predictions must be saved before development GT is opened")
    for record, (cid, _, folder) in zip(records, inputs):
        targets = list(folder.glob("*_target*.json"))
        if len(targets) != 1 or targets[0].resolve().parent != folder.resolve():
            raise ValueError(f"{cid}: require one allowed target")
        with guard.evaluating(targets[0], all_saved):
            target = json.loads(targets[0].read_text(encoding="utf-8"))
        record["evaluation"] = {}
        for mode in results:
            prediction = json.loads((args.output / cid / mode / "result.json").read_text(encoding="utf-8"))
            counts = component_counts(prediction["components"], target["components"],
                                      component_key_map(prediction["components"], target["components"]))
            record["evaluation"][mode] = prf(*counts)
    aggregate, paired = {}, {}
    for mode in results:
        values = [r["evaluation"][mode] for r in records]
        aggregate[mode] = {**prf(sum(v["tp"] for v in values), sum(v["pred"] for v in values), sum(v["gt"] for v in values)),
                           "macro_f1": sum(v["f1"] for v in values) / len(values)}
        differences = [r["evaluation"][mode]["f1"] - r["evaluation"]["baseline_v1"]["f1"] for r in records]
        paired[mode] = {"improved": sum(d > 1e-12 for d in differences), "regressed": sum(d < -1e-12 for d in differences),
                        "unchanged": sum(abs(d) <= 1e-12 for d in differences)}
    if hashes != {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}:
        raise ValueError("Component source changed during the experiment")
    report = {"ACTUAL_CASE_IDS": [cid for cid, _, _ in inputs], "OFFICIAL_SCORE": False,
              "Component_only": True, "sealed_holdout_used": False, "mode": "slice_recovery_png_ablation",
              "config": config.as_dict(), "aggregate": aggregate, "paired_cases": paired, "cases": records,
              "success_count": len(records), "failure_count": 0, "GT_opens": guard.target_opens,
              "source_sha256": hashes, "weights_sha256": detector.weight_sha256,
              "all_predictions_saved_before_GT": True}
    write(args.output / "report.json", report)
    print(json.dumps({"event": "complete", "aggregate": aggregate, "paired_cases": paired}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
