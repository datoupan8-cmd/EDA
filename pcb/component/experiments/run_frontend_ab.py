"""Run the actual Text/Component stages on allowed PNGs, then score Component only."""
from dataclasses import asdict
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
from pcb.core.registry import build_default_registry
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
    parser.add_argument("--baseline-config", default="configs/component/team_peer_latest.json")
    parser.add_argument("--candidate-config", default="configs/component/team_text_enriched_v1.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs/team_cache")
    args = parser.parse_args()
    inputs = select_inputs(None, args.dataset, args.cases)
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    source_files = tuple(str(p.relative_to(ROOT)) for p in sorted(set(
        list((ROOT / "pcb/component").rglob("*.py")) + list((ROOT / "configs").rglob("*.json"))
        + [ROOT / "assets/cpntLibrary.json", ROOT / "evaluate_v2.py", Path(__file__).resolve()])))
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    guard = Guard()
    config = load_team_config(project_path(args.candidate_config))
    baseline_config = load_team_config(project_path(args.baseline_config))
    candidate_fields, baseline_fields = config.as_dict(), baseline_config.as_dict()
    candidate_fields.pop("component")
    baseline_fields.pop("component")
    if candidate_fields != baseline_fields:
        raise ValueError("Component A/B configs may differ only in the component selection")
    pipeline, ocr, detector, device = build_runtime(config, args.device, args.device, project_path(args.cache_dir))
    baseline_stage = build_default_registry().create("component", baseline_config.component)
    print(json.dumps({"event": "start", "ACTUAL_CASE_IDS": [c for c, _, _ in inputs],
                      "mode": "fresh_png_frontend_ab", "device": device}), flush=True)
    records = []
    for cid, image_path, _ in inputs:
        start = time.perf_counter()
        with guard.inferring():
            image = read_image(image_path)
            height, width = image.shape[:2]
            context = PipelineContext(image_path.name, width, height, config, ocr, detector)
            text = pipeline.text.run(image, context)
            a = baseline_stage.run(image, text, context)
            b = pipeline.component.run(image, text, context)
            a_fields = [asdict(c) for c in a.components]
            b_fields = [asdict(c) for c in b.components]
            for before, after in zip(a_fields, b_fields):
                for field in ("name", "value"):
                    before.pop(field)
                    after.pop(field)
            if a_fields != b_fields:
                raise AssertionError("Only name/value fields may change")
            for mode, result in (("baseline", a), ("candidate", b)):
                scene = Scene(width, height, result.components, result.texts, diagnostics=result.diagnostics)
                prediction = export(scene)
                validate_strict(prediction, (width, height))
                write(args.output / cid / mode / "result.json", prediction)
                write(args.output / cid / mode / "diagnostics.json", result.diagnostics)
            record = {"case_id": cid, "image_name": image_path.name,
                      "seconds": time.perf_counter() - start, "component_count": len(a.components)}
            records.append(record)
        write(args.output / "progress.json", {"cases": records, "GT_opens": guard.target_opens})
        print(json.dumps({"event": "case_complete", **record}), flush=True)
    # All predictions for both arms are saved before the first target is opened.
    for record, (cid, _, folder) in zip(records, inputs):
        targets = list(folder.glob("*_target*.json"))
        if len(targets) != 1 or targets[0].resolve().parent != folder.resolve():
            raise ValueError(f"{cid}: require exactly one allowed target")
        saved = [args.output / cid / mode / "result.json" for mode in ("baseline", "candidate")]
        with guard.evaluating(targets[0], saved):
            target = json.loads(targets[0].read_text(encoding="utf-8"))
        record["evaluation"] = {}
        for mode, path in zip(("baseline", "candidate"), saved):
            prediction = json.loads(path.read_text(encoding="utf-8"))
            counts = component_counts(prediction["components"], target["components"],
                                      component_key_map(prediction["components"], target["components"]))
            record["evaluation"][mode] = prf(*counts)
    aggregate = {}
    for mode in ("baseline", "candidate"):
        values = [r["evaluation"][mode] for r in records]
        aggregate[mode] = {**prf(sum(v["tp"] for v in values), sum(v["pred"] for v in values), sum(v["gt"] for v in values)),
                           "macro_f1": sum(v["f1"] for v in values) / len(values)}
    report = {"ACTUAL_CASE_IDS": [cid for cid, _, _ in inputs], "OFFICIAL_SCORE": False,
              "mode": "fresh_png_frontend_ab", "Component_only": True, "sealed_holdout_used": False,
              "config": config.as_dict(), "aggregate": aggregate, "cases": records,
              "baseline_config": baseline_config.as_dict(), "all_predictions_saved_before_GT": True,
              "GT_opens": guard.target_opens, "success_count": len(records), "failure_count": 0,
              "source_sha256": source_hashes, "weights_sha256": detector.weight_sha256,
              "content_addressed_cache_allowed": True}
    if source_hashes != {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}:
        raise ValueError("Source files changed during the experiment")
    write(args.output / "report.json", report)
    print(json.dumps({"event": "complete", "aggregate": aggregate}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
