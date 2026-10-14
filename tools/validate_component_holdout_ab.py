"""Explicitly authorized, read-only 0151..0200 Component validation.

This is an evaluation entry, not a development or parameter-selection runner.
Development case selectors and access guards remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE_IDS = tuple(f"{number:04d}" for number in range(151, 201))


def write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class ValidationGuard:
    """Permit only approved images and post-inference targets; reject dataset writes."""

    def __init__(self, dataset: Path):
        self.dataset = dataset.resolve()
        self.phase = "idle"
        self.allowed_image = None
        self.allowed_target = None
        self.target_opens = 0
        self.blocked = 0
        sys.addaudithook(self.audit)

    def reject(self, message):
        self.blocked += 1
        raise PermissionError(message)

    def audit(self, event, args):
        if event not in {"open", "os.listdir", "os.scandir"} or not args:
            return
        if not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        parts = [part.lower() for part in path.parts]
        if any(part in {"golden", "sealed", "eda_pin_crossing_quicktest"}
               or part.startswith(("10gt", "10_gtcase")) for part in parts):
            self.reject("Other reserved datasets are outside this validation authorization")
        in_dataset = path.is_relative_to(self.dataset)
        if "200_train_cases" in parts and not in_dataset:
            self.reject("Other datasets are outside this validation authorization")
        if in_dataset:
            relative = path.relative_to(self.dataset).parts
            if relative and relative[0] not in CASE_IDS:
                self.reject("Only explicitly authorized validation cases 0151..0200")
            if event == "open":
                mode = args[1] if len(args) > 1 else None
                flags = args[2] if len(args) > 2 else 0
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+"))
                writing = writing or (isinstance(flags, int) and bool(flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)))
                if writing:
                    self.reject("Validation must never write into the dataset")
                if path == self.allowed_image and self.phase == "inference":
                    return
                if path == self.allowed_target and self.phase == "evaluation":
                    self.target_opens += 1
                    return
                self.reject("Dataset reads require an approved inference image or evaluation target")
        elif event == "open" and path.suffix.lower() == ".json" and "_target" in path.name.lower():
            self.reject("No other annotations may be opened")

    @contextmanager
    def inferring(self, image):
        self.phase, self.allowed_image = "inference", image.resolve()
        try:
            yield
        finally:
            self.phase, self.allowed_image = "idle", None

    @contextmanager
    def evaluating(self, target, all_saved):
        if len(all_saved) != 100 or not all(path.is_file() for path in all_saved):
            raise AssertionError("Save both arms for every one of the 50 cases before GT access")
        self.phase, self.allowed_target = "evaluation", target.resolve()
        try:
            yield
        finally:
            self.phase, self.allowed_target = "idle", None


def select_cases(dataset: Path):
    if dataset.name != "200_train_cases" or not dataset.is_dir():
        raise ValueError("--dataset must be the existing 200_train_cases directory")
    # Check identifiers only, without descending into any unapproved directory.
    numeric_names = [path.name for path in dataset.iterdir() if path.name.isdigit()]
    for cid in CASE_IDS:
        aliases = [name for name in numeric_names if int(name) == int(cid)]
        if aliases != [cid]:
            raise ValueError(f"{cid}: missing or duplicate case ID; validation stops")
    rows = []
    for cid in CASE_IDS:
        folder = dataset / cid
        if not folder.is_dir() or folder.resolve() != folder:
            raise ValueError(f"{cid}: require a canonical case directory without redirection")
        images = list(folder.glob("*.png"))
        if len(images) != 1 or images[0].resolve().parent != folder:
            raise ValueError(f"{cid}: require exactly one PNG within this case")
        rows.append((cid, images[0], folder))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-holdout-readonly", action="store_true",
                        help="Required explicit opt-in; only after user authorizes validation")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.validate_holdout_readonly:
        parser.error("Holdout access is disabled without --validate-holdout-readonly")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.is_relative_to(dataset) or output.exists():
        raise ValueError("Use a fresh output directory outside the dataset")
    output.mkdir(parents=True)
    cache = output / "validation_cache"
    guard = ValidationGuard(dataset)
    # Freeze all project production code/configs before discovering case inputs.
    source_files = sorted(set(
        list((ROOT / "pcb").rglob("*.py")) + list((ROOT / "configs").rglob("*.json"))
        + [ROOT / "assets/cpntLibrary.json", ROOT / "evaluate_v2.py",
           ROOT / "tools/run_team_pipeline.py", Path(__file__).resolve()]))
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    model_files = [ROOT / "models/component_yolo11n_continue_v2_best.pt"]
    model_files.extend(sorted((ROOT / "models/easyocr").glob("*.pth")))
    model_hashes = {str(path.relative_to(ROOT)): digest(path) for path in model_files}
    write(output / "frozen_manifest.json", {
        "ACTUAL_CASE_IDS": CASE_IDS, "authorization": "User requested last 50 cases",
        "purpose": "read_only_validation", "OFFICIAL_SCORE": False,
        "source_sha256": source_hashes, "model_sha256": model_hashes,
        "isolated_validation_cache": str(cache), "GT_opens": guard.target_opens})
    rows = select_cases(dataset)

    from tools.run_team_pipeline import build_runtime, parse_ids
    from pcb.core.context import PipelineContext
    from pcb.core.team_config import load_team_config
    from pcb.component.latest.stage import ComponentStageV4
    from pcb.schema import Scene
    from pcb.io import read_image
    from pcb.submission import export, validate_strict
    from evaluate_v2 import component_counts, component_key_map, prf

    # Prove this explicit evaluator did not relax the development whitelist.
    try:
        parse_ids("0151")
    except PermissionError:
        pass
    else:
        raise AssertionError("Development whitelist unexpectedly permits holdouts")
    config = load_team_config(ROOT / "configs/component/team_text_enriched_v1.json")
    baseline_config = load_team_config(ROOT / "configs/component/team_peer_latest.json")
    if config.component != "text_enriched_v1" or baseline_config.component != "peer_latest":
        raise AssertionError("Expected frozen baseline and candidate component selections")
    pipeline, ocr, detector, device = build_runtime(config, args.device, args.device, cache)
    baseline_stage = ComponentStageV4()
    print(json.dumps({"event": "start", "ACTUAL_CASE_IDS": CASE_IDS,
                      "mode": "holdout_png_component_ab", "device": device,
                      "GT_opens": guard.target_opens, "OFFICIAL_SCORE": False}), flush=True)
    records, all_saved = [], []
    started = time.perf_counter()
    for cid, image_path, _ in rows:
        start = time.perf_counter()
        with guard.inferring(image_path):
            image = read_image(image_path)
            height, width = image.shape[:2]
            context = PipelineContext(image_path.name, width, height, config, ocr, detector)
            texts = pipeline.text.run(image, context)
            before = baseline_stage.run(image, texts, context)
            after = pipeline.component.run(image, texts, context)
            fields = [[asdict(c) for c in result.components] for result in (before, after)]
            for components in fields:
                for component in components:
                    component.pop("name")
                    component.pop("value")
            if fields[0] != fields[1]:
                raise AssertionError("Frozen candidate may change only name/value")
            for mode, result in (("baseline", before), ("candidate", after)):
                scene = Scene(width, height, result.components, result.texts,
                              diagnostics=result.diagnostics)
                prediction = export(scene)
                validate_strict(prediction, (width, height))
                path = output / cid / mode / "result.json"
                write(path, prediction)
                write(path.with_name("diagnostics.json"), result.diagnostics)
                all_saved.append(path)
        record = {"case_id": cid, "image_name": image_path.name, "status": "ok",
                  "seconds": time.perf_counter() - start, "component_count": len(before.components)}
        records.append(record)
        write(output / "progress.json", {"ACTUAL_CASE_IDS": CASE_IDS, "cases": records,
                                         "phase": "inference", "GT_opens": guard.target_opens})
        print(json.dumps({"event": "case_complete", **record}), flush=True)
    # No targets have been opened; seal every prediction before scoring either arm.
    if guard.target_opens != 0 or len(records) != 50 or len(all_saved) != 100:
        raise AssertionError("Expected 50 complete image-only A/B cases with zero GT reads")
    prediction_hashes = {str(path.relative_to(output)): digest(path) for path in all_saved}
    write(output / "predictions_sealed.json", {
        "ACTUAL_CASE_IDS": CASE_IDS, "prediction_sha256": prediction_hashes, "GT_opens": 0})
    if source_hashes != {str(path.relative_to(ROOT)): digest(path) for path in source_files}:
        raise ValueError("Code/configs changed after the pre-validation freeze")
    for record, (cid, _, folder) in zip(records, rows):
        targets = list(folder.glob("*_target*.json"))
        if len(targets) != 1 or targets[0].resolve().parent != folder:
            raise ValueError(f"{cid}: require exactly one target within this case")
        with guard.evaluating(targets[0], all_saved):
            target = json.loads(targets[0].read_text(encoding="utf-8"))
        record["evaluation"] = {}
        for mode in ("baseline", "candidate"):
            prediction = json.loads((output / cid / mode / "result.json").read_text(encoding="utf-8"))
            counts = component_counts(prediction["components"], target["components"],
                                      component_key_map(prediction["components"], target["components"]))
            record["evaluation"][mode] = prf(*counts)
    aggregate = {}
    for mode in ("baseline", "candidate"):
        values = [record["evaluation"][mode] for record in records]
        aggregate[mode] = {
            **prf(sum(v["tp"] for v in values), sum(v["pred"] for v in values), sum(v["gt"] for v in values)),
            "macro_f1": sum(v["f1"] for v in values) / len(values)}
    differences = [record["evaluation"]["candidate"]["f1"] - record["evaluation"]["baseline"]["f1"]
                   for record in records]
    paired = {"improved": sum(value > 1e-12 for value in differences),
              "regressed": sum(value < -1e-12 for value in differences),
              "unchanged": sum(abs(value) <= 1e-12 for value in differences)}
    if source_hashes != {str(path.relative_to(ROOT)): digest(path) for path in source_files}:
        raise ValueError("Code/configs changed during read-only validation")
    if model_hashes != {str(path.relative_to(ROOT)): digest(path) for path in model_files}:
        raise ValueError("Model weights changed during read-only validation")
    if prediction_hashes != {str(path.relative_to(output)): digest(path) for path in all_saved}:
        raise ValueError("Sealed predictions changed during evaluation")
    import platform
    import torch
    report = {
        "ACTUAL_CASE_IDS": CASE_IDS, "OFFICIAL_SCORE": False, "Component_only": True,
        "mode": "holdout_png_component_ab", "purpose": "read_only_validation",
        "sealed_holdout_used": True, "holdout_used_for_training_or_tuning": False,
        "config": config.as_dict(), "baseline_config": baseline_config.as_dict(),
        "aggregate": aggregate, "paired_cases": paired, "cases": records,
        "GT_opens": guard.target_opens, "blocked_accesses": guard.blocked,
        "success_count": len(records), "failure_count": 0,
        "all_predictions_saved_before_GT": True, "prediction_sha256": prediction_hashes,
        "source_sha256": source_hashes, "model_sha256": model_hashes,
        "weights_sha256": detector.weight_sha256, "isolated_validation_cache": str(cache),
        "seconds": time.perf_counter() - started,
        "runtime": {"platform": platform.platform(), "python": sys.version,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
        "production_code_and_models_unchanged": True}
    write(output / "report.json", report)
    write(output / "progress.json", {"ACTUAL_CASE_IDS": CASE_IDS, "phase": "complete",
                                     "success_count": len(records), "GT_opens": guard.target_opens})
    print(json.dumps({"event": "complete", "aggregate": aggregate, "paired_cases": paired,
                      "report": str(output / "report.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
