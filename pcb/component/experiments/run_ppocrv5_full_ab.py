"""Frozen full-pipeline OCR A/B experiment; default development IDs 0001..0150."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.run_team_pipeline import build_runtime, select_inputs, network_version_snapshot
from tools.team_access_guard import Guard
from pcb.core.context import PipelineContext
from pcb.core.pipeline import ModularPipeline
from pcb.core.registry import build_default_registry
from pcb.core.team_config import load_team_config
from pcb.component.alternative_ocr import MODEL_HASHES
from pcb.io import read_image, write_json
from pcb.submission import validate_strict
from evaluate_v2 import aggregate, evaluate_case, source_from_name

MODES = ("baseline_v2", "yolo_ppocrv5")


def write(path: Path, payload):
    write_json(path, payload)


def digest(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprints(paths):
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def paired_counts(records, metric):
    if metric == "FinalScore":
        differences = [row["evaluation"][MODES[1]][metric] - row["evaluation"][MODES[0]][metric]
                       for row in records]
    else:
        differences = [row["evaluation"][MODES[1]]["metrics"][metric]["f1"] -
                       row["evaluation"][MODES[0]]["metrics"][metric]["f1"] for row in records]
    return {"improved": sum(x > 1e-12 for x in differences),
            "regressed": sum(x < -1e-12 for x in differences),
            "unchanged": sum(abs(x) <= 1e-12 for x in differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", help="Unique IDs in 0001..0150; default all first 150")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    guard = Guard()
    inputs = select_inputs(None, args.dataset, args.cases)
    ids = [cid for cid, _, _ in inputs]
    for cid, image, folder in inputs:
        if folder.resolve() != folder or folder.name != cid or image.resolve().parent != folder:
            raise ValueError("Case aliases and redirects are prohibited")
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(args.dataset.resolve()):
        raise ValueError("Use a new output directory outside the read-only dataset")
    configs = {
        MODES[0]: load_team_config("configs/component/team_contextual_values_v2.json"),
        MODES[1]: load_team_config("configs/component/team_ppocrv5_yolo_v1.json"),
    }
    before, after = (configs[mode].as_dict() for mode in MODES)
    differences = {key for key in before if before[key] != after[key]}
    if differences != {"text"} or before["text"] != "current" or after["text"] != "ppocrv5_mobile":
        raise ValueError("Only the frozen full-image OCR selection may differ")
    sources = sorted(set(list((ROOT / "pcb").rglob("*.py")) +
                         list((ROOT / "configs").rglob("*.json")) +
                         [ROOT / "evaluate_v2.py", ROOT / "tools/team_access_guard.py",
                          ROOT / "tools/run_team_pipeline.py", Path(__file__).resolve()]))
    assets = [ROOT / "models/component_yolo11n_continue_v2_best.pt",
              ROOT / "reports/pin_learned_locator_l1/best.pt",
              ROOT / "assets/cpntLibrary.json"]
    assets += sorted((ROOT / "models/easyocr").glob("*.pth"))
    assets += [ROOT / "models/ocr_v5" / name for name in MODEL_HASHES]
    assets += sorted((ROOT / "vendor/rapidocr_runtime/rapidocr_onnxruntime/models").glob("*.onnx"))
    source_hashes, asset_hashes = fingerprints(sources), fingerprints(assets)
    output.mkdir(parents=True)
    baseline, ocr, detector, device = build_runtime(configs[MODES[0]], args.device, args.device,
                                                   ROOT / "runs/team_cache")
    pipelines = {MODES[0]: baseline,
                 MODES[1]: ModularPipeline(configs[MODES[1]], build_default_registry())}
    networks = {mode: network_version_snapshot(pipeline) for mode, pipeline in pipelines.items()}
    if networks[MODES[0]] != networks[MODES[1]]:
        raise AssertionError("Frozen network versions must match")
    manifest = {"ACTUAL_CASE_IDS": ids, "OFFICIAL_SCORE": False, "sealed_holdout_used": False,
                "configs": {mode: config.as_dict() for mode, config in configs.items()},
                "network_versions": networks, "source_sha256": source_hashes,
                "asset_sha256": asset_hashes, "device": device, "GT_opens": 0,
                "scope": "full_Component_Pin_Wire_Topology", "changed_selection": "text"}
    write(output / "frozen_manifest.json", manifest)
    print(json.dumps({"event": "start", "ACTUAL_CASE_IDS": ids, "modes": MODES,
                      "device": device, "OFFICIAL_SCORE": False, "sealed_holdout_used": False}), flush=True)
    records = []
    start = time.perf_counter()
    for cid, path, _ in inputs:
        case_start = time.perf_counter()
        durations, counts, hashes = {}, {}, {}
        with guard.inferring():
            image = read_image(path)
            height, width = image.shape[:2]
            for mode in MODES:
                t0 = time.perf_counter()
                context = PipelineContext(path.name, width, height, configs[mode], ocr, detector,
                    resources={"pin": {"device": device, "words_cache": ROOT / "runs/team_cache/pin_words"}})
                artifacts = pipelines[mode].run(image, context)
                validate_strict(artifacts.result, (width, height))
                durations[mode] = time.perf_counter() - t0
                artifacts.scene.diagnostics["seconds"] = durations[mode]
                folder = output / cid / mode
                write(folder / "result.json", artifacts.result)
                write(folder / "diagnostics.json", artifacts.scene.diagnostics)
                hashes[mode] = digest(folder / "result.json")
                counts[mode] = {"components": len(artifacts.result["components"]),
                                "pins": sum(map(len, artifacts.result["pins"].values())),
                                "nets": len(artifacts.result["nets"])}
                del artifacts
        record = {"case_id": cid, "image_name": path.name, "source": source_from_name(path.name),
                  "status": "ok", "seconds": time.perf_counter() - case_start,
                  "seconds_by_mode": durations, "counts": counts, "prediction_sha256": hashes}
        records.append(record)
        write(output / "progress.json", {"ACTUAL_CASE_IDS": ids, "cases": records,
                                           "GT_opens": guard.target_opens, "sealed_holdout_used": False})
        print(json.dumps({"event": "case_complete", "case_id": cid, "completed": len(records),
                          "total": len(inputs), "seconds": round(record["seconds"], 2),
                          "seconds_by_mode": {key: round(value, 2) for key, value in durations.items()},
                          "counts": counts, "GT_opens": guard.target_opens}), flush=True)
    saved = [output / cid / mode / "result.json" for cid, _, _ in inputs for mode in MODES]
    if guard.target_opens or not all(path.is_file() for path in saved):
        raise AssertionError("Save all full-pipeline predictions before any GT open")
    if fingerprints(sources) != source_hashes or fingerprints(assets) != asset_hashes:
        raise AssertionError("Frozen source/config/model changed during inference")
    write(output / "predictions_sealed_before_GT.json", {
        "ACTUAL_CASE_IDS": ids, "GT_opens": 0, "all_predictions_saved": True,
        "prediction_sha256": {f"{row['case_id']}/{mode}/result.json": row["prediction_sha256"][mode]
                              for row in records for mode in MODES}})
    for row, (cid, _, folder) in zip(records, inputs):
        targets = list(folder.glob("*_target*.json"))
        if len(targets) != 1 or targets[0].resolve().parent != folder:
            raise ValueError(f"{cid}: require one allowed evaluation target")
        with guard.evaluating(targets[0], saved):
            target = json.loads(targets[0].read_text(encoding="utf-8"))
        row["evaluation"] = {}
        for mode in MODES:
            prediction_path = output / cid / mode / "result.json"
            if digest(prediction_path) != row["prediction_sha256"][mode]:
                raise AssertionError("Saved prediction changed after sealing")
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            diagnostics = json.loads(prediction_path.with_name("diagnostics.json").read_text(encoding="utf-8"))
            row["evaluation"][mode] = evaluate_case(prediction, target, diagnostics)
    aggregates = {mode: aggregate([{**row, "evaluation": row["evaluation"][mode]} for row in records])
                  for mode in MODES}
    paired = {metric: paired_counts(records, metric)
              for metric in ("Component", "Pin", "NetHypergraph", "NetLine", "FinalScore")}
    if fingerprints(sources) != source_hashes or fingerprints(assets) != asset_hashes:
        raise AssertionError("Frozen algorithm or asset changed during evaluation")
    report = {**manifest, "GT_opens": guard.target_opens, "success_count": len(records),
              "failure_count": 0, "blocked_accesses": guard.blocked,
              "all_predictions_saved_and_hashed_before_GT": True,
              "seconds": time.perf_counter() - start,
              "aggregate": aggregates, "paired_cases": paired, "cases": records}
    write(output / "report.json", report)
    print(json.dumps({"event": "complete", "overall": {mode: values["overall"] for mode, values in aggregates.items()},
                      "paired_cases": paired, "GT_opens": guard.target_opens}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
