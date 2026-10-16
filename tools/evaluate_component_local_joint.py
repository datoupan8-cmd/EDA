"""PNG-only Component ablations. All development inputs are fenced to 0001..0150.

This deliberately does not construct ModularPipeline or any Pin/Wire/Net stage.
Predictions for ALL selected pages are saved and hashed before opening any GT.
The scorer is the existing non-official Component-only proxy in evaluate_v2.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_cases(root, ids=None):
    root = Path(root).resolve()
    if root.name != "200_train_cases" or any(p.lower().startswith(("10gt", "10_gtcase")) for p in root.parts):
        raise PermissionError("Require the public 200_train_cases root")
    requested = list(range(1, 151)) if ids is None else [int(x) for x in ids.split(",")]
    if not requested or len(set(requested)) != len(requested) or any(not 1 <= i <= 150 for i in requested):
        raise PermissionError("Only unique development IDs 0001..0150")
    rows = []
    # Validate the complete whitelist by exact paths; never enumerate the root
    # or inspect 0151..0200. Missing/redirected/duplicate inputs fail closed.
    for number in range(1, 151):
        cid = f"{number:04d}"
        folder = root / cid
        if not folder.is_dir() or folder.resolve() != folder:
            raise ValueError(f"Missing or redirected development directory: {cid}")
        images = list(folder.glob("*.png"))
        if len(images) != 1 or images[0].resolve().parent != folder:
            raise ValueError(f"{cid}: expected exactly one unredirected PNG")
        if number in requested:
            rows.append((cid, images[0], folder))
    if len({path.resolve() for _, path, _ in rows}) != len(rows):
        raise ValueError("Duplicate image paths")
    return rows


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def export_components(output, height):
    from pcb.coordinates import opencv_bbox_to_target
    from pcb.official_types import to_official_type
    exported = {}
    for component in output.components:
        if component.key in exported:
            raise AssertionError("Duplicate component identity")
        exported[component.key] = {
            "Name": component.name or component.observable_designator or component.net_label or component.key,
            "type": to_official_type(component.type), "value": component.value,
            "bbox": [round(v, 3) for v in opencv_bbox_to_target(component.bbox, height)]}
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cases", help="Default is the complete development whitelist")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/component_alternative_cache")
    parser.add_argument("--yolo-cache", type=Path, default=ROOT / "runs/team_cache/yolo")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--variants", default="baseline,local,local_joint",
                        help="baseline,local,local_joint,joint_only,value_groups,guarded_values,ordered_values")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    from tools.team_access_guard import Guard
    guard = Guard()
    rows = select_cases(args.dataset, args.cases)
    variants = args.variants.split(",")
    if "baseline" not in variants or len(set(variants)) != len(variants) or set(variants) - {"baseline", "local", "local_joint", "joint_only", "value_groups", "guarded_values", "ordered_values"}:
        raise ValueError("Invalid ablations; include baseline")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    ids = [cid for cid, _, _ in rows]
    manifest = {"ACTUAL_CASE_IDS": ids, "scope": "Component_only", "OFFICIAL_SCORE": False,
                "sealed_holdout_used": False, "GT_used_in_inference": False,
                "variants": variants, "dataset": str(args.dataset.resolve()),
                "config_before": json.loads((ROOT / "configs/component/current.json").read_text(encoding="utf-8")),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [ROOT / "pcb/component/local_joint.py", ROOT / "pcb/component/alternative_ocr.py",
                                            ROOT / "pcb/component/value_groups_v2.py",
                                            Path(__file__), ROOT / "evaluate_v2.py"]}}
    write(output / "manifest.json", manifest)
    print(json.dumps({"ACTUAL_CASE_IDS": ids, "scope": "Component_only"}), flush=True)
    from pcb.runtime_paths import prepend_compatible_vendor_paths
    prepend_compatible_vendor_paths(ROOT)
    import torch
    from pcb.component.alternative_ocr import TextStagePPOcrV5
    from pcb.component.contextual_values import ComponentStageContextualValues
    from pcb.component.detector_yolo import YoloComponentDetector
    from pcb.component.local_joint import ComponentStageLocalOCR, ComponentStagePrintedGroups, ComponentStageValueGroups, rebuild_fields
    from pcb.component.value_groups_v2 import ComponentStageGuardedValues, ComponentStageOrderedValues
    from pcb.core.config import PipelineConfig
    from pcb.core.context import PipelineContext
    from pcb.io import read_image
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    detector = YoloComponentDetector(ROOT / "models/component_yolo11n_continue_v2_best.pt",
                                    device="0" if device == "cuda" else device, cache_dir=args.yolo_cache)
    text_stage, baseline = TextStagePPOcrV5(), ComponentStageContextualValues()
    local = ComponentStageLocalOCR()
    printed_groups = ComponentStagePrintedGroups()
    value_groups = ComponentStageValueGroups()
    newer_stages = {"guarded_values": ComponentStageGuardedValues(), "ordered_values": ComponentStageOrderedValues()}
    timings, saved = {}, []
    start = time.perf_counter()
    for position, (cid, image_path, folder) in enumerate(rows, 1):
        with guard.inferring():
            image = read_image(image_path)
            height, width = image.shape[:2]
            context = PipelineContext(image_path.stem, width, height,
                                      PipelineConfig(text="ppocrv5_mobile", component="contextual_values_v2"),
                                      ocr=None, detector=detector,
                                      resources={"component": {"cache_dir": str(args.cache)}})
            t0 = time.perf_counter()
            text = text_stage.run(image, context)
            frozen = baseline.run(image, text, context)
            baseline_seconds = time.perf_counter() - t0
            results = {"baseline": frozen}
            elapsed = {"baseline": baseline_seconds}
            if "local" in variants or "local_joint" in variants:
                t0 = time.perf_counter()
                refined = local.refine(image, frozen, context)
                elapsed["local"] = baseline_seconds + time.perf_counter() - t0
                if "local" in variants:
                    results["local"] = refined
                if "local_joint" in variants:
                    # Use exactly the same local OCR hypotheses for A and B.
                    t0 = time.perf_counter()
                    local_texts, local_diagnostics = local.reader.recognize(image, frozen)
                    combined = rebuild_fields(frozen, local_texts, joint=True)
                    combined.diagnostics["component_local_ocr"] = local_diagnostics
                    combined.diagnostics["frontend"] = "component_local_joint_v1"
                    results["local_joint"] = combined
                    elapsed["local_joint"] = elapsed["local"] + time.perf_counter() - t0
            if "joint_only" in variants:
                t0 = time.perf_counter()
                results["joint_only"] = printed_groups.refine(image, frozen, context)
                elapsed["joint_only"] = baseline_seconds + time.perf_counter() - t0
            if "value_groups" in variants:
                t0 = time.perf_counter()
                results["value_groups"] = value_groups.refine(image, frozen, context)
                elapsed["value_groups"] = baseline_seconds + time.perf_counter() - t0
            for variant, stage in newer_stages.items():
                if variant in variants:
                    t0 = time.perf_counter()
                    results[variant] = stage.refine(image, frozen, context)
                    elapsed[variant] = baseline_seconds + time.perf_counter() - t0
            for variant, result in results.items():
                if [(c.type, c.bbox) for c in result.components] != [(c.type, c.bbox) for c in frozen.components]:
                    raise AssertionError("Ablation changed symbol geometry or types")
                destination = output / cid / variant
                pred = destination / "components.json"
                write(pred, export_components(result, height))
                saved.append(pred)
                write(destination / "diagnostics.json", result.diagnostics)
            timings[cid] = elapsed
        progress = {"phase": "inference", "completed": position, "total": len(rows), "case": cid,
                    "wall_seconds": time.perf_counter() - start, "seconds": elapsed,
                    "GT_opens": guard.target_opens}
        write(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    if guard.target_opens:
        raise AssertionError("GT accessed before prediction sealing")
    sealed = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in saved}
    write(output / "predictions_sealed_before_GT.json", {"sha256": sealed, "GT_opens": 0})
    write(output / "timings.json", timings)
    if args.evaluate:
        # Import only Component scoring primitives. Never call evaluate_case.
        from evaluate_v2 import component_counts, component_key_map, prf
        cases = []
        for cid, _, folder in rows:
            targets = list(folder.glob("*_target*.json"))
            if len(targets) != 1 or targets[0].resolve().parent != folder.resolve():
                raise ValueError(f"{cid}: expected exactly one development target")
            with guard.evaluating(targets[0], [output / cid / v / "components.json" for v in variants]):
                reference = json.loads(targets[0].read_text(encoding="utf-8"))["components"]
            scores = {}
            for variant in variants:
                predpath = output / cid / variant / "components.json"
                if hashlib.sha256(predpath.read_bytes()).hexdigest() != sealed[str(predpath.relative_to(output))]:
                    raise AssertionError("Prediction was modified after sealing")
                predicted = json.loads(predpath.read_text(encoding="utf-8"))
                scores[variant] = prf(*component_counts(predicted, reference, component_key_map(predicted, reference)))
            cases.append({"case_id": cid, "scores": scores})
        aggregate = {}
        for variant in variants:
            metrics = [r["scores"][variant] for r in cases]
            base = [r["scores"]["baseline"]["f1"] for r in cases]
            aggregate[variant] = {
                "macro_ComponentF1": statistics.mean(m["f1"] for m in metrics),
                "micro_Component": prf(*(sum(m[key] for m in metrics) for key in ("tp", "pred", "gt"))),
                "improved_cases": sum(m["f1"] > b for m, b in zip(metrics, base)),
                "regressed_cases": sum(m["f1"] < b for m, b in zip(metrics, base)),
                "mean_seconds_including_cached_baseline": statistics.mean(timings[cid][variant] for cid in ids)}
            # Reuse the existing development partition (120/30, whole pages).
            # Both groups are inside the authorized 0001..0150 whitelist.
            for partition, dev in (("train120", False), ("check30", True)):
                subset = [r["scores"][variant] for r in cases if (int(r["case_id"]) % 5 == 0) == dev]
                if subset:
                    aggregate[variant][partition] = {
                        "cases": len(subset), "macro_ComponentF1": statistics.mean(m["f1"] for m in subset),
                        "micro_Component": prf(*(sum(m[key] for m in subset) for key in ("tp", "pred", "gt")))}
        report = {**manifest, "aggregate": aggregate, "cases": cases, "GT_opens": guard.target_opens,
                  "elapsed_seconds": time.perf_counter() - start,
                  "scoring_rule": "evaluate_v2 Component only: key/type/bbox<=20; box Name; r/c value; GND spatial"}
        write(output / "report.json", report)
        print(json.dumps({"aggregate": aggregate, "report": str(output / "report.json")}), flush=True)
    write(output / "progress.json", {"phase": "finished", "completed": len(rows), "total": len(rows),
                                      "ACTUAL_CASE_IDS": ids, "GT_opens": guard.target_opens})


if __name__ == "__main__":
    main()
