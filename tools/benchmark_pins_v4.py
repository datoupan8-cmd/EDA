"""Paired pin V3/V4 development benchmark with frozen image-only frontends.

Only 0001..0150 are permitted. Reuse signed component/OCR predictions from a
completed nets benchmark; never feed targets into inference. Strict scoring,
identity-normalized scoring, and geometry-only diagnostics remain separate.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pin_frontend_cache import (
    build_frontend, canonical, development_manifest, digest_value,
    parse_case_ids, reconstruct_scene, sha256_file, verify_scene_checksum,
)

VIEWS = ("strict", "identity_normalized", "geometry_connectivity", "retained_geometry_connectivity")


def frozen_component_fields(scene):
    return {"width": scene.width, "height": scene.height,
            "components": [{key: value for key, value in asdict(component).items() if key != "pins"}
                           for component in scene.components],
            "texts": [asdict(text) for text in scene.texts]}


def restore_roles(scene):
    from pcb.schema import Text
    from pcb.text_detection import TokenRole
    rows = scene.diagnostics.get("text_roles")
    if rows is None:
        raise ValueError("Frozen frontend has no text role diagnostics")
    return [TokenRole(index=row["index"], token=Text(row["text"], tuple(row["bbox"])),
                      role=row["role"], normalized=row["normalized"],
                      confidence=row["confidence"], reason=row.get("reason", "")) for row in rows]


def verify_reference(reference, required_ids=None, allow_preparing=False):
    ids = [f"{number:04d}" for number in range(1, 151)]
    if not allow_preparing and (reference.get("paired_case_ids") != ids or reference.get("success_count") != 150):
        raise ValueError("Reference must be a complete successful 0001..0150 frontend preparation")
    if allow_preparing and reference.get("case_ids") != ids:
        raise ValueError("Preparing reference must declare the complete development whitelist")
    if reference.get("holdout_used") is not False:
        raise ValueError("Reference must explicitly exclude holdout data")
    rows = reference.get("cases", [])
    row_ids = [row.get("case_id") for row in rows]
    if (not allow_preparing and row_ids != ids) or (allow_preparing and (
        len(set(row_ids)) != len(row_ids) or any(case_id not in ids for case_id in row_ids)
    )):
        raise ValueError("Reference case IDs are incomplete, duplicated, or out of order")
    indexed = {row["case_id"]: row for row in rows}
    for case_id in required_ids or ids:
        if case_id not in indexed or indexed[case_id].get("status") != "ok":
            raise ValueError(f"Requested frontend is not ready: {case_id}")
    signature = reference["frontend_signature"]
    for relative, expected in signature["source_sha256"].items():
        if relative == "benchmark_nets.build_frontend":
            actual = hashlib.sha256(inspect.getsource(build_frontend).encode("utf-8")).hexdigest()
        else:
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Reference source path escapes project")
            actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen frontend source changed: {relative}")
    for relative, expected in reference["nets_source_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path) != expected:
            raise ValueError(f"Frozen net stage source changed: {relative}")
    return indexed


def load_frozen(row, image_path, reference):
    path = Path(row["frontend_cache"]).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Frontend cache must be inside the project's runs directory")
    cache = json.loads(path.read_text(encoding="utf-8"))
    expected = {**reference["frontend_signature"], "image_sha256": sha256_file(image_path),
                "image_name": image_path.name}
    if cache["signature"] != expected or cache["signature_sha256"] != digest_value(expected):
        raise ValueError("Frozen frontend image/source signature mismatch")
    if row["frontend_signature_sha256"] != cache["signature_sha256"]:
        raise ValueError("Reference/cache signature mismatch")
    checksum_mode = verify_scene_checksum(cache)
    return reconstruct_scene(cache["scene"]), checksum_mode


def run_pin_stages(image, image_path, frontend, config, registry):
    from pcb.core.context import PipelineContext
    from pcb.core.interfaces import ComponentStageOutput
    scene = copy.deepcopy(frontend)
    scene.nets = []
    before = canonical(frozen_component_fields(scene))
    for component in scene.components:
        component.pins = []
    output = ComponentStageOutput(scene.components, scene.texts, restore_roles(scene), scene.diagnostics)
    context = PipelineContext(image_path.name, scene.width, scene.height, config, None)
    localization = registry.create("pin_localization", config.pin_localization).run(image, output, context)
    semantics = registry.create("pin_semantics", config.pin_semantics).run(output, localization, context)
    scene.components = semantics.components
    scene.diagnostics.update(localization.diagnostics)
    scene.diagnostics.update(semantics.diagnostics)
    if canonical(frozen_component_fields(scene)) != before:
        raise AssertionError("Pin stage changed component or OCR fields")
    return scene


def internal_copy(scene):
    """Make unknown pins uniquely addressable for geometry-only diagnostics.

    Never modify the official scene or claim temporary IDs were recognized.
    Renaming precedes network reconstruction, so duplicate unknown numbers
    cannot accidentally short two physically separate pins.
    """
    result = copy.deepcopy(scene)
    result.nets = []
    audit = []
    for component in result.components:
        used = {pin.number.casefold() for pin in component.pins if pin.exportable}
        for index, pin in enumerate(component.pins, 1):
            original = pin.number
            unknown = not pin.exportable
            if unknown:
                number = f"__TMP_{index}"
                while number.casefold() in used:
                    number += "_"
                pin.number = number
                pin.exportable = True
                used.add(number.casefold())
            audit.append({"component": component.key, "internal_number": pin.number,
                          "original_number": original, "recognized_number": not unknown,
                          "number_source": pin.number_source, "observable_number": pin.observable_number,
                          "tip": pin.tip, "coordinate_system": "original_image_top_left"})
    return result, audit


def run_network(image, image_path, scene, config, registry):
    from pcb.core.context import PipelineContext
    from pcb.submission import export
    before = canonical(frozen_component_fields(scene))
    pins_before = canonical([[asdict(pin) for pin in component.pins] for component in scene.components])
    context = PipelineContext(image_path.name, scene.width, scene.height, config, None)
    registry.create("topology", config.topology).run(
        scene, registry.create("wire", config.wire).run(image, scene, context), context)
    if before != canonical(frozen_component_fields(scene)) or pins_before != canonical(
        [[asdict(pin) for pin in component.pins] for component in scene.components]
    ):
        raise AssertionError("Frozen network stages mutated their upstream inputs")
    return export(scene)


def infer_arm(image, image_path, scene, config, registry, output_dir):
    from pcb.io import write_json
    started = time.perf_counter()
    internal, audit = internal_copy(scene)
    result = run_network(image, image_path, scene, config, registry)
    retained = run_network(image, image_path, internal, config, registry)
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "diagnostics.json", scene.diagnostics)
    write_json(output_dir / "scene.json", asdict(scene))
    write_json(output_dir / "internal_connectivity.json", retained)
    write_json(output_dir / "pin_audit.json", {"OFFICIAL_SUBMISSION": False,
               "description": "Unknown pins retained only for geometry diagnostics; not recognized IDs",
               "pins": audit})
    return {"status": "ok", "result_path": str((output_dir / "result.json").resolve()),
            "diagnostics_path": str((output_dir / "diagnostics.json").resolve()),
            "internal_result_path": str((output_dir / "internal_connectivity.json").resolve()),
            "pin_candidate_count": sum(len(component.pins) for component in scene.components),
            "exportable_pin_count": sum(pin.exportable for component in scene.components for pin in component.pins),
            "seconds": time.perf_counter() - started}


def evaluate(records, manifest):
    """Targets are read here only, after every image-only inference has ended."""
    import evaluate_v2 as strict
    import evaluate_diagnostic as diagnostic
    groups = {arm: {view: [] for view in VIEWS} for arm in ("baseline", "candidate")}
    for record in records:
        if record["status"] != "ok":
            continue
        case_id = record["case_id"]
        image_path, target_path = manifest[case_id]
        try:
            target = json.loads(target_path.read_text(encoding="utf-8"))
            pending = {}
            for arm in groups:
                bundle = record[arm]
                prediction = json.loads(Path(bundle["result_path"]).read_text(encoding="utf-8"))
                diagnostics = json.loads(Path(bundle["diagnostics_path"]).read_text(encoding="utf-8"))
                retained = json.loads(Path(bundle["internal_result_path"]).read_text(encoding="utf-8"))
                normalized = diagnostic.evaluate_case(prediction, target, diagnostics)
                internal_metrics = diagnostic.evaluate_case(retained, target, diagnostics)["geometry_connectivity"]
                views = {"strict": strict.evaluate_case(prediction, target, diagnostics),
                         "identity_normalized": normalized["identity_normalized"],
                         "geometry_connectivity": normalized["geometry_connectivity"],
                         "retained_geometry_connectivity": internal_metrics}
                bundle["evaluation"] = views
                bundle["error_breakdown"] = normalized["error_breakdown"]
                bundle["matching"] = normalized["matching"]
                pending[arm] = views
            for arm, views in pending.items():
                for view, metrics in views.items():
                    groups[arm][view].append({"case_id": case_id, "status": "ok",
                        "source": strict.source_from_name(image_path.name), "evaluation": metrics})
        except Exception as exc:
            record["evaluation_error"] = repr(exc)
            record["status"] = "failed"
    aggregates = {arm: {view: strict.aggregate(rows) for view, rows in views.items()}
                  for arm, views in groups.items()}
    changes = {}
    if groups["baseline"]["strict"]:
        for view in VIEWS:
            changes[view] = {}
            for metric in ("Component", "Pin", "NetHypergraph", "NetLine", "PinPair"):
                before = aggregates["baseline"][view]["overall"]["metrics"][metric]
                after = aggregates["candidate"][view]["overall"]["metrics"][metric]
                changes[view][metric] = {field: 100 * (after[field] - before[field])
                                        for field in ("precision", "recall", "f1", "macro_f1")}
    return aggregates, changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, default=Path("runs/pins_v3_prepare/report.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-preparing", action="store_true",
                        help="Use completed requested cases from a concurrently written progress.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--ocr-device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--case-ids")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pins_v4_full"))
    parser.add_argument("--candidate-config", type=Path, default=Path("configs/examples/pins_v4.json"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    ids = parse_case_ids(args.case_ids)
    manifest = development_manifest(args.data_root)
    if args.threads < 1:
        raise ValueError("Threads must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new output directory; prior evidence is never overwritten")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    import main as project_entry  # noqa: F401; reuse the application's vendor setup
    import cv2
    from pcb.core.config import PipelineConfig
    from pcb.core.registry import build_default_registry
    from pcb.io import read_image, write_json
    cv2.setNumThreads(args.threads)
    baseline = PipelineConfig.load(ROOT / "configs/current.json")
    registry = build_default_registry()
    if args.prepare_only:
        from tools.pin_frontend_cache import prepare
        return prepare(args, manifest, baseline, registry)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    indexed = verify_reference(reference, ids, args.allow_preparing)
    if Path(reference["dataset_root"]).resolve() != args.data_root.resolve():
        raise ValueError("Reference and requested dataset roots differ")
    candidate = PipelineConfig.load(args.candidate_config)
    if (baseline.pin_localization, baseline.pin_semantics) != ("v3", "v3"):
        raise ValueError("Baseline must use original V3 pins")
    base_fields = {key: val for key, val in baseline.as_dict().items() if key not in {"pin_localization", "pin_semantics"}}
    candidate_fields = {key: val for key, val in candidate.as_dict().items() if key not in {"pin_localization", "pin_semantics"}}
    if base_fields != candidate_fields:
        raise ValueError("Candidate may change only the two pin stages")
    registry = build_default_registry()
    sources = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted((ROOT / "pcb").rglob("*.py"))}
    sources.update({name: sha256_file(ROOT / name) for name in
                    ("evaluate_v2.py", "evaluate_diagnostic.py", "tools/benchmark_pins_v4.py")})
    meta = {"actual_case_ids": [], "case_ids": ids, "holdout_used": False, "OFFICIAL_SCORE": False,
            "reference_report": str(args.reference_report.resolve()), "dataset_root": str(args.data_root.resolve()),
            "baseline_config": baseline.as_dict(), "candidate_config": candidate.as_dict(),
            "source_sha256": sources, "metric_deltas_unit": "percentage_points",
            "diagnostic_notice": "Geometry-only and retained-pin views are not official identity accuracy"}
    write_json(args.output_dir / "manifest.json", meta)
    print(canonical({"event": "start", "case_ids": ids, "holdout_used": False}), flush=True)
    records = []
    for index, case_id in enumerate(ids, 1):
        image_path, _ = manifest[case_id]
        row = {"case_id": case_id, "status": "failed"}
        meta["actual_case_ids"].append(case_id)
        print(canonical({"event": "case_start", "case_id": case_id, "index": index, "total": len(ids)}), flush=True)
        try:
            image = read_image(image_path)
            frontend, checksum_mode = load_frozen(indexed[case_id], image_path, reference)
            row["cache_checksum_verification"] = checksum_mode
            base_scene = copy.deepcopy(frontend)
            cand_scene = run_pin_stages(image, image_path, frontend, candidate, registry)
            if canonical(frozen_component_fields(base_scene)) != canonical(frozen_component_fields(cand_scene)):
                raise AssertionError("Paired components or OCR changed")
            row["baseline"] = infer_arm(image, image_path, base_scene, baseline, registry, args.output_dir / "baseline" / case_id)
            row["candidate"] = infer_arm(image, image_path, cand_scene, candidate, registry, args.output_dir / "candidate" / case_id)
            original_path = Path(indexed[case_id]["baseline"]["result_path"])
            previous = json.loads(original_path.read_text(encoding="utf-8"))
            repeated = json.loads(Path(row["baseline"]["result_path"]).read_text(encoding="utf-8"))
            if canonical(previous) != canonical(repeated):
                raise AssertionError("V3-pin baseline differs from the frozen current-branch run")
            row.update(status="ok", components_and_ocr_unchanged=True, baseline_matches_previous=True)
        except Exception as exc:
            row.update(error=repr(exc), traceback=traceback.format_exc())
        records.append(row)
        write_json(args.output_dir / "progress.json", {**meta, "cases": records})
        print(canonical({"event": "case_complete", "case_id": case_id, "status": row["status"],
                         "error": row.get("error")}), flush=True)
    print(canonical({"event": "evaluation_start", "case_ids": ids}), flush=True)
    aggregate, delta = evaluate(records, manifest)
    report = {**meta, "cases": records, "aggregate": aggregate, "delta": delta,
              "success_count": sum(row["status"] == "ok" for row in records),
              "failure_count": sum(row["status"] != "ok" for row in records),
              "all_components_and_ocr_unchanged": all(row.get("components_and_ocr_unchanged", False) for row in records)}
    write_json(args.output_dir / "manifest.json", meta)
    write_json(args.output_dir / "report.json", report)
    print(canonical({"event": "complete", "success_count": report["success_count"],
                     "failure_count": report["failure_count"], "report": str((args.output_dir / "report.json").resolve()),
                     "delta": delta}), flush=True)
    return int(report["failure_count"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
