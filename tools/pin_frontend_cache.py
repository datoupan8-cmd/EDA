"""Signed image-only component/OCR/pin-V3 snapshots, restricted to 0001..0150."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical(value):
    def encode(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(type(item).__name__)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), default=encode)


def digest_value(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case_ids(value):
    if value is None:
        return [f"{n:04d}" for n in range(1, 151)]
    parts = value.split(",")
    if not parts or any(not part.strip().isdigit() for part in parts):
        raise ValueError("Case IDs must be integers")
    numbers = [int(part.strip()) for part in parts]
    if any(n < 1 or n > 150 for n in numbers):
        raise ValueError("Holdout guard: only 0001..0150 are allowed")
    if len(set(numbers)) != len(numbers):
        raise ValueError("Duplicate case IDs")
    return [f"{n:04d}" for n in numbers]


def development_manifest(root):
    root = Path(root).resolve(strict=True)
    if root.name != "200_train_cases":
        raise ValueError("Dataset root must be 200_train_cases")
    result = {}
    for case_id in parse_case_ids(None):
        folder = root / case_id
        resolved = folder.resolve(strict=True)
        if not folder.is_dir() or resolved.parent != root or resolved.name != case_id:
            raise ValueError(f"Unsafe whitelist directory: {case_id}")
        images = list(folder.glob("*.png"))
        targets = list(folder.glob("*_target*.json"))
        if len(images) != 1 or len(targets) != 1:
            raise ValueError(f"{case_id}: require one image and target; found {len(images)}, {len(targets)}")
        for path in images + targets:
            if path.resolve(strict=True).parent != resolved:
                raise ValueError("File redirects outside whitelist")
        result[case_id] = images[0], targets[0]
    return result


def reconstruct_scene(payload):
    from pcb.schema import Component, Pin, Scene, Text
    if payload.get("nets"):
        raise ValueError("Frontend cache must not contain nets")
    components = []
    for raw in payload["components"]:
        row = dict(raw)
        row["bbox"] = tuple(row["bbox"])
        if row.get("body_bbox") is not None:
            row["body_bbox"] = tuple(row["body_bbox"])
        pins = []
        for raw_pin in row.pop("pins"):
            pin = dict(raw_pin)
            pin["tip"] = tuple(pin["tip"])
            if pin.get("base") is not None:
                pin["base"] = tuple(pin["base"])
            pins.append(Pin(**pin))
        components.append(Component(**row, pins=pins))
    texts = [Text(**{**row, "bbox": tuple(row["bbox"])}) for row in payload["texts"]]
    return Scene(payload["width"], payload["height"], components, texts,
                 diagnostics=payload.get("diagnostics", {}))


def verify_scene_checksum(cache):
    if digest_value(cache["scene"]) != cache.get("scene_sha256"):
        raise ValueError("Frontend cache checksum mismatch")
    return "json_canonical"


def build_frontend(image, image_path, config, registry, ocr, detector):
    from pcb.core.context import PipelineContext
    from pcb.schema import Scene
    height, width = image.shape[:2]
    context = PipelineContext(image_path.name, width, height, config, ocr, detector)
    text = registry.create("text", config.text).run(image, context)
    component = registry.create("component", config.component).run(image, text, context)
    localization = registry.create("pin_localization", "v3").run(image, component, context)
    semantics = registry.create("pin_semantics", "v3").run(component, localization, context)
    scene = Scene(width, height, semantics.components, component.texts, diagnostics=component.diagnostics)
    scene.diagnostics.update(semantics.diagnostics)
    if getattr(ocr, "last_diagnostics", None):
        scene.diagnostics["ocr"] = dict(ocr.last_diagnostics)
    return scene


def signature(config, args):
    from importlib import metadata, util
    excluded = {"pcb/core/registry.py", "pcb/core/pipeline.py", "pcb/pin_detection_v4.py", "pcb/pin_semantics_v4.py",
                "pcb/pin/localization/v4.py", "pcb/pin/semantics/v4.py"}
    sources = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted((ROOT / "pcb").rglob("*.py"))
               if p.relative_to(ROOT).as_posix() not in excluded
               and not p.relative_to(ROOT).as_posix().startswith(("pcb/wire", "pcb/topology"))}
    sources["main.py"] = sha256_file(ROOT / "main.py")
    sources["benchmark_nets.build_frontend"] = hashlib.sha256(inspect.getsource(build_frontend).encode("utf-8")).hexdigest()
    spec = util.find_spec("rapidocr_onnxruntime")
    if not spec or not spec.origin:
        raise ImportError("RapidOCR runtime unavailable")
    models = [ROOT / "models/component_yolo11n_continue_v2_best.pt"]
    models += sorted((ROOT / "models/easyocr").glob("*.pth"))
    models += sorted(Path(spec.origin).parent.rglob("*.onnx"))
    if len(models) < 4:
        raise ValueError("Required local OCR/detector weights missing; downloads are disabled")
    return {"cache_version": 2, "mode": "image-only", "upstream_config": config.as_dict(),
            "source_sha256": sources, "model_sha256": {str(p.resolve()): sha256_file(p) for p in models},
            "package_versions": {p: metadata.version(p) for p in ("torch", "numpy", "opencv-python", "easyocr", "onnxruntime")},
            "python": sys.version, "device": args.device, "ocr_device": args.ocr_device,
            "threads": args.threads, "seed": 0}


def prepare(args, manifest, config, registry):
    """Prepare caches and an explicit current-branch baseline, without any GT input."""
    import copy
    import torch
    import numpy as np
    from pcb.core.context import PipelineContext
    from pcb.io import read_image, write_json
    from pcb.submission import export
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    common = signature(config, args)
    ids = parse_case_ids(args.case_ids)
    meta = {"case_ids": ids, "actual_case_ids": [], "holdout_used": False, "mode": "image-only",
            "dataset_root": str(args.data_root.resolve()), "frontend_signature": common,
            "nets_source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in sorted((ROOT / "pcb").rglob("*.py"))
                if p.relative_to(ROOT).as_posix().startswith(("pcb/wire", "pcb/topology"))}}
    records = []
    ocr = detector = None
    print(canonical({"event": "prepare_start", "actual_case_ids": ids, "holdout_used": False}), flush=True)
    for index, case_id in enumerate(ids, 1):
        image_path, _ = manifest[case_id]
        row = {"case_id": case_id, "status": "failed"}
        meta["actual_case_ids"].append(case_id)
        started = time.perf_counter()
        print(canonical({"event": "case_start", "case_id": case_id, "index": index, "total": len(ids)}), flush=True)
        try:
            image = read_image(image_path)
            sign = {**common, "image_sha256": sha256_file(image_path), "image_name": image_path.name}
            digest = digest_value(sign)
            cache_path = ROOT / "runs/pin_frontend" / case_id / f"{digest[:24]}.json"
            if cache_path.exists():
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                if cache.get("signature") != sign or cache.get("signature_sha256") != digest:
                    raise ValueError("Frontend cache signature mismatch")
                verify_scene_checksum(cache)
                scene = reconstruct_scene(cache["scene"])
                row["cache_hit"] = True
            else:
                if ocr is None:
                    from pcb.vision import OCR
                    from pcb.ocr_backends import HybridOCR, TiledEasyOCR
                    from pcb.component_detector_yolo import YoloComponentDetector
                    tiled = TiledEasyOCR(ROOT / "runs/ocr_cache_v4_1", ROOT / "models/easyocr",
                                         device=args.ocr_device, allow_download=False)
                    ocr = HybridOCR(tiled, OCR(ROOT / "runs/ocr_cache_v4_1"))
                    detector = YoloComponentDetector(ROOT / "models/component_yolo11n_continue_v2_best.pt", args.device,
                                                     cache_dir=ROOT / "runs/yolo_cache")
                scene = build_frontend(image, image_path, config, registry, ocr, detector)
                payload = json.loads(canonical(asdict(scene)))
                write_json(cache_path, {"signature": sign, "signature_sha256": digest,
                                       "scene": payload, "scene_sha256": digest_value(payload)})
                row["cache_hit"] = False
            base = copy.deepcopy(scene)
            context = PipelineContext(image_path.name, base.width, base.height, config, None)
            wire = registry.create("wire", config.wire).run(image, base, context)
            registry.create("topology", config.topology).run(base, wire, context)
            destination = args.output_dir / "baseline" / case_id
            write_json(destination / "result.json", export(base))
            write_json(destination / "diagnostics.json", base.diagnostics)
            row.update(status="ok", frontend_cache=str(cache_path.resolve()), frontend_signature_sha256=digest,
                       baseline={"result_path": str((destination / "result.json").resolve())})
        except Exception as exc:
            import traceback
            row.update(error=repr(exc), traceback=traceback.format_exc())
        row["seconds"] = time.perf_counter() - started
        records.append(row)
        write_json(args.output_dir / "progress.json", {**meta, "cases": records})
        print(canonical({"event": "case_complete", "case_id": case_id, "status": row["status"],
                         "seconds": round(row["seconds"], 2), "error": row.get("error")}), flush=True)
    report = {**meta, "cases": records, "success_count": sum(row["status"] == "ok" for row in records),
              "paired_case_ids": ids if all(row["status"] == "ok" for row in records) else []}
    write_json(args.output_dir / "report.json", report)
    print(canonical({"event": "prepare_complete", "success_count": report["success_count"],
                     "report": str((args.output_dir / "report.json").resolve())}), flush=True)
    return int(report["success_count"] != len(ids))
