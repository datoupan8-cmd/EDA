"""Portable team entry: image-only inference, then optional independent scoring."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcb.runtime_paths import prepend_compatible_vendor_paths
prepend_compatible_vendor_paths(ROOT)
from pcb.core.team_config import load_team_config
from pcb.core.registry import build_default_registry


def parse_ids(value: str | None) -> list[int]:
    if value is None:
        return list(range(1, 151))
    pieces = value.split(",")
    if not pieces or any(not x.strip().isdigit() for x in pieces):
        raise ValueError("--cases requires comma-separated case numbers")
    ids = [int(x.strip()) for x in pieces]
    if len(set(ids)) != len(ids) or any(not 1 <= x <= 150 for x in ids):
        raise PermissionError("Only unique development cases 0001..0150")
    return ids


def assert_development_path(path: Path, *, allow_dataset_root: bool = False):
    parts = [x.lower() for x in path.resolve().parts]
    if any(x in {"golden", "sealed", "eda_pin_crossing_quicktest"}
           or x.startswith(("10gt", "10_gtcase")) for x in parts):
        raise PermissionError("Sealed/Golden/QuickTest input prohibited")
    if "200_train_cases" in parts:
        n = parts.index("200_train_cases")
        if allow_dataset_root and n + 1 == len(parts):
            return
        if n + 1 >= len(parts) or not parts[n+1].isdigit() or not 1 <= int(parts[n+1]) <= 150:
            raise PermissionError("Only development images 0001..0150")


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def build_runtime(config, device, ocr_device, cache: Path):
    """Generic runtime setup. Models/algorithm decisions belong to stages."""
    import torch
    from pcb.component.detector_yolo import YoloComponentDetector
    from pcb.text.ocr_backends import HybridOCR, TiledEasyOCR
    from pcb.vision import OCR
    from pcb.core.pipeline import ModularPipeline
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ocr_device == "auto":
        ocr_device = device
    tiled = TiledEasyOCR(cache / "ocr", ROOT / "models/easyocr", device=ocr_device, allow_download=False)
    ocr = HybridOCR(tiled, OCR(cache / "ocr"))
    detector = YoloComponentDetector(ROOT / "models/component_yolo11n_continue_v2_best.pt",
                                    device="0" if device == "cuda" else device,
                                    cache_dir=cache / "yolo")
    return ModularPipeline(config, build_default_registry()), ocr, detector, device


def select_inputs(image: Path | None, dataset: Path | None, cases: str | None):
    if image is not None:
        path = image.resolve()
        assert_development_path(path)
        if path.suffix.lower() != ".png" or not path.is_file():
            raise ValueError("--image must be an existing PNG")
        return [(path.stem, path, None)]
    root = dataset.resolve()
    assert_development_path(root, allow_dataset_root=True)
    if root.name != "200_train_cases":
        raise ValueError("--dataset must name the 200_train_cases directory")
    rows = []
    for number in parse_ids(cases):
        cid = f"{number:04d}"
        folder = root / cid
        if folder.resolve().parent != root:
            raise PermissionError("Development folder redirects outside dataset")
        images = sorted(folder.glob("*.png"))
        if len(images) != 1:
            raise ValueError(f"{cid}: require exactly one PNG")
        if images[0].resolve().parent != folder.resolve():
            raise PermissionError("Image redirects outside allowed case")
        # Target discovery is deferred until inference has finished.
        rows.append((cid, images[0], folder))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--dataset", type=Path)
    parser.add_argument("--cases", help="e.g. 0001,0014,0087; dataset default is all 150")
    parser.add_argument("--config", default="configs/team.json")
    parser.add_argument("--output", default="runs/team_latest")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--ocr-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cache-dir", default="runs/team_cache")
    parser.add_argument("--evaluate", action="store_true", help="Read development GT only AFTER inference")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--list-stages", action="store_true")
    args = parser.parse_args(argv)
    config = load_team_config(args.config)
    if args.list_stages:
        print(json.dumps({"config": config.as_dict(), "stages": build_default_registry().describe()}, indent=2))
        return 0
    if args.image is None and args.dataset is None:
        parser.error("Provide --image or --dataset")
    if args.evaluate and args.dataset is None:
        parser.error("--evaluate requires --dataset")
    if args.cases and args.dataset is None:
        parser.error("--cases requires --dataset")
    rows = select_inputs(args.image, args.dataset, args.cases)
    output, cache = project_path(args.output), project_path(args.cache_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new --output directory; previous evidence is not overwritten")
    from pcb.core.context import PipelineContext
    from pcb.io import read_image, write_json, write_image
    from pcb.submission import validate_strict
    # Shared guard is reused exactly, without importing experiment algorithms.
    from tools.team_access_guard import Guard
    guard = Guard()
    pipeline, ocr, detector, device = build_runtime(config, args.device, args.ocr_device, cache)
    records = []
    start = time.perf_counter()
    print(json.dumps({"event": "start", "device": device, "config": config.as_dict(),
                      "case_count": len(rows), "OFFICIAL_SCORE": False}, ensure_ascii=False), flush=True)
    for cid, image_path, _ in rows:
        t0 = time.perf_counter()
        with guard.inferring():
            image = read_image(image_path)
            h, w = image.shape[:2]
            context = PipelineContext(image_path.name, w, h, config, ocr, detector,
                resources={"pin": {"device": device, "words_cache": cache / "pin_words"}})
            artifacts = pipeline.run(image, context)
            validate_strict(artifacts.result, (w, h))
            folder = output / cid
            write_json(folder / "result.json", artifacts.result)
            artifacts.scene.diagnostics["seconds"] = time.perf_counter() - t0
            write_json(folder / "diagnostics.json", artifacts.scene.diagnostics)
            if args.save_debug:
                write_json(folder / "scene.json", asdict(artifacts.scene))
                write_image(folder / "wire_mask.png", artifacts.wire.mask)
        records.append({"case_id": cid, "status": "ok", "seconds": time.perf_counter()-t0,
                        "result_path": str((folder / 'result.json').resolve())})
        print(json.dumps({"event": "case_complete", "case_id": cid,
                          "seconds": round(records[-1]["seconds"], 2)}, ensure_ascii=False), flush=True)
        write_json(output / "progress.json", {"cases": records, "sealed_holdout_used": False})
    report = {"OFFICIAL_SCORE": False, "sealed_holdout_used": False,
              "config": config.as_dict(), "device": device, "success_count": len(records),
              "failure_count": 0, "seconds": time.perf_counter()-start, "cases": records}
    if args.evaluate:
        from evaluate_v2 import evaluate_case, aggregate, source_from_name
        for record, (_, image_path, folder) in zip(records, rows):
            targets = sorted(folder.glob("*_target*.json"))
            if len(targets) != 1 or targets[0].resolve().parent != folder.resolve():
                raise ValueError(f"{record['case_id']}: require exactly one safe target")
            saved = [Path(record['result_path'])]
            with guard.evaluating(targets[0], saved):
                target = json.loads(targets[0].read_text(encoding="utf-8"))
                pred = json.loads(saved[0].read_text(encoding="utf-8"))
                diagnostics = json.loads(saved[0].with_name("diagnostics.json").read_text(encoding="utf-8"))
                record["evaluation"] = evaluate_case(pred, target, diagnostics)
                record["source"] = source_from_name(image_path.name)
        report["aggregate"] = aggregate(records)
    report["target_opens"] = guard.target_opens
    write_json(output / "report.json", report)
    print(json.dumps({"event": "complete", "report": str(output / 'report.json'),
                      "overall": report.get('aggregate', {}).get('overall')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
