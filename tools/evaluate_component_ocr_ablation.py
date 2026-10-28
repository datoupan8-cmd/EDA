"""Component-only 2x2 OCR experiment, fenced to public cases 0001..0150.

Each normalization runs the network once and generates paired greedy/beam OCR.
All four predictions are sealed before any GT is opened. No Pin/Net stages.
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
from tools.evaluate_component_local_joint import select_cases, write, export_components

VARIANTS = {"baseline": ("legacy", "greedy"), "normalization": ("v5", "greedy"),
            "beam": ("legacy", "beam"), "combined": ("v5", "beam")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cases", help="Default: all 150 allowed development cases")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/component_ocr_ablation_cache")
    parser.add_argument("--yolo-cache", type=Path, default=ROOT / "runs/team_cache/yolo")
    parser.add_argument("--ocr-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-runtime-root", type=Path, help="Optional existing NVIDIA runtime root; no installation")
    parser.add_argument("--cudnn-dir", type=Path, help="Optional existing cuDNN DLL directory on Windows")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--baseline-parity", action="store_true", help="Compare greedy legacy OCR to current unmodified frontend")
    args = parser.parse_args()
    from tools.team_access_guard import Guard
    guard = Guard()
    rows = select_cases(args.dataset, args.cases)
    ids = [cid for cid, _, _ in rows]
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "pcb/component/ocr_ablation.py", ROOT / "pcb/component/alternative_ocr.py",
               ROOT / "pcb/component/value_groups_v2.py", ROOT / "evaluate_v2.py"]
    manifest = {"ACTUAL_CASE_IDS": ids, "scope": "Component_only", "OFFICIAL_SCORE": False,
                "sealed_holdout_used": False, "GT_used_in_inference": False,
                "variants": VARIANTS, "ocr_device": args.ocr_device,
                "config_before": json.loads((ROOT / "configs/component/current.json").read_text(encoding="utf-8")),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    write(args.output / "manifest.json", manifest)
    print(json.dumps({"ACTUAL_CASE_IDS": ids, "ocr_device": args.ocr_device}), flush=True)
    from pcb.runtime_paths import prepend_compatible_vendor_paths
    prepend_compatible_vendor_paths(ROOT)
    import torch  # Load CUDA libraries before ORT if the CUDA provider is requested.
    dll_handles = []
    if args.ocr_device == "cuda":
        import os
        import onnxruntime as ort
        if args.cuda_runtime_root:
            dll_handles.extend(os.add_dll_directory(str(p.resolve())) for p in args.cuda_runtime_root.glob("*/bin"))
        if args.cudnn_dir:
            dll_handles.append(os.add_dll_directory(str(args.cudnn_dir.resolve())))
        ort.preload_dlls(cuda=True, cudnn=False, directory="")
        ort.preload_dlls(cuda=False, cudnn=True, directory=str(args.cudnn_dir) if args.cudnn_dir else None)
    from pcb.component.ocr_ablation import PairedPPOcrV5Backend
    from pcb.component.alternative_ocr import PPOcrV5Backend
    from pcb.component.detector_yolo import YoloComponentDetector
    from pcb.component.value_groups_v2 import ComponentStageOrderedValues
    from pcb.core.config import PipelineConfig
    from pcb.core.context import PipelineContext
    from pcb.core.interfaces import TextStageOutput
    from pcb.io import read_image
    detector = YoloComponentDetector(ROOT / "models/component_yolo11n_continue_v2_best.pt", cache_dir=args.yolo_cache)
    backends = {norm: PairedPPOcrV5Backend(args.cache, normalization=norm, use_cuda=args.ocr_device == "cuda")
                for norm in ("legacy", "v5")}
    original = PPOcrV5Backend(ROOT / "runs/component_alternative_cache/ocr_v5") if args.baseline_parity else None
    stage = ComponentStageOrderedValues()
    timings, parity, saved = {}, {}, []
    start = time.perf_counter()
    for position, (cid, image_path, _) in enumerate(rows, 1):
        with guard.inferring():
            image = read_image(image_path)
            height, width = image.shape[:2]
            context = PipelineContext(image_path.stem, width, height,
                                      PipelineConfig(text="ppocrv5_mobile", component="ordered_value_groups_v2"),
                                      ocr=None, detector=detector, resources={})
            text_pairs, details, elapsed = {}, {}, {}
            for norm, backend in backends.items():
                t0 = time.perf_counter()
                text_pairs[norm] = backend.recognize_pair(image)
                elapsed[norm + "_ocr_paired"] = time.perf_counter() - t0
                details[norm] = dict(backend.last_diagnostics)
            if original:
                expected = [asdict(t) for t in original.recognize(image)]
                actual = [asdict(t) for t in text_pairs["legacy"]["greedy"]]
                equal = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
                parity[cid] = {"equal": equal, "old_tokens": len(expected), "new_tokens": len(actual)}
                if args.ocr_device == "cpu" and not equal:
                    raise AssertionError("Paired greedy output differs from current CPU baseline: " + cid)
            for variant, (norm, decoder) in VARIANTS.items():
                t0 = time.perf_counter()
                result = stage.run(image, TextStageOutput(text_pairs[norm][decoder]), context)
                elapsed[variant + "_component"] = time.perf_counter() - t0
                destination = args.output / cid / variant
                predpath = destination / "components.json"
                write(predpath, export_components(result, height))
                saved.append(predpath)
                write(destination / "diagnostics.json", result.diagnostics)
            timings[cid] = elapsed
            write(args.output / cid / "ocr_diagnostics.json", details)
        progress = {"phase": "inference", "completed": position, "total": len(rows), "case": cid,
                    "wall_seconds": time.perf_counter() - start, "seconds": elapsed, "GT_opens": guard.target_opens,
                    "beam_changes": {n: len(d["beam_changes"]) for n, d in details.items()}}
        write(args.output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    if guard.target_opens:
        raise AssertionError("GT accessed before prediction sealing")
    sealed = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in saved}
    write(args.output / "predictions_sealed_before_GT.json", {"sha256": sealed, "GT_opens": 0})
    write(args.output / "timings.json", timings)
    write(args.output / "baseline_parity.json", parity)
    if args.evaluate:
        from evaluate_v2 import component_counts, component_key_map, prf
        cases = []
        for cid, _, folder in rows:
            targets = list(folder.glob("*_target*.json"))
            if len(targets) != 1 or targets[0].resolve().parent != folder:
                raise ValueError(cid + ": expected exactly one unredirected target")
            with guard.evaluating(targets[0], [args.output / cid / v / "components.json" for v in VARIANTS]):
                gt = json.loads(targets[0].read_text(encoding="utf-8"))["components"]
            scores = {}
            for variant in VARIANTS:
                predpath = args.output / cid / variant / "components.json"
                if hashlib.sha256(predpath.read_bytes()).hexdigest() != sealed[str(predpath.relative_to(args.output))]:
                    raise AssertionError("Prediction modified after sealing")
                pred = json.loads(predpath.read_text(encoding="utf-8"))
                scores[variant] = prf(*component_counts(pred, gt, component_key_map(pred, gt)))
            cases.append({"case_id": cid, "scores": scores})
        aggregate = {}
        for variant in VARIANTS:
            metrics = [r["scores"][variant] for r in cases]
            base = [r["scores"]["baseline"]["f1"] for r in cases]
            aggregate[variant] = {
                "macro_ComponentF1": statistics.mean(m["f1"] for m in metrics),
                "micro_Component": prf(*(sum(m[k] for m in metrics) for k in ("tp", "pred", "gt"))),
                "improved_cases": sum(m["f1"] > b for m, b in zip(metrics, base)),
                "regressed_cases": sum(m["f1"] < b for m, b in zip(metrics, base))}
            for name, check in (("train120", False), ("check30", True)):
                part = [r["scores"][variant] for r in cases if (int(r["case_id"]) % 5 == 0) == check]
                if part:
                    aggregate[variant][name] = {
                        "cases": len(part), "macro_ComponentF1": statistics.mean(m["f1"] for m in part),
                        "micro_Component": prf(*(sum(m[k] for m in part) for k in ("tp", "pred", "gt")))}
        report = {**manifest, "aggregate": aggregate, "cases": cases, "GT_opens": guard.target_opens,
                  "baseline_parity": parity, "elapsed_seconds": time.perf_counter() - start,
                  "scoring_rule": "Existing evaluate_v2 Component-only proxy, NOT official evaluator"}
        write(args.output / "report.json", report)
        print(json.dumps({"aggregate": aggregate, "report": str(args.output / "report.json")}), flush=True)
    write(args.output / "progress.json", {"phase": "finished", "ACTUAL_CASE_IDS": ids, "GT_opens": guard.target_opens})


if __name__ == "__main__":
    main()
