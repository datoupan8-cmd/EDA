"""Dataset-fenced, GT-free warm runtime comparison of greedy and beam OCR."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_component_local_joint import select_cases, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cases", default="1,2,3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization", choices=["legacy", "v5"], default="legacy")
    parser.add_argument("--ocr-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-runtime-root", type=Path)
    parser.add_argument("--cudnn-dir", type=Path)
    args = parser.parse_args()
    rows = select_cases(args.dataset, args.cases)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh benchmark directory")
    args.output.mkdir(parents=True, exist_ok=True)
    ids = [cid for cid, _, _ in rows]
    print(json.dumps({"ACTUAL_CASE_IDS": ids, "GT_used": False}), flush=True)
    from tools.team_access_guard import Guard
    guard = Guard()
    import torch
    import onnxruntime as ort
    handles = []
    if args.ocr_device == "cuda":
        if args.cuda_runtime_root:
            handles.extend(os.add_dll_directory(str(p.resolve())) for p in args.cuda_runtime_root.glob("*/bin"))
        if args.cudnn_dir:
            handles.append(os.add_dll_directory(str(args.cudnn_dir.resolve())))
        ort.preload_dlls(cuda=True, cudnn=False, directory="")
        ort.preload_dlls(cuda=False, cudnn=True, directory=str(args.cudnn_dir) if args.cudnn_dir else None)
    from pcb.component.ocr_ablation import PairedPPOcrV5Backend
    from pcb.io import read_image
    backend = PairedPPOcrV5Backend(args.output / "cache/warmup", normalization=args.normalization,
                                  use_cuda=args.ocr_device == "cuda")
    engine = backend._ensure_engine()
    greedy_decoder = backend.decoder.greedy
    results = []
    for position, (cid, image_path, _) in enumerate(rows):
        with guard.inferring():
            image = read_image(image_path)
            # Warm the exact image shapes using the same model session.
            backend.cache = (args.output / "cache/warmup").resolve()
            engine.text_rec.postprocess_op = greedy_decoder
            backend.recognize(image)
            backend.cache = (args.output / "cache/measured").resolve()
            elapsed, texts = {}, {}
            order = ("greedy", "beam") if position % 2 == 0 else ("beam", "greedy")
            for mode in order:
                engine.text_rec.postprocess_op = greedy_decoder if mode == "greedy" else backend.decoder
                t0 = time.perf_counter()
                value = backend.recognize(image) if mode == "greedy" else backend.recognize_pair(image)
                elapsed[mode] = time.perf_counter() - t0
                if backend.last_diagnostics["cache_hit"]:
                    raise AssertionError("A timed measurement consumed cached OCR")
                texts[mode] = value
            equal = json.dumps([asdict(t) for t in texts["greedy"]], sort_keys=True) == json.dumps(
                [asdict(t) for t in texts["beam"]["greedy"]], sort_keys=True)
            result = {"case": cid, "seconds": elapsed, "greedy_output_equal": equal,
                      "beam_over_greedy": elapsed["beam"] / elapsed["greedy"]}
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {"ACTUAL_CASE_IDS": ids, "GT_opens": guard.target_opens, "normalization": args.normalization,
              "ocr_device": args.ocr_device, "ort": ort.__version__, "warm_shapes": True,
              "providers": engine.text_rec.session.session.get_providers(), "cases": results,
              "total_seconds": {mode: sum(r["seconds"][mode] for r in results) for mode in ("greedy", "beam")}}
    write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
