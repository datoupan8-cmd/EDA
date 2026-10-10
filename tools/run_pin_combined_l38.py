"""Standalone PNG inference and gated, frozen L38 integration verification."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
from pin_combined_l38 import build_runtime, digest
from pcb.core.context import PipelineContext
from pcb.core.pipeline import ModularPipeline
from pcb.core.registry import build_default_registry
from pcb.io import read_image, write_json

OUT = ROOT / "reports/pin_combined_l38"
DESIGN = ("0014", "0017", "0018", "0087", "0103")
CHECK = tuple(f"{i:04d}" for i in range(5, 151, 5))
SMOKE = ("0014", "0087", "0103")
ALL = DESIGN + CHECK


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Guard:
    """Disallow inference GT and archived intermediates; only evaluator opts in."""
    def __init__(self, install=True):
        self.phase = "idle"
        self.allowed_target = None
        self.target_opens = self.blocked = 0
        if install:
            sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        value = str(args[0]).replace("\\", "/").lower()
        parts = value.split("/")
        if any(p in {"golden", "sealed", "eda_pin_crossing_quicktest"} or p.startswith(("10gt", "10_gtcase")) for p in parts):
            self.blocked += 1
            raise PermissionError("Reserved/QuickTest data prohibited")
        if "200_train_cases" in parts:
            pos = parts.index("200_train_cases")
            if pos + 1 < len(parts):
                cid = parts[pos + 1]
                if not cid.isdigit() or not 1 <= int(cid) <= 150:
                    self.blocked += 1
                    raise PermissionError("Only development cases 0001..0150")
        if value.endswith("_target.json"):
            if value != self.allowed_target or self.phase != "evaluation":
                self.blocked += 1
                raise PermissionError("GT may only be read after saved predictions")
            self.target_opens += 1
        mode = args[1] if len(args) > 1 else None
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        archived = ("/tmp/pin_sina_frontend/" in value or "/predictions/" in value
                    or "/words/" in value or value.endswith("raw_terminals.json")
                    or ("/reports/" in value and value.endswith("/complete.json")))
        if self.phase == "inference" and archived and not writing:
            self.blocked += 1
            raise PermissionError("Image inference cannot consume archived stage outputs")

    @contextmanager
    def inferring(self):
        self.phase = "inference"
        try:
            yield
        finally:
            self.phase = "idle"

    @contextmanager
    def evaluating(self, path, saved):
        if not all(Path(p).is_file() for p in saved):
            raise AssertionError("Save both PNG-only predictions first")
        self.allowed_target = str(Path(path).resolve()).replace("\\", "/").lower()
        self.phase = "evaluation"
        try:
            yield
        finally:
            self.phase, self.allowed_target = "idle", None


def git_state():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "branch": git("branch", "--show-current").decode().strip(),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary")).hexdigest(),
            "status_short": git("status", "--short").decode(errors="replace")}


def freeze():
    fp = OUT / "initial_state.json"
    if fp.exists():
        result = read(fp)
    else:
        prev = ROOT / "reports/pin_library_semantics_l37"
        hashes = dict(read(prev / "initial_state.json")["sha256"])
        hashes.update(read(prev / "delivery_verification.json")["artifact_sha256"])
        hashes[str(prev / "delivery_verification.json")] = digest(prev / "delivery_verification.json")
        for name in ("experiments/pin_combined_l38.py", "tools/run_pin_combined_l38.py", "tests/pin/test_pin_combined_l38.py", "configs/experiments/pin_combined_l38.json", "reports/pin_combined_l38/experiment_plan.md"):
            hashes[str(ROOT / name)] = digest(ROOT / name)
        result = {"sha256": hashes, "git": git_state(), "Design": DESIGN, "Check": CHECK,
                  "single_variable": "frozen combination image-only integration", "OFFICIAL_SCORE": False}
        write_json(fp, result)
    verify(result)
    return result


def verify(state):
    for name, expected in state["sha256"].items():
        if digest(Path(name)) != expected:
            raise AssertionError("Frozen source/input changed: " + name)
    now = git_state()
    if any(now[k] != state["git"][k] for k in ("head", "branch", "tracked_diff_sha256")):
        raise AssertionError("Prior tracked changes altered")


def normalized(value):
    def convert(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(type(item).__name__)
    return json.loads(json.dumps(value, default=convert))


def differences(a, b, prefix="", limit=8):
    """First exact JSON discrepancies; not a fuzzy scoring rule."""
    found = []
    def visit(x, y, path):
        if len(found) >= limit:
            return
        if type(x) is not type(y):
            found.append({"path": path, "old": str(x)[:120], "new": str(y)[:120]})
        elif isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    found.append({"path": path + "/" + str(k), "missing": "old" if k not in x else "new"})
                else:
                    visit(x[k], y[k], path + "/" + str(k))
        elif isinstance(x, list):
            if len(x) != len(y):
                found.append({"path": path, "old_count": len(x), "new_count": len(y)})
            else:
                for i, (u, v) in enumerate(zip(x, y)):
                    visit(u, v, path + "/" + str(i))
        elif x != y:
            found.append({"path": path, "old": x, "new": y})
    visit(normalized(a), normalized(b), prefix)
    return found[:limit]


def archived_case(case):
    phase = "design" if case in DESIGN else "check"
    folder = ROOT / "reports/pin_library_semantics_l37" / phase
    return read(folder / f"cases/{case}.json"), folder


def persist(outputs, folder, name):
    for variant, artifacts in (("A", outputs.baseline), ("B", outputs.candidate)):
        write_json(folder / f"predictions/{variant}/{name}/result.json", artifacts.result)
        write_json(folder / f"predictions/{variant}/{name}/diagnostics.json", artifacts.scene.diagnostics)
    write_json(folder / f"raw/{name}.json", outputs.raw)
    write_json(folder / f"word_pools/{name}.json", outputs.words)
    write_json(folder / f"trace/{name}.json", outputs.trace)


def compare_archive(outputs, case):
    # Validation only: this function is called after the new prediction was saved.
    meta, folder = archived_case(case)
    front = read(ROOT.parent / f"tmp/pin_sina_frontend/{case}.json")
    rawpath = (ROOT / f"reports/pin_terminal_reader_l2/design/raw_terminals/{case}.json" if case in DESIGN
               else ROOT / f"reports/pin_learned_locator_l1/validation/predictions/L1/{case}/raw_terminals.json")
    poolpath = ROOT / "reports/pin_word_decoder_l3" / ("read_design" if case in DESIGN else "check") / f"words/{case}.json"
    comparisons = {
        "frontend_components": (front["components"], [asdict(c) for c in outputs.frontend.components]),
        "frontend_texts": (front["texts"], [asdict(t) for t in outputs.frontend.texts]),
        "frontend_roles": (front["roles"], [asdict(r) for r in outputs.frontend.roles]),
        "raw_terminals": (read(rawpath), outputs.raw),
        "whole_word_pool": (read(poolpath), outputs.words),
        "final_prediction": (read(folder / f"predictions/{case}/result.json"), outputs.candidate.result),
    }
    return {key: {"exact": not differences(a, b), "differences": differences(a, b)} for key, (a, b) in comparisons.items()}


def run_phase(task, args, state, guard):
    if task == "design" and not read(OUT / "smoke/complete.json")["all_reproduced"]:
        raise RuntimeError("Smoke not reproduced; do not expand")
    if task == "check" and not read(OUT / "design/complete.json")["proceed"]:
        raise RuntimeError("Design does not justify Check")
    folder = OUT / task
    if (folder / "complete.json").exists():
        raise RuntimeError("Completed phase preserved; no reselection")
    with guard.inferring():
        pipeline, ocr, detector = build_runtime(ROOT, args.library, ROOT / "runs/ocr_cache_v4_1", ROOT / "reports/pin_word_decoder_l3/ocr_cache")
    cases = SMOKE if task == "smoke" else DESIGN if task == "design" else CHECK
    rows = []
    from tools.run_pin_side_joint_l36 import metrics, strict_ids, pair_changes, aggregate
    for case in cases:
        record_path = folder / f"cases/{case}.json"
        if record_path.exists():
            row = read(record_path)
            if not row["all_reproduced"]:
                raise RuntimeError("Preserved failed case; no blind rerun")
            rows.append(row)
            continue
        meta, _ = archived_case(case)
        image_paths = [Path(p) for p in meta["inputs"] if p.lower().endswith(".png")]
        if len(image_paths) != 1:
            raise AssertionError("Unique signed image required")
        image_path = image_paths[0]
        if digest(image_path) != meta["inputs"][str(image_path)]:
            raise AssertionError("Dataset image changed; version audit first")
        with guard.inferring():
            image = read_image(image_path)
            context = PipelineContext(image_path.name, image.shape[1], image.shape[0], pipeline.config, ocr, detector)
            outputs = pipeline.run(image, context)
        persist(outputs, folder, case)
        parity = compare_archive(outputs, case)
        baseline_parity = None
        if task == "smoke" and case == SMOKE[0]:
            with guard.inferring():
                reference = ModularPipeline(pipeline.config, build_default_registry()).run(image, context)
            baseline_parity = reference.result == outputs.baseline.result
        passed = all(v["exact"] for v in parity.values()) and baseline_parity is not False
        row = {"case_id": case, "source": meta["source"], "parity": parity,
               "formal_ModularPipeline_JSON_parity": baseline_parity, "all_reproduced": passed,
               "seconds": outputs.trace["total_seconds"], "image_sha256": digest(image_path),
               "prediction_sha256": {v: digest(folder / f"predictions/{v}/{case}/result.json") for v in ("A", "B")},
               "frontend_diagnostics": outputs.baseline.scene.diagnostics.get("ocr", {}),
               "local_word_cache": {"hits": pipeline.reader.cache_hits, "detector_calls": pipeline.reader.calls, "direct_crops": pipeline.reader.direct_calls}}
        write_json(record_path, row)
        if not passed:
            write_json(folder / "reproduction_failure.json", row)
            raise RuntimeError("Image-only reproduction differs; stop at " + case)
        if task != "smoke":
            saved = [folder / f"predictions/{v}/{case}/result.json" for v in ("A", "B")]
            with guard.evaluating(meta["target_path"], saved):
                if digest(Path(meta["target_path"])) != meta["target_sha256"]:
                    raise AssertionError("Latest target drift; dataset audit required")
                target = read(meta["target_path"])
            for label, artifacts in (("before", outputs.baseline), ("after", outputs.candidate)):
                row[label] = metrics(artifacts.result, target, artifacts.scene.diagnostics, case, meta["source"])
            old, new = strict_ids(outputs.baseline.result, target), strict_ids(outputs.candidate.result, target)
            row.update(gained=sorted(new - old), lost=sorted(old - new),
                       pair_changes={k:len(v) for k,v in pair_changes(outputs.baseline.result, outputs.candidate.result,target).items()})
            write_json(record_path, row)
        rows.append(row)
        write_json(folder / "progress.json", {"case_ids": [r["case_id"] for r in rows], "done": len(rows)})
        print(json.dumps({"case": case, "reproduced": passed, "seconds": row["seconds"],
              "scores": {k:row[k]["evaluation"]["FinalScore"] for k in ("before", "after") if k in row}},ensure_ascii=False), flush=True)
    verify(state)
    result = {"cases":rows,"all_reproduced":all(r["all_reproduced"] for r in rows),
              "OFFICIAL_SCORE":False,"sealed_holdout_used":False,"target_opens":guard.target_opens,
              "blocked_access_attempts":guard.blocked,"full_150_run":False,"formal_config_changed":False}
    if task != "smoke":
        summary={k:aggregate([r[k] for r in rows]) for k in ("before","after")}
        delta=summary["after"]["overall"]["FinalScore"]-summary["before"]["overall"]["FinalScore"]
        result.update(summary=summary,score_delta_points=delta,proceed=delta>1e-9,
                      gained=[{"case_id":r["case_id"],"pin":p} for r in rows for p in r["gained"]],
                      lost=[{"case_id":r["case_id"],"pin":p} for r in rows for p in r["lost"]])
        pairs=Counter()
        for r in rows:pairs.update(r["pair_changes"])
        result["pair_changes"]=dict(pairs)
        result["priority_for_150_evidence"]=(task=="check" and delta>=.5)
    write_json(folder/"complete.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("cases","summary","gained","lost")},ensure_ascii=False),flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=("smoke","design","check","verify"))
    parser.add_argument("--image",type=Path,help="Standalone PNG; no GT or archived case metadata")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--fresh-cache",action="store_true",help="Standalone inference with a new isolated cache; original caches untouched")
    args=parser.parse_args();guard=Guard()
    if args.image:
        if not args.output or args.task:parser.error("--image requires --output and no --task")
        cache=args.output/"independent_cache" if args.fresh_cache else ROOT/"runs/ocr_cache_v4_1"
        words=args.output/"independent_words" if args.fresh_cache else ROOT/"reports/pin_word_decoder_l3/ocr_cache"
        with guard.inferring():
            pipeline,ocr,detector=build_runtime(ROOT,args.library,cache,words,yolo_cache=cache/"yolo" if args.fresh_cache else None)
            image=read_image(args.image)
            context=PipelineContext(args.image.name,image.shape[1],image.shape[0],pipeline.config,ocr,detector)
            output=pipeline.run(image,context)
        persist(output,args.output,args.image.stem)
        write_json(args.output/"runtime.json",{"trace":output.trace,"ocr":ocr.last_diagnostics,
                   "local_calls":pipeline.reader.calls,"local_direct_crops":pipeline.reader.direct_calls,
                   "cache_hits":pipeline.reader.cache_hits,"target_opens":guard.target_opens,
                   "fresh_cache":args.fresh_cache,"GT_read_in_inference":False,"OFFICIAL_SCORE":False})
        print(json.dumps({"image_only":True,"seconds":output.trace["total_seconds"],"target_opens":guard.target_opens},ensure_ascii=False))
        return
    if not args.task:parser.error("Select --task or --image")
    state=freeze()
    if args.task!="verify":run_phase(args.task,args,state,guard)
    verify(state)
    write_json(OUT/f"{args.task}_verification.json",{"protected_files":len(state["sha256"]),"unchanged":True,"OFFICIAL_SCORE":False,"formal_config_changed":False,"commit":False,"push":False})


if __name__=="__main__":main()
