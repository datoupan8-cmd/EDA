"""Compare Component text fields on frozen image-only frontends, then read GT."""
from pathlib import Path
import argparse
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pcb.component.latest.text_roles import classify_tokens
from pcb.component.text_enrichment import enrich_fields
from pcb.schema import Component, Pin, Text
from evaluate_v2 import component_counts, component_key_map, prf
from tools.team_access_guard import Guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", help="Unique development IDs; default 0001..0150")
    args = parser.parse_args()
    numbers = list(range(1, 151)) if args.cases is None else [int(x) for x in args.cases.split(",")]
    if not numbers or len(set(numbers)) != len(numbers) or any(n < 1 or n > 150 for n in numbers):
        raise ValueError("Only unique development cases 0001..0150")
    ids = [f"{n:04d}" for n in numbers]
    if args.dataset.name != "200_train_cases":
        raise ValueError("Require the explicit development dataset root")
    for cid in ids:
        folder = args.dataset / cid
        if not folder.is_dir() or folder.resolve().parent != args.dataset.resolve():
            raise ValueError(f"Missing or redirected allowed case {cid}")
        for name in ("result.json", "scene.json"):
            if not (args.baseline / cid / name).is_file():
                raise ValueError(f"Missing saved frontend {cid}/{name}")
    args.output.mkdir(parents=True, exist_ok=False)
    guard = Guard()
    modes = {"baseline": (False, False), "names": (True, False), "shorthand": (False, True), "both": (True, True)}
    print(json.dumps({"event": "start", "ACTUAL_CASE_IDS": ids, "mode": "frozen_frontend_replay"}), flush=True)
    for cid in ids:
        scene = json.loads((args.baseline / cid / "scene.json").read_text(encoding="utf-8"))
        baseline = json.loads((args.baseline / cid / "result.json").read_text(encoding="utf-8"))
        components = [Component(**{**c, "pins": [Pin(**p) for p in c["pins"]]}) for c in scene["components"]]
        texts = [Text(**t) for t in scene["texts"]]
        roles = classify_tokens(texts, [c.bbox for c in components if c.type in ("box", "block")])
        for mode, (names, shorthand) in modes.items():
            with guard.inferring():
                changed, diagnostics = enrich_fields(components, roles, scene["diagnostics"], names=names, shorthand=shorthand)
                prediction = copy.deepcopy(baseline)
                for component in changed:
                    prediction["components"][component.key]["Name"] = component.name or component.observable_designator or component.net_label or component.key
                    prediction["components"][component.key]["value"] = component.value
                if mode == "baseline" and prediction != baseline:
                    raise AssertionError("Frozen scene does not reproduce the saved Component fields")
                folder = args.output / cid / mode
                folder.mkdir(parents=True)
                (folder / "result.json").write_text(json.dumps(prediction, ensure_ascii=False), encoding="utf-8")
                (folder / "changes.json").write_text(json.dumps(diagnostics["component_text_enrichment"], ensure_ascii=False), encoding="utf-8")
    records = {mode: [] for mode in modes}
    for cid in ids:
        folder = args.dataset / cid
        targets = list(folder.glob("*_target*.json"))
        if len(targets) != 1 or targets[0].resolve().parent != folder.resolve():
            raise ValueError(f"Require one allowed target for {cid}")
        paths = [args.output / cid / mode / "result.json" for mode in modes]
        with guard.evaluating(targets[0], paths):
            gt = json.loads(targets[0].read_text(encoding="utf-8"))
        for mode in modes:
            prediction = json.loads((args.output / cid / mode / "result.json").read_text(encoding="utf-8"))
            counts = component_counts(prediction["components"], gt["components"], component_key_map(prediction["components"], gt["components"]))
            records[mode].append({"case_id": cid, **prf(*counts)})
    aggregates = {}
    for mode, rows in records.items():
        aggregates[mode] = {**prf(sum(r["tp"] for r in rows), sum(r["pred"] for r in rows), sum(r["gt"] for r in rows)),
                            "macro_f1": sum(r["f1"] for r in rows) / len(rows)}
    report = {"ACTUAL_CASE_IDS": ids, "OFFICIAL_SCORE": False, "sealed_holdout_used": False,
              "mode": "frozen_frontend_replay", "GT_opens": guard.target_opens,
              "aggregate": aggregates, "cases": records}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
