"""E0 frozen-result audit and E1 single-Wire-change experiment on scheme G.

Uses signed image-only frontend snapshots. Never executes OCR/YOLO, never
changes current configuration, and never passes a target into any Stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from pin_tip_regression import (
    PREVIOUS, OUT as G_REPORTS, load_cases, read_json, write_json,
    verify_formal_state, AccessGuard, PipelineConfig, PipelineContext,
    signed_frontend, digest, reconstruct_scene, normalized_pairs,
)
from pin_tip_repair import saved_paths
from pin_skeleton_followup import SETS, strict_ids, followup_localization
from tools.analyze_pin_localization import strict_component_pairs, coordinate_assignment
from tools.run_box_skeleton_experiment import metrics as box_metrics
from evaluate_v2 import pin_counts, prf, evaluate_case, aggregate, source_from_name, net_members, pair_set
from pcb.coordinates import target_to_opencv
from pcb.core.registry import build_default_registry
from pcb.io import read_image, write_image
from pcb.submission import export, validate_strict
from wire_stub_preservation import restore_supported_stubs, POLICY

import cv2
import numpy as np

OUT = ROOT / "reports/pin_g_wire_e1"


def git_state():
    def query(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return {"head": query("rev-parse", "HEAD"), "branch": query("branch", "--show-current"),
            "status_short": query("status", "--short"), "tracked_diff_names": query("diff", "--name-only"),
            "tracked_diff_stat": query("diff", "--stat")}


def snapshot_paths(subset, variant):
    report = read_json((PREVIOUS / ("text_" + subset) if variant == "A" else G_REPORTS / subset) / "latest.json")
    return Path(report["metadata"]["run_directory"]) / "predictions" / variant, report


def conditional_pin_counts(pred, target):
    """Strict key/name/point metric, restricted to evaluator-TP Components.

    This is conditional, not a GT-Component Oracle. It still excludes upstream
    misses and cannot establish performance on those missed Components.
    """
    pairs = strict_component_pairs(pred["components"], target["components"])
    mapping = dict(pairs)
    pp = {pk: pred["pins"].get(pk, {}) for pk, _ in pairs}
    gg = {gk: target["pins"].get(gk, {}) for _, gk in pairs}
    counts = pin_counts(pp, gg, mapping, False)
    per_type = defaultdict(Counter)
    for pk, gk in pairs:
        values = pin_counts({pk: pp[pk]}, {gk: gg[gk]}, {pk: gk}, False)
        per_type[target["components"][gk]["type"]].update(dict(zip(("tp", "pred", "gt"), values)))
    return {"counts": dict(zip(("tp", "pred", "gt"), counts)), "matched_components": len(pairs),
            "per_type": {k: dict(v) for k, v in per_type.items()}}


def conditional_localization(pred, target, raw, height):
    """P0-compatible coordinate-only Hungarian assignment, then 5/10/20 gates."""
    pairs = strict_component_pairs(pred["components"], target["components"])
    lookup = {block["component"]: block["terminals"] for block in raw}
    counts = Counter(matched_components=len(pairs))
    for pk, gk in pairs:
        terms = lookup.get(pk, [])
        points = [target_to_opencv(p["point"]["x"], p["point"]["y"], height)
                  for p in target["pins"].get(gk, {}).values()]
        matches = coordinate_assignment(terms, points)
        counts.update(pred=len(terms), gt=len(points), count_abs_error=abs(len(terms) - len(points)),
                      exact_count=int(len(terms) == len(points)))
        for radius in (5, 10, 20):
            counts[f"tp_{radius}"] += sum(m["distance"] <= radius for m in matches)
    return dict(counts)


def localization_metrics(counts):
    n = counts.get("matched_components", 0)
    return {"counts": counts, "matching": "within strict matched Component; coordinates only; minimum-distance Hungarian",
            "pin_count_error_scope": "strict matched Components only, including in the end-to-end denominator diagnostic",
            "thresholds": {str(r): prf(counts.get(f"tp_{r}", 0), counts.get("pred", 0), counts.get("gt", 0))
                           for r in (5, 10, 20)},
            "pin_count_mae": counts.get("count_abs_error", 0) / n if n else None,
            "exact_count_accuracy": counts.get("exact_count", 0) / n if n else None}


def audit_e0():
    start = time.perf_counter()
    formal = verify_formal_state()
    git = git_state()
    guard = AccessGuard()
    cases = load_cases()
    config = PipelineConfig.load(ROOT / "configs/current.json")
    model_sha = digest(ROOT / "models/component_yolo11n_continue_v2_best.pt")
    records = []
    end_to_end = {v: [] for v in ("A", "G")}
    pin_totals = {v: Counter() for v in end_to_end}
    local_totals = {v: Counter() for v in end_to_end}
    end_local_totals = {v: Counter() for v in end_to_end}
    box_totals = {v: Counter() for v in end_to_end}
    types = {v: defaultdict(Counter) for v in end_to_end}
    for subset in ("design", "check"):
        inputs = {v: snapshot_paths(subset, v) for v in end_to_end}
        subset_tp = Counter()
        for key in SETS[subset]:
            case = cases[key]
            image = read_image(case.image_path)
            h, w = image.shape[:2]
            front = signed_frontend(case, config, model_sha)
            preds, diags, raw = {}, {}, {}
            for v, (folder, _) in inputs.items():
                preds[v] = read_json(folder / key / "result.json")
                diags[v] = read_json(folder / key / "diagnostics.json")
                raw[v] = read_json(folder / key / "raw_terminals.json")
                validate_strict(preds[v], (w, h))
                if [b["component"] for b in raw[v]] != [c.key for c in front.components]:
                    raise AssertionError("Raw terminal owner ordering differs from signed frontend")
            if preds["A"]["components"] != preds["G"]["components"]:
                raise AssertionError("Frozen Component predictions differ")
            with guard.evaluator():
                target = read_json(case.target_path)
                target_sha = digest(case.target_path)
                row = {"case_id": key, "subset": subset, "source": source_from_name(case.image_path.name),
                       "image_sha256": digest(case.image_path), "target_sha256": target_sha, "variants": {}}
                for v in end_to_end:
                    ev = evaluate_case(preds[v], target, diags[v])
                    subset_tp[v] += ev["metrics"]["Pin"]["tp"]
                    end_to_end[v].append({"case_id": key, "source": row["source"], "status": "ok", "evaluation": ev})
                    pins = conditional_pin_counts(preds[v], target)
                    local = conditional_localization(preds[v], target, raw[v], h)
                    e2e_local = {**local,
                                 "pred": sum(len(b["terminals"]) for b in raw[v]),
                                 "gt": sum(len(pins) for pins in target["pins"].values())}
                    pin_totals[v].update(pins["counts"])
                    local_totals[v].update(local)
                    end_local_totals[v].update(e2e_local)
                    for kind, cnt in pins["per_type"].items():
                        types[v][kind].update(cnt)
                    from pcb.core.interfaces import PinLocalizationOutput
                    loc = PinLocalizationOutput([(c, b["terminals"]) for c, b in zip(front.components, raw[v])])
                    box = followup_localization(front, {v: loc}, target, h)[v]
                    box_totals[v].update(box)
                    row["variants"][v] = {"end_to_end": ev["metrics"], "conditional_pin": pins,
                                          "conditional_localization": local, "box_side_constrained": box}
                    row["variants"][v]["end_to_end_raw_localization"] = e2e_local
                records.append(row)
            print(json.dumps({"E0_case": key, "strict_pin": {v: row["variants"][v]["end_to_end"]["Pin"]["tp"] for v in end_to_end},
                              "conditional_pin": {v: row["variants"][v]["conditional_pin"]["counts"] for v in end_to_end}}), flush=True)
        for v, (_, old) in inputs.items():
            expected = old["end_to_end"][v]["overall"]["metrics"]["Pin"]["tp"]
            if subset_tp[v] != expected:
                raise AssertionError(f"Saved metrics/GT drift for {subset}/{v}: {subset_tp[v]} != {expected}")
    payload = {"metadata": {"OFFICIAL_SCORE": False, "case_ids": [r["case_id"] for r in records],
                "case_count": len(records), "pipeline_config": config.as_dict(), "model_sha256": model_sha,
                "formal_state": verify_formal_state(), "git_before": git, "sealed_holdout_used": False,
                "target_in_inference_attempts": guard.target_attempts, "sealed_attempts": guard.sealed_attempts,
                "prediction_inference_rerun": False, "seconds": time.perf_counter() - start,
                "check_is_previously_used_development_regression_set": True},
               "end_to_end": {v: aggregate(rows) for v, rows in end_to_end.items()},
               "conditional_strict_pin": {v: prf(**dict(pin_totals[v])) for v in pin_totals},
               "conditional_raw_localization": {v: localization_metrics(dict(c)) for v, c in local_totals.items()},
               "end_to_end_raw_localization": {v: localization_metrics(dict(c)) for v, c in end_local_totals.items()},
               "matched_box_side_constrained": {v: box_metrics(dict(c)) for v, c in box_totals.items()},
               "conditional_strict_pin_by_type": {v: {k: prf(**dict(c)) for k, c in d.items()} for v, d in types.items()},
               "cases": records}
    write_json(OUT / "e0/latest.json", payload)
    lines = ["# E0：方案A与G的固定基线核验", "", "本地非官方诊断，固定14个已使用的开发case；不是150例或独立泛化成绩。",
             "复用已有预测与真实raw terminal；没有运行YOLO/OCR。元件匹配严格复用P0/evaluate_v2组件TP规则。", "",
             "|指标（微平均，宏平均另注明）|A|G|", "|---|---:|---:|"]
    for key in ("tp", "precision", "recall", "f1"):
        lines.append(f"|matched-component严格Pin {key}|{payload['conditional_strict_pin']['A'][key]:.6f}|{payload['conditional_strict_pin']['G'][key]:.6f}|")
    for r in (5, 10, 20):
        lines.append(f"|matched-component纯定位 Recall@{r}|{payload['conditional_raw_localization']['A']['thresholds'][str(r)]['recall']:.6f}|{payload['conditional_raw_localization']['G']['thresholds'][str(r)]['recall']:.6f}|")
    for key in ("tp", "macro_f1"):
        lines.append(f"|原始target端到端Pin {key}|{payload['end_to_end']['A']['overall']['metrics']['Pin'][key]:.6f}|{payload['end_to_end']['G']['overall']['metrics']['Pin'][key]:.6f}|")
    lines += ["", "严格Pin必须同时满足pin key、pinname和位置≤5px。纯定位只比较raw terminal与GT点，不使用脚号或脚名帮助匹配。",
              "条件指标排除了未严格匹配的Component，不是GT Component Oracle，也不是只删掉网络评分得到的所谓Pin独立分。",
              "JSON另保存端到端raw localization：GT分母使用全部GT terminal，预测分母使用全部raw terminal；未匹配元件的GT仍为漏检。",
              "evaluate_v2的原Pin计分仅用Component key映射，未要求其bbox/Name/value也达到Component TP。因此原端到端Pin TP可能高于严格matched-component条件TP，不能将两者直接相减解释为Pin算法收益。",
              "全类型纯定位沿用P0最低总距离Hungarian；box额外保留此前owner/side一致、阈值内最大数量优先的指标，两种口径不混用。",
              "G仍为独立实验方案；current.json及209个保护文件未变。后50例/Golden/QuickTest均未读取。"]
    (OUT / "e0/latest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"E0_complete": payload["conditional_strict_pin"], "formal": formal}, ensure_ascii=False), flush=True)
    return payload


def pair_changes(before, after, target):
    truth = pair_set([net_members(n) for n in target["nets"].values()])
    a, b = normalized_pairs(before, target), normalized_pairs(after, target)
    return {"correct_gained": sorted((b & truth) - (a & truth)),
            "correct_lost": sorted((a & truth) - (b & truth)),
            "incorrect_added": sorted((b - truth) - (a - truth)),
            "incorrect_removed": sorted((a - truth) - (b - truth))}


def _overlay(image, before, after, scene, path):
    component = next(c for c in scene.components if c.key == "U2")
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    xa, xb = max(0, round(x1 - 130)), min(scene.width, round(x2 + 35))
    ya, yb = max(0, round(y1 - 30)), min(scene.height, round(y2 + 40))
    panels = [image[ya:yb, xa:xb].copy()]
    panels += [cv2.cvtColor(m[ya:yb, xa:xb], cv2.COLOR_GRAY2BGR) for m in (before.mask, after.mask)]
    diff = image[ya:yb, xa:xb].copy()
    diff[(after.mask[ya:yb, xa:xb] > 0) & (before.mask[ya:yb, xa:xb] == 0)] = (0, 0, 255)
    panels.append(diff)
    for i, canvas in enumerate(panels):
        canvas = cv2.copyMakeBorder(canvas, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(canvas, ("Image", "G + current Wire", "G + supported Wire", "Red: added ink")[i],
                    (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
        panels[i] = canvas
    write_image(path, np.concatenate(panels, axis=1))


def run_e1(subset):
    start = time.perf_counter()
    verify_formal_state()
    guard = AccessGuard()
    cases = load_cases()
    config = PipelineConfig.load(ROOT / "configs/current.json")
    model_sha = digest(ROOT / "models/component_yolo11n_continue_v2_best.pt")
    registry = build_default_registry()
    policy_sha = digest(ROOT / "experiments/wire_stub_preservation.py")
    if subset == "check":
        design = read_json(OUT / "design/latest.json")
        if design["metadata"]["policy_source_sha256"] != policy_sha:
            raise AssertionError("Wire policy changed after Design")
        if not design["guards"]["no_new_incorrect_pairs"] or not design["guards"]["no_correct_pair_loss"]:
            raise AssertionError("Design guards failed; stop rather than tune on Check")
    keys = ["0011"] if subset == "smoke" else SETS[subset]
    folder = OUT / subset / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    evaluations = {v: [] for v in ("G_W0", "G_W1")}
    additions = Counter()
    for key in keys:
        case = cases[key]
        source_subset = "design" if key in SETS["design"] else "check"
        source = saved_paths(source_subset, key)["G"]
        image = read_image(case.image_path)
        h, w = image.shape[:2]
        front = signed_frontend(case, config, model_sha)
        context = PipelineContext(case.image_path.name, w, h, config, None)
        raw = read_json(source / "raw_terminals.json")
        reference = read_json(source / "result.json")
        scene, loc = reconstruct_scene(front, raw, image, context)
        if export(scene)["pins"] != reference["pins"] or export(scene)["components"] != reference["components"]:
            raise AssertionError("Reconstructed G Pin/Component field parity failed")
        domain_before = copy.deepcopy(scene.components)
        wire0 = registry.create("wire", "v3").run(image, scene, context)
        scene1 = copy.deepcopy(scene)
        wire1 = restore_supported_stubs(image, scene1, wire0)
        if scene.components != domain_before or scene1.components != domain_before:
            raise AssertionError("Wire modified frozen Pin/Component structures")
        if not np.array_equal(wire0.color_debug, wire1.color_debug):
            raise AssertionError("Palette changed")
        added = int(((wire1.mask > 0) & (wire0.mask == 0)).sum())
        if np.any((wire0.mask > 0) & (wire1.mask == 0)):
            raise AssertionError("E1 removed existing wire pixels")
        preds, diags = {}, {}
        for v, sc, wire in (("G_W0", scene, wire0), ("G_W1", scene1, wire1)):
            if v == "G_W1" and added == 0:
                preds[v] = copy.deepcopy(preds["G_W0"])
                diags[v] = copy.deepcopy(diags["G_W0"])
                diags[v]["wire_stub_preservation_e1"] = scene1.diagnostics["wire_stub_preservation_e1"]
            else:
                registry.create("topology", "v3").run(sc, wire, context)
                preds[v] = registry.create("submission", "official").run(sc, context).data
                diags[v] = sc.diagnostics
            validate_strict(preds[v], (w, h))
            if preds[v]["components"] != reference["components"] or preds[v]["pins"] != reference["pins"]:
                raise AssertionError("E1 final Component/Pin fields changed")
            write_json(folder / "predictions" / v / key / "result.json", preds[v])
            write_json(folder / "predictions" / v / key / "diagnostics.json", diags[v])
        if preds["G_W0"] != reference:
            raise AssertionError("Original G full JSON replay parity failed")
        with guard.evaluator():
            target = read_json(case.target_path)
            changes = pair_changes(preds["G_W0"], preds["G_W1"], target)
            evs = {v: evaluate_case(preds[v], target, diags[v]) for v in preds}
            if strict_ids(preds["G_W0"], target) != strict_ids(preds["G_W1"], target):
                raise AssertionError("Strict Pin TP identities changed")
            for v in preds:
                evaluations[v].append({"case_id": key, "source": source_from_name(case.image_path.name),
                                       "status": "ok", "evaluation": evs[v]})
            known_pair = ("U2.4", "U2.9")
            check_pair = known_pair in normalized_pairs(preds["G_W1"], target) if key == "0011" else None
        record = {"case_id": key, "added_pixels": added, "evaluation": evs, "pair_changes": changes,
                  "known_0011_pair_restored": check_pair,
                  "wire_decisions": scene1.diagnostics["wire_stub_preservation_e1"],
                  "G_full_JSON_replay_parity": True, "G_pin_fields_identical": True}
        records.append(record)
        for name, values in changes.items():
            additions[name] += len(values)
        if key == "0011":
            _overlay(image, wire0, wire1, scene1, folder / "0011_U2_wire_preservation.png")
        print(json.dumps({"E1_case": key, "added_pixels": added, "pair_changes": {k: len(v) for k, v in changes.items()},
                          "Pin_TP": evs["G_W1"]["metrics"]["Pin"]["tp"], "known_pair_restored": check_pair}), flush=True)
    payload = {"metadata": {"OFFICIAL_SCORE": False, "subset": subset, "case_ids": keys, "case_count": len(keys),
                "run_directory": str(folder), "policy_source_sha256": policy_sha, "policy": POLICY.__dict__,
                "seconds": time.perf_counter() - start, "sealed_holdout_used": False,
                "target_in_inference_attempts": guard.target_attempts, "sealed_attempts": guard.sealed_attempts,
                "formal_state": verify_formal_state(), "G_is_experimental_baseline": True,
                "check_is_previously_used_development_regression_set": True},
               "end_to_end": {v: aggregate(rows) for v, rows in evaluations.items()},
               "pair_change_counts": dict(additions),
               "guards": {"G_pin_and_component_unchanged": True, "G_full_JSON_replay_parity": True,
                          "strict_submission_pass_count": len(keys) * 2,
                          "no_correct_pair_loss": additions["correct_lost"] == 0,
                          "no_new_incorrect_pairs": additions["incorrect_added"] == 0,
                          "known_0011_pair_restored": next((r["known_0011_pair_restored"] for r in records if r["case_id"] == "0011"), None)},
               "cases": records}
    write_json(folder / "report.json", payload)
    write_json(OUT / subset / "latest.json", payload)
    print(json.dumps({"E1_complete": subset, "guards": payload["guards"], "pair_changes": dict(additions)}, ensure_ascii=False), flush=True)
    return payload


def finalize():
    e0 = read_json(OUT / "e0/latest.json")
    runs = {sub: read_json(OUT / sub / "latest.json") for sub in ("smoke", "design", "check") if (OUT / sub / "latest.json").exists()}
    rows_by_case = {r["case_id"]: r for sub in ("smoke", "design", "check") if sub in runs for r in runs[sub]["cases"]}
    rows = list(rows_by_case.values())
    lines = ["# G基础上的E0/E1实验决策", "", "本地非官方诊断；不包含后50例/Golden/QuickTest。未提交、上传或修改正式默认配置。", "",
             "## E0：补齐Pin自身指标", "", "matched-component严格Pin必须脚号/key、脚名、5px位置同时正确。", "",
             "|组|严格Pin TP|条件Precision|条件Recall|条件F1|端到端Pin宏F1|", "|---|---:|---:|---:|---:|---:|"]
    for v in ("A", "G"):
        c = e0["conditional_strict_pin"][v]
        lines.append(f"|{v}|{c['tp']}|{c['precision']:.6f}|{c['recall']:.6f}|{c['f1']:.6f}|{e0['end_to_end'][v]['overall']['metrics']['Pin']['macro_f1']:.6f}|")
    lines += ["", "条件指标不是GT Component Oracle；只回答现有严格匹配元件上的Pin表现。", "",
              "## E1：只改变Wire的图像笔画保护", "",
              "G的全部Pin/Component字段固定。原Wire/Topology/Submission逐字段复现后，独立实验Wire仅恢复真实、连续轴向笔画。",
              "保护范围由body尺寸/同侧间距/线宽决定；只接受连到已有外部Wire、实际8连通的笔画，不跨空白、不穿器件内部。",
              "没有沿用旧Pin坐标，也没有把GT长度写进推理规则。所有新增阈值在Design/Check前固定。", "",
              "|集合|案例数|新增正确Pair|丢失正确Pair|新增错误Pair|0011恢复|", "|---|---:|---:|---:|---:|---|"]
    for sub, r in runs.items():
        c = r["pair_change_counts"]
        known = r['guards']['known_0011_pair_restored'] if '0011' in r['metadata']['case_ids'] else '未包含0011'
        lines.append(f"|{sub}|{r['metadata']['case_count']}|{c.get('correct_gained', 0)}|{c.get('correct_lost', 0)}|{c.get('incorrect_added', 0)}|{known}|")
    summaries = {}
    if rows:
        summaries = {v: aggregate([{"case_id": r["case_id"], "source": source_from_name(load_cases()[r["case_id"]].image_path.name),
                                   "status": "ok", "evaluation": r["evaluation"][v]} for r in rows]) for v in ("G_W0", "G_W1")}
        lines += ["", "|已执行开发case合计指标|G+W0|G+W1|", "|---|---:|---:|"]
        for name in ("Pin", "NetHypergraph", "NetLine", "PinPair"):
            a, b = (summaries[v]["overall"]["metrics"][name] for v in ("G_W0", "G_W1"))
            lines.append(f"|{name}宏F1|{a['macro_f1']:.6f}|{b['macro_f1']:.6f}|")
            if name == "PinPair":
                lines.append(f"|PinPair TP / Pred|{a['tp']} / {a['pred']}|{b['tp']} / {b['pred']}|")
    safe = len(rows) == 14 and all(r["guards"]["no_new_incorrect_pairs"] and r["guards"]["no_correct_pair_loss"] for r in runs.values())
    fixed = runs.get("smoke", {}).get("guards", {}).get("known_0011_pair_restored", False)
    decision = "保留独立Wire实验，14例回归通过；仍不自动切换正式配置。" if safe and fixed else "未满足全部回归保护，不推广到正式配置、不跑150例，也不通过改Pin补偿网络。"
    if (OUT / "regression_trace.json").exists():
        trace = read_json(OUT / "regression_trace.json")
        failing = [r for r in trace["per_stub"] if r["pair_changes"]["incorrect_added"]]
        lines += ["", "## 0014回归归因", "",
                  "逐stub诊断反事实（不是部署规则，也不按GT选择恢复哪些stub）：",
                  "|仅恢复stub|唯一新增笔画像素|新增错误Pair|", "|---|---:|---:|"]
        for r in trace["per_stub"]:
            lines.append(f"|{r['restored_stub']}|{r['unique_added_pixels']}|{len(r['pair_changes']['incorrect_added'])}|")
        if failing:
            lines += ["", "仅恢复U1.18即可重现18个新增错误Pair；其余逐stub对照没有新增错误Pair。",
                      "原W0已把18个U1引用合并为一个大网。W1恢复该stub后，原singleton R28.2接入这个大网，Pair组合因此增加18条，并非图像中新增了18根导线。",
                      "GT中R28.2连接(U1.169,R28.2)；当前该位置附近预测为U1.18，pinname=PA2B/RESET，GT最近点key=pin_169、pinname=P_69，相距约12.69px。",
                      "不能凭internal-like格式宣布GT不可观测，也不能把全部回归归咎于新增假笔画。当前存在Pin引用/脚名/位置不一致与此前网络过合并，两者会放大真实笔画恢复后的评分损失。",
                      "该对照支持因果定位，不构成按case或GT过滤stub的算法。后续应把图像笔画恢复是否正确与Pin标识、网络过合并分别验证。"]
        boundary = trace.get("original_body_boundary_trace", {})
        if boundary.get("route_exists_in_original_mask"):
            lines += ["", f"原W0骨架中U1.18→U1.24的路径有{boundary['route_pixel_count']}个像素，其中{boundary['route_pixels_within_2px_of_left_boundary']}个在左body边界x={boundary['left_body_boundary_x']}的2px范围内。",
                      "局部图也保留了IC矩形框线。这进一步支持既有box边框残留参与了过合并；本轮没有修改barrier/Topology来掩盖该问题。"]
    lines += ["", "## 决策", "", decision, "",
              "Design和Check都是此前已使用的开发样本，不能据此声称独立泛化。E0/E1只核验当前假设，没有执行E2/E3或接入新模型。",
              "本轮E0使用14例；E1执行0011 smoke和6例Design（7个不同case）。因Design新增错误Pair，未进入8例Check或150例。",
              "下一步优先做E2的Pin锚点/标识证据审计，并将0014的已有过合并证据交给Wire/Topology负责人；不通过少恢复真实笔画来掩盖Pin/网络问题。"]
    state = verify_formal_state()
    payload = {"OFFICIAL_SCORE": False, "executed_unique_cases": len(rows), "decision": decision,
               "E0": e0["conditional_strict_pin"], "E1_aggregate": summaries,
               "subset_guards": {k: r["guards"] for k, r in runs.items()}, "formal_state": state,
               "git_after": git_state(), "case_ids": [r["case_id"] for r in rows],
               "E0_case_count": e0["metadata"]["case_count"], "check_executed": "check" in runs,
               "full_150_executed": False, "sealed_holdout_used": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "decision.json", payload)
    print(json.dumps({"decision": decision, "formal_state": state}, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("e0", "smoke", "design", "check", "finalize"), required=True)
    args = parser.parse_args()
    if args.task == "e0":
        audit_e0()
    elif args.task == "finalize":
        finalize()
    else:
        run_e1(args.task)
