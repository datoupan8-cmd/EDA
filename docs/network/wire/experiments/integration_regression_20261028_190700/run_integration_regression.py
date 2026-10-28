from __future__ import annotations
import copy, hashlib, itertools, json, math, re, sys, traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from docs.network.wire.experiments.palette_recall_20261026_144402.run_wire_palette_replay import (
    DATA, edge_coverage, make_scene, mask_distance, read_gt_edges, saved_prediction,
)
from pcb.core.registry import StageRegistry
from pcb.core.team_config import load_team_config
from pcb.core.context import PipelineContext
from pcb.io import read_image
from pcb.network_shared.selection import selected_versions
from pcb.topology_stage.versions import TopologyStageV3
from pcb.wire_stage.registry import register_stages as register_wire_stages
import pcb.network_shared.composite as composite

OUT = Path(__file__).with_name("results.json")
REPORT = Path(__file__).with_name("report.md")
SCENARIOS = (
    ("default_v3_legacy", "v3_frame_guard", "legacy_v2"),
    ("v9_legacy", "v9_branch_salvage_recall", "legacy_v2"),
    ("v9_collinear", "v9_branch_salvage_recall", "path_collinear_v1"),
)
PIN_RE = re.compile(r"([^(),\s]+)\.([^(),\s]+)")
STAGES = ("wire_mask", "graph_paths", "geometry_result", "scene_nets")


def gt_membership_and_edges(target, height):
    membership = defaultdict(set)
    nets = {}
    for net_id, net in target.get("nets", {}).items():
        members = {f"{c}.{p}" for c, p in PIN_RE.findall(net.get("hyperGraph", ""))}
        nets[str(net_id)] = members
        for pin in members:
            membership[pin].add(str(net_id))
    edges = read_gt_edges(target, height)
    return nets, membership, {pin for pin, owners in membership.items() if len(owners) > 1}, edges


def connection_metrics(gt_nets, gt_membership, ambiguous, pred_pin_ids, predicted_partitions):
    matched = (set(gt_membership) & pred_pin_ids) - ambiguous
    unmatched_gt = set(gt_membership) - pred_pin_ids
    unmatched_pred = pred_pin_ids - set(gt_membership)
    expected = set()
    excluded = 0
    for members in gt_nets.values():
        expected.update(tuple(p) for p in itertools.combinations(sorted(members & matched), 2))
        excluded += sum(1 for p in itertools.combinations(sorted(members), 2)
                        if p[0] not in matched or p[1] not in matched)
    predicted = set()
    for group in predicted_partitions:
        members = sorted(set(group) & matched)
        predicted.update(tuple(p) for p in itertools.combinations(members, 2))
    tp, fp, fn = expected & predicted, predicted - expected, expected - predicted
    return {
        "matched_unambiguous_pin_count": len(matched),
        "unmatched_gt_pin_count": len(unmatched_gt),
        "unmatched_predicted_pin_count": len(unmatched_pred),
        "ambiguous_gt_pin_count": len(ambiguous),
        "excluded_gt_pair_count_due_to_unmatched_or_ambiguous_pins": excluded,
        "eligible_gt_pin_pair_count": len(expected),
        "eligible_predicted_pin_pair_count": len(predicted),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "precision": len(tp) / len(predicted) if predicted else None,
        "recall": len(tp) / len(expected) if expected else None,
        "tp_pairs": [list(p) for p in sorted(tp)],
        "fp_pairs": [list(p) for p in sorted(fp)],
        "fn_pairs": [list(p) for p in sorted(fn)],
        "unmatched_gt_pins": sorted(unmatched_gt),
        "unmatched_predicted_pins": sorted(unmatched_pred),
        "ambiguous_gt_pins": sorted(ambiguous),
        "matched_pin_ids": sorted(matched),
    }


def raster_segments(shape, segments):
    mask = np.zeros(shape, np.uint8)
    for a, b in segments:
        cv2.line(mask, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), 255, 1)
    return mask


def paths_to_segments(paths):
    return [((float(a[1]), float(a[0])), (float(b[1]), float(b[0])))
            for path in paths for a, b in zip(path, path[1:])]


def coverage(mask, edges):
    distance = mask_distance(mask)
    values = [edge_coverage(distance, edge, 5) for edge in edges]
    return {
        "macro_gt_edge_coverage_5px": float(np.mean(values)) if values else None,
        "gt_edges_ge_80pct_5px": sum(value >= .8 for value in values),
        "gt_edge_count": len(values),
        "per_edge": values,
    }


def provenance_counts(trace):
    counts = Counter()
    reasons = Counter()
    for group in trace.groups:
        for segment in group.segments:
            for source in segment.sources:
                counts[source.resolution] += 1
                if source.reason:
                    reasons[source.reason] += 1
    return {"source_records_by_resolution": dict(counts),
            "unknown_reasons": dict(reasons)}


def main():
    config = load_team_config(ROOT / "configs/team.json", root=ROOT)
    selected = selected_versions(ROOT)
    default_wire = json.loads((ROOT / "configs/wire_topology/current.json").read_text(encoding="utf-8"))["wire"]
    default_lines = json.loads((ROOT / "configs/net_lines/current.json").read_text(encoding="utf-8"))["version"]
    wire_registry = StageRegistry()
    register_wire_stages(wire_registry)
    if default_wire != "v3_frame_guard" or default_lines != "legacy_v2":
        raise RuntimeError(f"default selector mismatch: wire={default_wire}, net_lines={default_lines}")
    wire_ids = sorted({wire for _, wire, _ in SCENARIOS})
    missing = [name for name in wire_ids if name not in wire_registry.versions("wire")]
    if missing:
        raise RuntimeError(f"wire versions not registered: {missing}")

    data = {
        "schema_version": 1,
        "kind": "authorized development-set integration diagnostic; not official score",
        "integration_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "defaults": {"wire": default_wire, "net_lines": default_lines},
        "other_network_versions": selected,
        "scoring": {"geometry": "sampled GT edge points within 5 px of stage geometry; per-edge macro and >=80% counts",
                    "connectivity": "exact component.pin IDs; omit ambiguous GT pins; unordered pair TP/FP/FN; diagnostic only"},
        "samples": [],
    }
    scenario_pairs = {key: {"tp": set(), "fp": set(), "fn": set()} for key, _, _ in SCENARIOS}
    connectivity_totals = {key: Counter() for key, _, _ in SCENARIOS}
    stage_values = {key: {stage: [] for stage in STAGES} for key, _, _ in SCENARIOS}
    first_loss = {key: Counter() for key, _, _ in SCENARIOS}
    hashes = {}
    failures = []

    for case_id in range(1, 14):
        sample_id = f"{case_id:04d}"
        image_path = DATA / str(case_id) / f"{case_id}.png"
        target_path = DATA / str(case_id) / f"{case_id}.json"
        row = {"sample_id": sample_id, "input_image": str(image_path), "target_path": str(target_path),
               "status": "running", "scenarios": {}}
        try:
            image = read_image(image_path)
            height, width = image.shape[:2]
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            hashes[sample_id] = image_hash
            prediction, diagnostics = saved_prediction(case_id)
            target = json.loads(target_path.read_text(encoding="utf-8"))
            gt_nets, membership, ambiguous, edges = gt_membership_and_edges(target, height)
            context = PipelineContext(image_path.name, width, height, config, None)
            wire_cache = {}
            sample_pair_sets = {}
            sample_first_loss = Counter()

            for key, wire_id, netline_id in SCENARIOS:
                try:
                    if wire_id not in wire_cache:
                        wire_scene = make_scene(image, prediction, diagnostics)
                        wire_stage = wire_registry.create("wire", wire_id)
                        wire_output = wire_stage.run(image, wire_scene, context)
                        wire_cache[wire_id] = (copy.deepcopy(wire_scene), wire_output)
                    wire_scene, wire_output = wire_cache[wire_id]
                    scene = copy.deepcopy(wire_scene)
                    versions = dict(selected)
                    versions["net_lines"] = netline_id
                    topology = TopologyStageV3(network_versions=versions)
                    capture = {}
                    original_validate, original_assemble = composite.validate_plan, composite.assemble
                    def capture_validate(plan, scene_view, graph):
                        original_validate(plan, scene_view, graph)
                        capture["plan"] = plan
                        capture["graph"] = graph
                    def capture_assemble(plan, geometry):
                        capture["geometry"] = geometry
                        return original_assemble(plan, geometry)
                    composite.validate_plan, composite.assemble = capture_validate, capture_assemble
                    try:
                        topology.run(scene, wire_output, context)
                    finally:
                        composite.validate_plan, composite.assemble = original_validate, original_assemble
                    graph, plan, geometry = capture["graph"], capture["plan"], capture["geometry"]
                    plain = topology.lines.run(graph, plan)
                    traced_geometry, trace = topology.lines.run_with_provenance(graph, plan)
                    if plain != traced_geometry or geometry != plain:
                        raise AssertionError("run(), run_with_provenance(), and Topology geometry differ")
                    if graph.reference_segment_edge_ranges is None:
                        raise AssertionError("reference_segment_edge_ranges missing")
                    reference_count = sum(len(group) for group in graph.reference_segments)
                    range_count = valid_range_count = 0
                    for path_id, items in enumerate(graph.reference_segment_edge_ranges):
                        if len(items) != len(graph.reference_segments[path_id]):
                            raise AssertionError(f"reference range alignment failed for path {path_id}")
                        for interval in items:
                            range_count += 1
                            if interval is not None and 0 <= interval[0] < interval[1] <= len(graph.paths[path_id]) - 1:
                                valid_range_count += 1
                    path_segments = paths_to_segments(graph.paths)
                    geometry_segments = [segment for group in geometry.groups for segment in group.segments]
                    exported_segments = [segment for net in scene.nets for segment in net.segments]
                    selected_path_ids = sorted({path_id for group in plan.groups for path_id in group.path_ids})
                    group_segment_counts = {group.group_id: len(group.segments) for group in geometry.groups}
                    no_pin_groups = [group for group in plan.groups if not group.pins]
                    filtered_geometry_count = sum(group_segment_counts.get(group.group_id, 0) for group in no_pin_groups)
                    masks = {
                        "wire_mask": np.asarray(wire_output.mask),
                        "graph_paths": raster_segments((height, width), path_segments),
                        "geometry_result": raster_segments((height, width), geometry_segments),
                        "scene_nets": raster_segments((height, width), exported_segments),
                    }
                    stage_coverage = {stage: coverage(mask, edges) for stage, mask in masks.items()}
                    for stage in STAGES:
                        stage_values[key][stage].extend(stage_coverage[stage]["per_edge"])
                    for edge_index, wire_cov in enumerate(stage_coverage["wire_mask"]["per_edge"]):
                        if wire_cov < .8:
                            continue
                        chain = [stage_coverage[stage]["per_edge"][edge_index] >= .8 for stage in STAGES]
                        for index in range(1, len(chain)):
                            if chain[index - 1] and not chain[index]:
                                first_loss[key][STAGES[index]] += 1
                                sample_first_loss[STAGES[index]] += 1
                                break

                    predicted_pin_ids = {f"{component.key}.{pin.number}" for component in scene.components for pin in component.pins}
                    conn = connection_metrics(gt_nets, membership, ambiguous, predicted_pin_ids,
                                              [tuple(net.pins) for net in scene.nets])
                    sample_pair_sets[key] = {name: { (sample_id, tuple(pair)) for pair in conn[f"{name}_pairs"] }
                                             for name in ("tp", "fp", "fn")}
                    for name in ("tp", "fp", "fn"):
                        scenario_pairs[key][name].update(sample_pair_sets[key][name])
                    for name in ("matched_unambiguous_pin_count", "unmatched_gt_pin_count",
                                 "unmatched_predicted_pin_count", "ambiguous_gt_pin_count",
                                 "eligible_gt_pin_pair_count", "tp", "fp", "fn"):
                        connectivity_totals[key][name] += conn[name]
                    record = {
                        "status": "passed",
                        "wire_version": wire_id, "netline_version": netline_id,
                        "wire_mask_pixels": int((np.asarray(wire_output.mask) > 0).sum()),
                        "graph_path_count": len(graph.paths),
                        "graph_reference_segment_count": reference_count,
                        "reference_edge_ranges_valid": valid_range_count,
                        "reference_edge_ranges_total": range_count,
                        "plan_group_count": len(plan.groups),
                        "plan_selected_path_count": len(selected_path_ids),
                        "groups_without_pins": len(no_pin_groups),
                        "geometry_result_segment_count": len(geometry_segments),
                        "geometry_segments_filtered_for_no_pin_groups": filtered_geometry_count,
                        "scene_net_count": len(scene.nets),
                        "scene_nets_segment_count": len(exported_segments),
                        "stage_geometry": {stage: {k: v for k, v in stats.items() if k != "per_edge"}
                                           for stage, stats in stage_coverage.items()},
                        "first_downstream_loss_after_wire_ge80": dict(sample_first_loss),
                        "provenance": provenance_counts(trace),
                        "run_matches_run_with_provenance": plain == traced_geometry,
                        "topology_geometry_matches_backend_run": geometry == plain,
                        "connectivity": conn,
                    }
                    row["scenarios"][key] = record
                except Exception as exc:
                    record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()}
                    row["scenarios"][key] = record
                    failures.append({"sample_id": sample_id, "scenario": key, **record})
            row["status"] = "passed" if len(row["scenarios"]) == len(SCENARIOS) and all(
                item["status"] == "passed" for item in row["scenarios"].values()) else "partial_failure"
            row["gt_edge_count"] = len(edges)
            row["image_sha256"] = image_hash
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            failures.append({"sample_id": sample_id, **row})
        data["samples"].append(row)

    summary = {}
    for key, _, _ in SCENARIOS:
        stage_summary = {}
        for stage in STAGES:
            values = stage_values[key][stage]
            stage_summary[stage] = {
                "macro_gt_edge_coverage_5px": float(np.mean(values)) if values else None,
                "sample_macro_gt_edge_coverage_5px": float(np.mean([
                    np.mean(sample["scenarios"][key]["stage_geometry"][stage]["macro_gt_edge_coverage_5px"])
                    for sample in data["samples"]
                    if sample.get("scenarios", {}).get(key, {}).get("status") == "passed"
                    and sample["scenarios"][key]["stage_geometry"][stage].get("gt_edge_count", 0) > 0
                ])) if values else None,
                "gt_edges_ge_80pct_5px": sum(v >= .8 for v in values),
                "gt_edge_count": len(values),
            }
        pair_sets = scenario_pairs[key]
        tp, fp, fn = (pair_sets[name] for name in ("tp", "fp", "fn"))
        predicted = len(tp) + len(fp); expected = len(tp) + len(fn)
        summary[key] = {
            "stages": stage_summary,
            "connectivity": {
                "tp": len(tp), "fp": len(fp), "fn": len(fn),
                "precision": len(tp) / predicted if predicted else None,
                "recall": len(tp) / expected if expected else None,
                "matched_unambiguous_pin_count_across_samples": connectivity_totals[key]["matched_unambiguous_pin_count"],
                "unmatched_gt_pin_count_across_samples": connectivity_totals[key]["unmatched_gt_pin_count"],
                "unmatched_predicted_pin_count_across_samples": connectivity_totals[key]["unmatched_predicted_pin_count"],
                "ambiguous_gt_pin_count_across_samples": connectivity_totals[key]["ambiguous_gt_pin_count"],
                "eligible_gt_pin_pair_count": expected,
            },
            "aggregate_counts": {name: int(connectivity_totals[key][name]) for name in
                                  ("matched_unambiguous_pin_count", "unmatched_gt_pin_count",
                                   "unmatched_predicted_pin_count", "ambiguous_gt_pin_count")},
        }
    data["summary"] = summary
    data["connectivity_comparison_vs_historical"] = {
        "historical_reference": {"tp": 2, "fp": 54, "fn": 324, "precision": 2/56, "recall": 2/326,
                                  "matched_unambiguous_pins": 370},
        "current_exact_id_pair_sets": {key: value["connectivity"] for key, value in summary.items()},
        "same_protocol_note": "same saved_prediction helper, GT ID parser, exact component.pin matching, ambiguity exclusion and pairwise metric implementation as the prior v9 detailed diagnostic; verify counts below before claiming comparability",
    }
    data["first_loss_summary_after_wire_ge80"] = {key: dict(value) for key, value in first_loss.items()}
    data["failures"] = failures
    data["cache_and_model"] = {"component_pin_cache_reused": True,
                               "component_pin_model_inference_rerun": False,
                               "prediction_sources": "existing saved pipeline result.json and diagnostics.json",
                               "images_and_targets": "authorized open_GT samples 0001-0013"}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = ["# Wire v9 + NetLine integration regression", "",
            f"Integration HEAD: `{data['integration_head']}`", "",
            "This is a local development-set diagnostic, not an official competition score. Component/Pin predictions were reused from saved artifacts; no Component/Pin inference was run.", "",
            "## Aggregate geometry (5 px sampled edge coverage)", "",
            "| Scenario | Wire mask | Graph paths | GeometryResult | Scene.nets | >=80% edges at Wire / Graph / Geometry / Scene | Net count (sum) | Geometry segments (sum) | Scene segments (sum) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, _, _ in SCENARIOS:
        s=summary[key]; cov=[s['stages'][stage]['macro_gt_edge_coverage_5px'] for stage in STAGES]
        ge=[s['stages'][stage]['gt_edges_ge_80pct_5px'] for stage in STAGES]
        net_count=sum(x['scenarios'][key].get('scene_net_count',0) for x in data['samples'] if x.get('scenarios',{}).get(key,{}).get('status')=='passed')
        geom_count=sum(x['scenarios'][key].get('geometry_result_segment_count',0) for x in data['samples'] if x.get('scenarios',{}).get(key,{}).get('status')=='passed')
        scene_count=sum(x['scenarios'][key].get('scene_nets_segment_count',0) for x in data['samples'] if x.get('scenarios',{}).get(key,{}).get('status')=='passed')
        rows.append(f"| {key} | {cov[0]:.2%} | {cov[1]:.2%} | {cov[2]:.2%} | {cov[3]:.2%} | {' / '.join(map(str,ge))} | {net_count} | {geom_count} | {scene_count} |")
    rows += ["", "## Connectivity exact-ID diagnostic", "",
             "| Scenario | TP | FP | FN | Precision | Recall | matched Pins (sum) | unmatched GT Pins (sum) | unmatched predicted Pins (sum) | ambiguous GT Pins (sum) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,_,_ in SCENARIOS:
        c=summary[key]['connectivity']; rows.append(f"| {key} | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']:.2%} | {c['recall']:.2%} | {c['matched_unambiguous_pin_count_across_samples']} | {c['unmatched_gt_pin_count_across_samples']} | {c['unmatched_predicted_pin_count_across_samples']} | {c['ambiguous_gt_pin_count_across_samples']} |")
    rows += ["", "Historical exact-ID pair baseline: TP=2, FP=54, FN=324, precision=3.57%, recall=0.61%, matched unambiguous pins=370. It is comparable only if current counts and protocol agree.", "",
             "## Per-sample status and stage coverage", "",
             "| Sample | Status | Scenario | GT edges | Wire mask | Graph paths | GeometryResult | Scene.nets | no-Pin groups | filtered geometry segments | complete/unknown provenance refs |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sample in data['samples']:
        for key,_,_ in SCENARIOS:
            r=sample.get('scenarios',{}).get(key,{})
            if r.get('status')!='passed':
                rows.append(f"| {sample['sample_id']} | {sample.get('status')} | {key} | {sample.get('gt_edge_count','?')} | FAILED | — | — | — | — | — | — |")
                continue
            cov=[r['stage_geometry'][stage]['macro_gt_edge_coverage_5px'] for stage in STAGES]
            prov=r['provenance']['source_records_by_resolution']
            rows.append(f"| {sample['sample_id']} | {sample['status']} | {key} | {sample['gt_edge_count']} | {cov[0]:.2%} | {cov[1]:.2%} | {cov[2]:.2%} | {cov[3]:.2%} | {r['groups_without_pins']} | {r['geometry_segments_filtered_for_no_pin_groups']} | {prov.get('complete',0)}/{prov.get('unknown',0)} |")
    rows += ["", "## Known sample 0001 cases", "",
             "Coverage values for net_4/edge_4 and net_12/edge_2 are recorded in `results.json` for each scenario. Group-level spatial association is diagnostic only; it does not establish GT electrical membership. No Pin-less group export policy was changed.", "",
             "## Limitations and owner handoff", "",
             "- Wire/Graph geometry proximity is not proof of electrical connectivity.",
             "- The exact-ID connectivity metric excludes unmatched and ambiguous Pins; it does not assess them as correct.",
             "- `reference_segment_edge_ranges` maps simplified reference segments to raw path-edge intervals; it does not establish Pin-to-edge attachment.",
             "- Existing `assemble()` omits GeometryResult groups with no Pins. Whether those geometries should be exported requires Integration/Submission and Topology owner decision.",
             "- GT geometry present in Wire but absent in Graph paths should be investigated by Graph owner; Pin identity and plan membership issues belong to Pin/Topology/NetHypergraph owners.",
             "- Experimental Wire versions remain non-default; this report is not algorithm acceptance or an official score.", "",
             f"Failures: {len(failures)}. See `results.json` for errors and tracebacks.", ""]
    REPORT.write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"results": str(OUT), "report": str(REPORT), "samples": len(data['samples']),
                      "successful_samples": sum(x['status']=='passed' for x in data['samples']),
                      "failures": len(failures), "summary": summary,
                      "first_loss_summary": data['first_loss_summary_after_wire_ge80']}, ensure_ascii=False))

if __name__ == "__main__":
    main()
