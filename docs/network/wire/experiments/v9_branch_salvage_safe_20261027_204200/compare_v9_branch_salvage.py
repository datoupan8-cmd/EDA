from __future__ import annotations
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from docs.network.wire.experiments.palette_recall_20261026_144402.run_wire_palette_replay import (
    DATA, edge_coverage, make_scene, mask_distance, pin_sets, read_gt_edges, saved_prediction,
)
from pcb.core.context import PipelineContext
from pcb.core.team_config import load_team_config
from pcb.io import read_image
from pcb.network_shared.selection import selected_versions
from pcb.topology_stage.versions import TopologyStageV3
from pcb.wire_stage.registry import register_stages
from pcb.wire_stage.retained import (
    WireStageV3FrameGuard, WireStageV7PaletteCorridorRecall,
    WireStageV8BaselineAnchoredRecall, WireStageV9BranchSalvageRecall,
)

OUT = Path(__file__).with_name("results.json")
REPORT = Path(__file__).with_name("report.md")
PRED_ROOT = Path(r"D:\EDA\EDA-NetLine-Isolated\docs\network\lines\experiments")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gt_membership(target):
    result = {}
    for net_id, net in target.get("nets", {}).items():
        for component, pin in re.findall(r"([^(),\s]+)\.([^(),\s]+)", net.get("hyperGraph", "")):
            result[f"{component}.{pin}"] = net_id
    return result


def pair_set(partitions):
    return {tuple(sorted(pair)) for group in partitions for pair in itertools.combinations(group, 2)}


def gt_pair_set(target):
    result = set()
    for net in target.get("nets", {}).values():
        pins = re.findall(r"([^(),\s]+)\.([^(),\s]+)", net.get("hyperGraph", ""))
        members = [f"{component}.{pin}" for component, pin in pins]
        result.update(tuple(sorted(pair)) for pair in itertools.combinations(members, 2))
    return result


def main():
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite v8 experiment output")
    config = load_team_config(ROOT / "configs/team.json")
    topology = TopologyStageV3(network_versions=selected_versions())
    registry = {}

    class CaptureRegistry:
        def register(self, stage, name, cls):
            if stage == "wire":
                registry[name] = cls

    register_stages(CaptureRegistry())
    stages = {
        "v3_frame_guard": WireStageV3FrameGuard(),
        "v7_palette_corridor_recall": WireStageV7PaletteCorridorRecall(),
        "v8_baseline_anchored_recall": registry["v8_baseline_anchored_recall"](),
        "v9_branch_salvage_recall": registry["v9_branch_salvage_recall"](),
    }
    versions = list(stages)
    all_cov = {name: [] for name in versions}
    all_geometry_cov = {name: [] for name in versions}
    totals = {name: {"pixels": 0, "precision": [], "segments": 0, "geometry_precision": []}
              for name in versions}
    new_pairs = {name: set() for name in versions if name != "v3_frame_guard"}
    split_pairs = {name: set() for name in versions if name != "v3_frame_guard"}
    new_cross_gt = {name: set() for name in versions if name != "v3_frame_guard"}
    correct_pair_splits = {name: set() for name in versions if name != "v3_frame_guard"}
    samples = []
    image_hashes = []

    for case_id in range(1, 14):
        image_path = DATA / str(case_id) / f"{case_id}.png"
        image = read_image(image_path)
        height, width = image.shape[:2]
        result, diagnostics = saved_prediction(case_id)
        result_path = (Path(r"D:\EDA\EDA-NetLine-Isolated\docs\network\lines\experiments") /
                       ("real_image_ab_20261020_230608/pipeline_output_retry1/1" if case_id == 1 else
                        f"open_gt_baseline_20261024_161500_retry1/{case_id:04d}/pipeline_output/{case_id}"))
        target = json.loads((DATA / str(case_id) / f"{case_id}.json").read_text(encoding="utf-8"))
        edges = read_gt_edges(target, height)
        membership = gt_membership(target)
        true_pairs = gt_pair_set(target)
        context = PipelineContext(image_path.name, width, height, config, None)
        outputs, partitions, segments, stage_diagnostics = {}, {}, {}, {}
        for name, stage in stages.items():
            scene = make_scene(image, result, diagnostics)
            output = stage.run(image, scene, context)
            outputs[name] = output
            stage_diagnostics[name] = scene.diagnostics
            topology_scene = make_scene(image, result, diagnostics)
            topology.run(topology_scene, output, context)
            partitions[name] = pin_sets(topology_scene)
            segments[name] = [edge for net in topology_scene.nets for edge in net.segments]

        base_pairs = pair_set(partitions["v3_frame_guard"])
        for name in new_pairs:
            current_pairs = pair_set(partitions[name])
            new = current_pairs - base_pairs
            split = base_pairs - current_pairs
            new_pairs[name].update(new)
            split_pairs[name].update(split)
            new_cross_gt[name].update(
                pair for pair in new if pair[0] in membership and pair[1] in membership
                and membership[pair[0]] != membership[pair[1]]
            )
            correct_pair_splits[name].update(pair for pair in split if pair in true_pairs)

        gt_mask = np.zeros((height, width), np.uint8)
        for edge in edges:
            cv2.line(gt_mask, tuple(np.rint(edge["a_cv"]).astype(int)),
                     tuple(np.rint(edge["b_cv"]).astype(int)), 255, 1)
        near_gt = cv2.dilate(gt_mask, np.ones((11, 11), np.uint8)) > 0
        row = {"sample_id": f"{case_id:04d}", "gt_edge_count": len(edges), "versions": {}}
        for name, output in outputs.items():
            binary = output.mask > 0
            cov = [edge_coverage(mask_distance(output.mask), edge, 5) for edge in edges]
            all_cov[name].extend(cov)
            pixel_count = int(binary.sum())
            pixel_precision = float((binary & near_gt).sum() / max(1, pixel_count))
            totals[name]["pixels"] += pixel_count
            totals[name]["precision"].append(pixel_precision)

            geometry_mask = np.zeros((height, width), np.uint8)
            for a, b in segments[name]:
                cv2.line(geometry_mask, tuple(np.rint(a).astype(int)),
                         tuple(np.rint(b).astype(int)), 255, 1)
            geometry_cov = [edge_coverage(mask_distance(geometry_mask), edge, 5) for edge in edges]
            all_geometry_cov[name].extend(geometry_cov)
            geo_binary = geometry_mask > 0
            geometry_precision = float((geo_binary & near_gt).sum() / max(1, int(geo_binary.sum())))
            totals[name]["segments"] += len(segments[name])
            totals[name]["geometry_precision"].append(geometry_precision)
            candidate_guard = stage_diagnostics[name].get("wire_candidate_corridor_recall")
            row["versions"][name] = {
                "macro_gt_edge_coverage_5px": float(np.mean(cov)) if cov else None,
                "gt_edges_ge_80pct_5px": sum(value >= .8 for value in cov),
                "mask_pixels": pixel_count,
                "mask_pixel_precision_proxy5": pixel_precision,
                "connected_components": int(cv2.connectedComponents(binary.astype(np.uint8), 8)[0] - 1),
                "topology_net_count": len(partitions[name]),
                "topology_member_count": sum(map(len, partitions[name])),
                "topology_members_equal_v3": partitions[name] == partitions["v3_frame_guard"],
                "candidate_guard": candidate_guard,
                "netline_geometry": {
                    "segment_count": len(segments[name]),
                    "macro_gt_edge_coverage_5px": float(np.mean(geometry_cov)) if geometry_cov else None,
                    "gt_edges_ge_80pct_5px": sum(value >= .8 for value in geometry_cov),
                    "geometry_pixel_precision_proxy5": geometry_precision,
                },
            }
        samples.append(row)
        image_hashes.append({"sample_id": f"{case_id:04d}", "image": str(image_path),
                             "image_sha256": sha256(image_path),
                             "saved_component_pin_result": str(result_path / "result.json"),
                             "saved_component_pin_diagnostics": str(result_path / "diagnostics.json")})

    summaries = {}
    for name in versions:
        summaries[name] = {
            "gt_edges": len(all_cov[name]),
            "macro_gt_edge_coverage_5px": float(np.mean(all_cov[name])),
            "gt_edges_ge_80pct_5px": sum(value >= .8 for value in all_cov[name]),
            "mask_pixels": totals[name]["pixels"],
            "mask_pixel_precision_proxy5_sample_macro": float(np.mean(totals[name]["precision"])),
            "new_pin_pairs_vs_v3": len(new_pairs.get(name, ())),
            "new_cross_gt_net_pairs_vs_v3": len(new_cross_gt.get(name, ())),
            "v3_pairs_split": len(split_pairs.get(name, ())),
            "known_gt_correct_v3_pairs_split": len(correct_pair_splits.get(name, ())),
            "assembled_netline_geometry": {
                "segment_count": totals[name]["segments"],
                "macro_gt_edge_coverage_5px": float(np.mean(all_geometry_cov[name])),
                "gt_edges_ge_80pct_5px": sum(value >= .8 for value in all_geometry_cov[name]),
                "geometry_pixel_precision_proxy5_sample_macro": float(np.mean(totals[name]["geometry_precision"])),
            },
        }
        if name != "v3_frame_guard":
            summaries[name]["new_cross_gt_net_pairs"] = [list(pair) for pair in sorted(new_cross_gt[name])]
            summaries[name]["known_gt_correct_v3_pairs_split_list"] = [list(pair) for pair in sorted(correct_pair_splits[name])]

    data = {
        "experiment": "Wire v8 baseline-anchored palette candidate replay",
        "input": {
            "authorized_dataset_root": str(DATA),
            "sample_ids": [f"{i:04d}" for i in range(1, 14)],
            "saved_component_pin_outputs_reused": True,
            "component_pin_inference_rerun": False,
            "ground_truth_used_only_for_offline_diagnostics": True,
            "inputs": image_hashes,
        },
        "runtime": {"python": sys.version, "opencv": cv2.__version__,
                    "default_wire_version": json.loads((ROOT / "configs/wire_topology/current.json").read_text(encoding="utf-8"))["wire"],
                    "topology_versions": selected_versions(), "distance_tolerance_px": 5},
        "candidate": "v9_branch_salvage_recall",
        "method": "same v8 baseline/corridor anchored candidates; at multi-baseline candidate components, remove small windows around skeleton branch clusters, retain only image-derived fragments that touch exactly one v7 baseline component, then re-run the ordinary bridge guard; no candidate pixels are synthesized and the full v7 baseline is retained",
        "limitations": [
            "5px coverage and precision are local development diagnostics, not official scores.",
            "Topology checks can only identify cross-GT-net pairs among pins matched by exact identifiers; unmatched pins remain an evaluation blind spot.",
            "A candidate component attached to one baseline region can still alter connectivity at crossings.",
            "Branch-salvaged fragments are not proof of electrical connectivity.",
            "No independent holdout set was used; all 13 samples informed earlier iterations.",
            "Branch-cut salvage can discard legitimate junction pixels and cannot resolve whether two baseline regions belong to one electrical net.",
        ],
        "summaries": summaries,
        "pair_changes": {name: {"new_pairs": [list(pair) for pair in sorted(new_pairs[name])],
                                "split_pairs": [list(pair) for pair in sorted(split_pairs[name])],
                                "new_cross_gt_net_pairs": [list(pair) for pair in sorted(new_cross_gt[name])],
                                "known_gt_correct_pairs_split": [list(pair) for pair in sorted(correct_pair_splits[name])]}
                         for name in new_pairs},
        "samples": samples,
    }
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Wire v8 baseline-anchored recall replay", "",
             "Cached Component/Pin outputs were reused for authorized development samples 0001–0013. No Component/Pin inference was rerun. GT was read only for offline geometry and pin-pair diagnostics; it was not passed to Wire or Topology.", "",
             "## Aggregate comparison", "",
             "| Version | GT edge coverage @5px | GT edges >=80% | Mask pixels | Mask precision proxy | New cross-GT pairs | Known correct v3 pairs split | NetLine geometry coverage | NetLine segments |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in versions:
        s = summaries[name]
        g = s["assembled_netline_geometry"]
        lines.append(f"| {name} | {s['macro_gt_edge_coverage_5px']:.2%} | {s['gt_edges_ge_80pct_5px']}/744 | {s['mask_pixels']} | {s['mask_pixel_precision_proxy5_sample_macro']:.2%} | {s['new_cross_gt_net_pairs_vs_v3']} | {s['known_gt_correct_v3_pairs_split']} | {g['macro_gt_edge_coverage_5px']:.2%} | {g['segment_count']} |")
    lines += ["", "## Per-sample comparison", "",
              "| Sample | Edges | v3 cov | v7 cov | v8 cov | v9 cov | v7 >=80% | v8 >=80% | v9 >=80% | v8 px | v9 px | Salvaged px |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sample in samples:
        v = sample["versions"]
        guard = v["v9_branch_salvage_recall"]["candidate_guard"] or {}
        lines.append(f"| {sample['sample_id']} | {sample['gt_edge_count']} | {v['v3_frame_guard']['macro_gt_edge_coverage_5px']:.2%} | {v['v7_palette_corridor_recall']['macro_gt_edge_coverage_5px']:.2%} | {v['v8_baseline_anchored_recall']['macro_gt_edge_coverage_5px']:.2%} | {v['v9_branch_salvage_recall']['macro_gt_edge_coverage_5px']:.2%} | {v['v7_palette_corridor_recall']['gt_edges_ge_80pct_5px']} | {v['v8_baseline_anchored_recall']['gt_edges_ge_80pct_5px']} | {v['v9_branch_salvage_recall']['gt_edges_ge_80pct_5px']} | {v['v8_baseline_anchored_recall']['mask_pixels']} | {v['v9_branch_salvage_recall']['mask_pixels']} | {guard.get('branch_salvage_retained_pixels', 0)} |")
    lines += ["", "## Interpretation", "",
              "Prior production-stage trace on the same 13 samples measured 744/744 GT edges with >=80% proximity coverage in the raw palette union, 721/744 after line support, 307/744 after v7 candidate anchoring but before its bridge guard, and 167/744 after the guard. This locates the major remaining loss in candidate anchoring and whole-component bridge rejection, not primarily in raw color generation.",
              "",
              "These are geometry and connectivity diagnostics only. A higher coverage or fewer rejected pixels does not establish electrical correctness or official competition accuracy. The exact-pin-id limitation leaves unmatched-pin risks unmeasured.",
              "", "Default configuration remains `v3_frame_guard`. No production default or shared module was changed.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "results": str(OUT), "summaries": summaries,
                      "pair_changes": data["pair_changes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()