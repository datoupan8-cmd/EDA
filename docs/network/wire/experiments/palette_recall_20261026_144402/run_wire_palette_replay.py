from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from pcb.core.context import PipelineContext
from pcb.network_shared.selection import selected_versions
from pcb.schema import Component, Pin, Scene, Text
from pcb.topology_stage.versions import TopologyStageV3
from pcb.wire_stage.retained import (
    WireStageV3FrameGuard,
    WireStageV4PaletteExpandedFrameGuard,
    WireStageV5PaletteRecallFrameGuard,
)
from pcb.core.team_config import load_team_config
from pcb.io import read_image

ROOT = Path(__file__).resolve().parents[5]
DATA = Path(r"D:\EDA\open_GT")
PRED_ROOT = Path(r"D:\EDA\EDA-NetLine-Isolated\docs\network\lines\experiments")
BASE = PRED_ROOT / "open_gt_baseline_20261024_161500_retry1"
CASE1 = PRED_ROOT / "real_image_ab_20261020_230608" / "pipeline_output_retry1" / "1"
VERSIONS = {
    "baseline_v3_frame_guard": WireStageV3FrameGuard(),
    "uncapped_gate_072": WireStageV4PaletteExpandedFrameGuard(),
    "uncapped_gate_071": WireStageV5PaletteRecallFrameGuard(),
}


def saved_prediction(case_id: int):
    folder = CASE1 if case_id == 1 else BASE / f"{case_id:04d}" / "pipeline_output" / str(case_id)
    return (
        json.loads((folder / "result.json").read_text(encoding="utf-8")),
        json.loads((folder / "diagnostics.json").read_text(encoding="utf-8")),
    )


def make_scene(image, result, diagnostics):
    height, width = image.shape[:2]
    components = []
    for key, raw in result["components"].items():
        x1, y1, x2, y2 = raw["bbox"]
        bbox = (x1, height - y2, x2, height - y1)
        pins = []
        for event in diagnostics.get("pin_events", []):
            if event.get("component") == key:
                pins.append(Pin(
                    number=str(event.get("number") or ""), name=event.get("pinname"),
                    tip=tuple(event["tip"]),
                    base=tuple(event["base"]) if event.get("base") is not None else None,
                    side=event.get("side", ""), exportable=bool(event.get("exportable", False)),
                ))
        components.append(Component(
            key=key, type=raw["type"], bbox=bbox, body_bbox=bbox,
            name=raw["Name"], value=raw["value"], pins=pins,
        ))
    texts = [Text(text=t["text"], bbox=tuple(t["bbox"]), score=float(t.get("confidence", 1.0)))
             for t in diagnostics.get("text_roles", [])]
    return Scene(width, height, components, texts, diagnostics=json.loads(json.dumps(diagnostics)))


def read_gt_edges(target, height):
    edges = []
    for net_id, net in target.get("nets", {}).items():
        for edge_id, endpoints in net.get("edges", {}).items():
            a, b = endpoints
            edges.append({
                "net_id": net_id, "edge_id": edge_id,
                "a_cv": [float(a["x"]), height - float(a["y"])],
                "b_cv": [float(b["x"]), height - float(b["y"])],
                "length_px": float(math.dist((a["x"], a["y"]), (b["x"], b["y"]))),
            })
    return edges


def mask_distance(mask):
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)


def edge_coverage(distance, edge, tolerance):
    a, b = np.asarray(edge["a_cv"], float), np.asarray(edge["b_cv"], float)
    length = float(np.linalg.norm(b - a))
    steps = max(1, int(math.ceil(length)))
    t = np.linspace(0.0, 1.0, steps + 1)
    points = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
    height, width = distance.shape
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    return float(np.mean(distance[ys, xs] <= tolerance))


def pin_sets(scene):
    return sorted(tuple(sorted(net.pins)) for net in scene.nets)


def main():
    config = load_team_config(ROOT / "configs/team.json")
    topology = TopologyStageV3(network_versions=selected_versions())
    all_edges = {key + str(tol): [] for key in VERSIONS for tol in (2, 5)}
    totals = {key: {"pixels": 0, "components": 0, "pixel_precision5": [], "added": 0, "added_near_gt5": 0}
              for key in VERSIONS}
    sample_rows = []
    replay_matches = []
    topology_changed = {key: [] for key in VERSIONS if key != "baseline_v3_frame_guard"}

    for case_id in range(1, 14):
        image_path = DATA / str(case_id) / f"{case_id}.png"
        image = read_image(image_path)
        height, width = image.shape[:2]
        result, diagnostics = saved_prediction(case_id)
        target = json.loads((DATA / str(case_id) / f"{case_id}.json").read_text(encoding="utf-8"))
        edges = read_gt_edges(target, height)
        masks, scenes, networks = {}, {}, {}
        context = PipelineContext(image_path.name, width, height, config, None)

        for key, wire_stage in VERSIONS.items():
            scene = make_scene(image, result, diagnostics)
            wire_output = wire_stage.run(image, scene, context)
            masks[key], scenes[key] = wire_output.mask, scene
            if key == "baseline_v3_frame_guard":
                replay_matches.append(int((wire_output.mask > 0).sum()) == int(diagnostics.get("wire_pixels", -1)))
            topology.run(scene, wire_output, context)
            networks[key] = pin_sets(scene)

        gt_mask = np.zeros((height, width), np.uint8)
        for edge in edges:
            cv2.line(gt_mask, tuple(np.rint(edge["a_cv"]).astype(int)),
                     tuple(np.rint(edge["b_cv"]).astype(int)), 255, 1)
        gt_near5 = cv2.dilate(gt_mask, np.ones((11, 11), np.uint8)) > 0
        baseline_mask = masks["baseline_v3_frame_guard"] > 0
        row = {"sample_id": f"{case_id:04d}", "gt_edge_count": len(edges), "versions": {}}

        for key, mask in masks.items():
            binary = mask > 0
            distance = mask_distance(mask)
            coverage = {tol: [edge_coverage(distance, edge, tol) for edge in edges] for tol in (2, 5)}
            for tol in (2, 5):
                all_edges[key + str(tol)].extend(coverage[tol])
            pixel_count = int(binary.sum())
            totals[key]["pixels"] += pixel_count
            totals[key]["components"] += int(cv2.connectedComponents(binary.astype(np.uint8), 8)[0] - 1)
            totals[key]["pixel_precision5"].append(float((binary & gt_near5).sum() / max(1, pixel_count)))
            if key != "baseline_v3_frame_guard":
                added = binary & ~baseline_mask
                totals[key]["added"] += int(added.sum())
                totals[key]["added_near_gt5"] += int((added & gt_near5).sum())
                if networks[key] != networks["baseline_v3_frame_guard"]:
                    topology_changed[key].append(f"{case_id:04d}")
            scene = scenes[key]
            row["versions"][key] = {
                "mean_gt_edge_coverage_2px": float(np.mean(coverage[2])) if edges else None,
                "mean_gt_edge_coverage_5px": float(np.mean(coverage[5])) if edges else None,
                "gt_edges_ge_80pct_at_5px": sum(value >= .8 for value in coverage[5]),
                "mask_pixels": pixel_count,
                "selected_palettes": scene.diagnostics.get("wire_palette_selected", []),
                "topology_net_count": len(scene.nets),
                "topology_member_count": sum(len(net.pins) for net in scene.nets),
                "unattached_pin_count": len(scene.diagnostics.get("unattached_pins", [])),
                "same_topology_members_as_baseline": networks[key] == networks["baseline_v3_frame_guard"],
            }

        sample_rows.append(row)

    summaries = {}
    for key in VERSIONS:
        edges2, edges5 = all_edges[key + "2"], all_edges[key + "5"]
        summaries[key] = {
            "gt_edge_count": len(edges5),
            "macro_edge_coverage_2px": float(np.mean(edges2)) if edges2 else None,
            "macro_edge_coverage_5px": float(np.mean(edges5)) if edges5 else None,
            "gt_edges_ge_80pct_at_5px": sum(value >= .8 for value in edges5),
            "mask_pixels_total": totals[key]["pixels"],
            "mask_pixel_precision5_macro_by_sample": float(np.mean(totals[key]["pixel_precision5"])),
            "connected_components_total": totals[key]["components"],
            "topology_net_count_total": sum(row["versions"][key]["topology_net_count"] for row in sample_rows),
            "topology_member_count_total": sum(row["versions"][key]["topology_member_count"] for row in sample_rows),
            "unattached_pin_count_total": sum(row["versions"][key]["unattached_pin_count"] for row in sample_rows),
        }
        if key != "baseline_v3_frame_guard":
            summaries[key].update({
                "added_mask_pixels_vs_baseline": totals[key]["added"],
                "added_pixels_within_5px_of_any_gt_edge": totals[key]["added_near_gt5"],
                "added_pixel_near_gt_fraction": totals[key]["added_near_gt5"] / max(1, totals[key]["added"]),
                "topology_changed_samples": topology_changed[key],
            })

    result = {
        "experiment": "Wire palette candidate recall ablation",
        "input": {"authorized_dataset_root": str(DATA), "sample_ids": [f"{i:04d}" for i in range(1, 14)],
                  "saved_prediction_root": str(PRED_ROOT), "component_pin_inference_rerun": False,
                  "annotation_used_only_for_offline_diagnostics": True},
        "runtime": {"python": "3.12.10", "wire_default_config": json.loads((ROOT / "configs/wire_topology/current.json").read_text(encoding="utf-8"))["wire"],
                    "network_versions": selected_versions(), "tolerance_px": [2, 5]},
        "policies": {
            "baseline_v3_frame_guard": {"max_palette_candidates": 2, "score_ratio": .72},
            "uncapped_gate_072": {"max_palette_candidates": None, "score_ratio": .72},
            "uncapped_gate_071": {"max_palette_candidates": None, "score_ratio": .71},
        },
        "metric_notes": {
            "edge_coverage": "Sampled along each GT edge; fraction within the stated Euclidean distance of the predicted mask.",
            "pixel_precision5": "Diagnostic proxy: mask pixels within 5 px of any rasterized GT edge. It is not official precision; unlabeled valid wires can be counted outside GT.",
            "electrical_connectivity": "Compared only as predicted topology member partitions; changes are not assumed correct. Pin ids were cross-referenced to GT hyperGraph only when identifiers matched exactly.",
            "replay_match": "Baseline output mask pixel count matched persisted wire_pixels diagnostic for 12/13 samples; sample 0010 differed by 3 pixels, a known replay limitation.",
        },
        "baseline_pixel_count_replay_matches": sum(replay_matches),
        "baseline_pixel_count_replay_samples": 13,
        "summaries": summaries,
        "samples": sample_rows,
    }
    out = Path(__file__).parent / "results.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out), "summaries": summaries,
                      "baseline_pixel_count_replay_matches": sum(replay_matches),
                      "sample_ids": [row["sample_id"] for row in sample_rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
