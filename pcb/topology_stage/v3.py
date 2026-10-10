"""V3 topology policy around the validated V2 graph implementation."""
from __future__ import annotations
from .v2 import build_topology_v2


def build_topology_v3(scene, mask, merge_labels=True):
    skeleton = build_topology_v2(scene, mask, mode="directional", merge_labels=merge_labels)
    scene.diagnostics.update({
        "topology": "v3_component_barrier_directional",
        "same_cc_union_all_pins": False,
        "within_cc_policy": "skeleton paths joined only at electrical junction decisions",
        "crossover_policy": "four-arm crossings continue straight unless dot evidence",
        "short_gap_policy": "collinear outward endpoints within stroke-adaptive radius",
    })
    return skeleton
