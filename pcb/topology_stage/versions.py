"""Thin wrappers around all existing Topology implementations."""
from __future__ import annotations

from ..core.interfaces import TopologyStageOutput, WireStageOutput
from ..topology import build_topology
from ..topology_v2 import build_topology_v2
from ..topology_v3 import build_topology_v3


class TopologyStageV1:
    def run(self, scene, wire: WireStageOutput, context) -> TopologyStageOutput:
        skeleton = build_topology(scene, wire.mask, context.config.variant)
        return TopologyStageOutput(skeleton)


class TopologyStageV2:
    def run(self, scene, wire: WireStageOutput, context) -> TopologyStageOutput:
        mode = "directional" if context.config.variant != "baseline" else "fixed"
        skeleton = build_topology_v2(
            scene,
            wire.mask,
            mode,
            merge_labels=context.config.enable_flying_labels,
        )
        return TopologyStageOutput(skeleton)


class TopologyStageV3:
    def run(self, scene, wire: WireStageOutput, context) -> TopologyStageOutput:
        skeleton = build_topology_v3(
            scene,
            wire.mask,
            merge_labels=context.config.enable_flying_labels,
        )
        return TopologyStageOutput(skeleton)
