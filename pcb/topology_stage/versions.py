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


class _SplitStage:
    def __init__(self, network_versions=None):
        from ..network_shared.selection import selected_versions, backends
        self.network_versions = dict(selected_versions() if network_versions is None else network_versions)
        self.hypergraph, self.lines = backends(self.network_versions)


class TopologyStageV2(_SplitStage):
    def run(self, scene, wire: WireStageOutput, context) -> TopologyStageOutput:
        mode = "directional" if context.config.variant != "baseline" else "fixed"
        from ..network_shared.composite import build_networks
        skeleton = build_networks(
            scene,
            wire.mask,
            mode,
            merge_labels=context.config.enable_flying_labels,
            hypergraph=self.hypergraph,
            lines=self.lines,
        )
        return TopologyStageOutput(skeleton)


class TopologyStageV3(_SplitStage):
    def run(self, scene, wire: WireStageOutput, context) -> TopologyStageOutput:
        skeleton = build_topology_v3(
            scene,
            wire.mask,
            merge_labels=context.config.enable_flying_labels,
            hypergraph=self.hypergraph,
            lines=self.lines,
        )
        return TopologyStageOutput(skeleton)
