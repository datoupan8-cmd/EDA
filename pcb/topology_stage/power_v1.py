"""Opt-in integration: frozen V3 topology plus supported power terminals."""
from ..core.interfaces import TopologyStageOutput
from ..network_shared.composite import assemble, validate_plan
from ..network_shared.contracts import snapshot
from ..network_shared.graph import build_graph
from .power_terminals import attach_power_terminals
from .versions import TopologyStageV3


class PowerTerminalTopology(TopologyStageV3):
    def run(self, scene, wire, context):
        graph = build_graph(wire.mask)
        logical = self.hypergraph.run(snapshot(scene), graph, mode='directional',
                                      merge_labels=context.config.enable_flying_labels)
        validate_plan(logical.plan, snapshot(scene), graph)
        plan, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        validate_plan(plan, snapshot(scene), graph)
        scene.nets = assemble(plan, self.lines.run(graph, plan))
        scene.diagnostics.update(logical.diagnostics)
        scene.diagnostics.update(topology='v3_power_terminals_v1', power_terminals=trace,
                                 nets=len(scene.nets), net_edges=sum(len(n.segments) for n in scene.nets))
        return TopologyStageOutput(graph.skeleton.copy())
