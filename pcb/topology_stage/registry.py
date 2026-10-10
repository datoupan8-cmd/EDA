"""Topology-owned versions."""
def register_stages(registry):
    from .versions import TopologyStageV1, TopologyStageV2, TopologyStageV3
    for name, cls in [("v1", TopologyStageV1), ("v2", TopologyStageV2), ("v3", TopologyStageV3)]:
        registry.register("topology", name, cls)
