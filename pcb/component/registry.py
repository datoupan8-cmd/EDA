"""Component-owned registrations. Add new component versions only here."""
def register_stages(registry):
    from .v3 import ComponentStageV3
    from .latest.stage import ComponentStageV4
    from .baseline_v41.stage import ComponentStageV4 as BaselineComponent
    registry.register("component", "v3", ComponentStageV3)
    registry.register("component", "v4", ComponentStageV4)
    registry.register("component", "peer_latest", ComponentStageV4)
    registry.register("component", "baseline_v41", BaselineComponent)
