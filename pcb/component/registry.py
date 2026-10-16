"""Component-owned registrations. Add new component versions only here."""
def register_stages(registry):
    from .v3 import ComponentStageV3
    from .latest.stage import ComponentStageV4
    from .baseline_v41.stage import ComponentStageV4 as BaselineComponent
    registry.register("component", "v3", ComponentStageV3)
    registry.register("component", "v4", ComponentStageV4)
    registry.register("component", "peer_latest", ComponentStageV4)
    registry.register("component", "baseline_v41", BaselineComponent)
    from .contextual_values import ComponentStageContextualValues
    from .alternative_ocr import TextStagePPOcrV5
    registry.register("component", "contextual_values_v2", ComponentStageContextualValues)
    registry.register("text", "ppocrv5_mobile", TextStagePPOcrV5)
    from .local_joint import ComponentStageLocalOCR, ComponentStageLocalJoint, ComponentStagePrintedGroups, ComponentStageValueGroups
    registry.register("component", "local_ocr_v1", ComponentStageLocalOCR)
    registry.register("component", "local_joint_v1", ComponentStageLocalJoint)
    registry.register("component", "printed_groups_v1", ComponentStagePrintedGroups)
    registry.register("component", "printed_value_groups_v1", ComponentStageValueGroups)
    from .value_groups_v2 import ComponentStageGuardedValues, ComponentStageOrderedValues
    registry.register("component", "guarded_value_groups_v2", ComponentStageGuardedValues)
    registry.register("component", "ordered_value_groups_v2", ComponentStageOrderedValues)
