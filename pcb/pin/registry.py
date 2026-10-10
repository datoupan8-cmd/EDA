"""Pin-owned registrations. Peer and historical versions never share names."""
def register_stages(registry):
    from .localization.v3 import PinLocalizationStageV3
    from .localization.v4 import PinLocalizationStageV4
    from .localization.v5 import PinLocalizationStageV5
    from .localization.v6 import PinLocalizationStageV6
    from .localization.v7 import PinLocalizationStageV7
    from .localization.v8 import PinLocalizationStageV8
    from .localization.box_skeleton_experiment import BoxScanPreserveStage, BoxSkeletonMaskedStage, BoxSkeletonPreserveStage
    from .semantics.v3 import PinSemanticsStageV3
    from .semantics.v4 import PinSemanticsStageV4
    from .semantics.v5 import PinSemanticsStageV5
    from .semantics.v6 import PinSemanticsStageV6
    from .semantics.v7 import PinSemanticsStageV7
    from .semantics.v8 import PinSemanticsStageV8
    from .semantics.v9 import PinSemanticsStageV9
    from .peer_v4.localization_stage import PinLocalizationStageV4 as PeerLocalization
    from .peer_v4.semantics_stage import PinSemanticsStageV4 as PeerSemantics
    from .frozen_l39.stages import L39Localization, L39Semantics
    for name, implementation in [
        ("v3", PinLocalizationStageV3), ("v4", PinLocalizationStageV4),
        ("v5", PinLocalizationStageV5), ("v6", PinLocalizationStageV6),
        ("v7", PinLocalizationStageV7), ("v8", PinLocalizationStageV8),
        ("box_scan_preserve", BoxScanPreserveStage), ("box_skeleton_masked", BoxSkeletonMaskedStage),
        ("box_skeleton_preserve", BoxSkeletonPreserveStage),
        ("peer_v4", PeerLocalization), ("l39", L39Localization),
    ]:
        registry.register("pin_localization", name, implementation)
    for name, implementation in [
        ("v3", PinSemanticsStageV3), ("v4", PinSemanticsStageV4),
        ("v5", PinSemanticsStageV5), ("v6", PinSemanticsStageV6),
        ("v7", PinSemanticsStageV7), ("v8", PinSemanticsStageV8), ("v9", PinSemanticsStageV9),
        ("peer_v4", PeerSemantics), ("l39", L39Semantics),
    ]:
        registry.register("pin_semantics", name, implementation)
