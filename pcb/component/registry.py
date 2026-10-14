"""Component-owned registrations. Add new component versions only here."""
def register_stages(registry):
    from .v3 import ComponentStageV3
    from .latest.stage import ComponentStageV4
    from .baseline_v41.stage import ComponentStageV4 as BaselineComponent
    registry.register("component", "v3", ComponentStageV3)
    registry.register("component", "v4", ComponentStageV4)
    registry.register("component", "peer_latest", ComponentStageV4)
    registry.register("component", "baseline_v41", BaselineComponent)
    from .text_enrichment import ComponentStageTextEnriched
    registry.register("component", "text_enriched_v1", ComponentStageTextEnriched)
    from .contextual_values import ComponentStageContextualValues
    registry.register("component", "contextual_values_v2", ComponentStageContextualValues)
    from .alternative_rcnn import ComponentStageFasterRcnn
    from .alternative_ocr import TextStagePPOcrV5
    registry.register("component", "faster_rcnn_v1", ComponentStageFasterRcnn)
    registry.register("text", "ppocrv5_mobile", TextStagePPOcrV5)
