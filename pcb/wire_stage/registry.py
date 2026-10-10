"""Wire-owned versions; the retained frame guard is explicitly selectable."""
def register_stages(registry):
    from .versions import WireStageV1, WireStageV2, WireStageV3
    from .retained import WireStageV3FrameGuard
    for name, cls in [("v1", WireStageV1), ("v2", WireStageV2), ("v3", WireStageV3),
                      ("v3_frame_guard", WireStageV3FrameGuard)]:
        registry.register("wire", name, cls)
