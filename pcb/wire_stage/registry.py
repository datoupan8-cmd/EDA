"""Wire-owned versions; the retained frame guard is explicitly selectable."""
def register_stages(registry):
    from .versions import WireStageV1, WireStageV2, WireStageV3
    from .retained import (WireStageV3FrameGuard, WireStageV4PaletteExpandedFrameGuard,
                           WireStageV5PaletteRecallFrameGuard, WireStageV6PaletteBridgeGuard,
                           WireStageV7PaletteCorridorRecall, WireStageV8BaselineAnchoredRecall,
                           WireStageV9BranchSalvageRecall)
    for name, cls in [("v1", WireStageV1), ("v2", WireStageV2), ("v3", WireStageV3),
                      ("v3_frame_guard", WireStageV3FrameGuard),
                      ("v4_palette_expanded", WireStageV4PaletteExpandedFrameGuard),
                      ("v5_palette_recall", WireStageV5PaletteRecallFrameGuard),
                      ("v6_palette_bridge_guard", WireStageV6PaletteBridgeGuard),
                      ("v7_palette_corridor_recall", WireStageV7PaletteCorridorRecall),
                      ("v8_baseline_anchored_recall", WireStageV8BaselineAnchoredRecall),
                      ("v9_branch_salvage_recall", WireStageV9BranchSalvageRecall)]:
        registry.register("wire", name, cls)
