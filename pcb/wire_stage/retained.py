"""Exact V3 + F wire sequence used by the validated L39 combination."""
from .versions import WireStageV3
from .frame_guard import suppress_box_frames


class WireStageV3FrameGuard:
    def run(self, image, scene, context):
        return suppress_box_frames(image, scene, WireStageV3().run(image, scene, context))
