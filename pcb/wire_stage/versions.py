"""Thin wrappers around all existing Wire implementations."""
from __future__ import annotations

from ..core.interfaces import WireStageOutput
from ..wire import extract_wire
from ..wire_v2 import extract_wire_v2
from ..wire_v3 import extract_wire_v3


class WireStageV1:
    def run(self, image, scene, context) -> WireStageOutput:
        return WireStageOutput(extract_wire(image, scene))


class WireStageV2:
    def run(self, image, scene, context) -> WireStageOutput:
        mask, color_debug, suppressed = extract_wire_v2(image, scene)
        return WireStageOutput(mask, color_debug, suppressed)


class WireStageV3:
    def run(self, image, scene, context) -> WireStageOutput:
        mask, color_debug, suppressed, corridor = extract_wire_v3(image, scene)
        return WireStageOutput(mask, color_debug, suppressed, corridor)
