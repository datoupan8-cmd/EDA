"""Unchanged P3.1 localization plus an explicit image handoff for local OCR."""
from __future__ import annotations

from .v7 import PinLocalizationStageV7


class PinLocalizationStageV8:
    """Geometry adapter, not a new detector; every V7 candidate stays unchanged."""

    def __init__(self, base_stage=None):
        self.base_stage = base_stage or PinLocalizationStageV7()

    def run(self, image, component, context):
        output = self.base_stage.run(image, component, context)
        output.debug = dict(output.debug)
        output.debug["pin_local_ocr_image"] = image
        return output
