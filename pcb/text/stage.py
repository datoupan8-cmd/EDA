"""Wrapper for the configured, unchanged OCR object."""
from __future__ import annotations

from ..core.interfaces import TextStageOutput


class CurrentTextStage:
    def run(self, image, context) -> TextStageOutput:
        texts = context.ocr.recognize(image)
        return TextStageOutput(texts, dict(getattr(context.ocr, "last_diagnostics", {}) or {}))
