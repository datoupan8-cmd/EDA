"""Wrapper around the unchanged Pin V3 semantic assignment."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from ...pin_semantics import assign_pin_semantics


class PinSemanticsStageV3:
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context) -> PinSemanticsOutput:
        pin_events = []
        for item, terminals in localization.terminals:
            pins, events = assign_pin_semantics(item, terminals, component.roles)
            item.pins = pins
            pin_events.extend({"component": item.key, **event} for event in events)
        diagnostics = {
            "mode": "image-only",
            "pipeline": context.config.legacy_pipeline_name,
            "pin_v3_frozen": True,
            "ocr_backend": getattr(context.ocr, "name", "rapidocr"),
            "ocr_diagnostics": dict(getattr(context.ocr, "last_diagnostics", {}) or {}),
            "terminal_candidates": sum(len(item.pins) for item in component.components),
            "exportable_pins": sum(pin.exportable for item in component.components for pin in item.pins),
            "pin_events": pin_events,
            "keypoint_provider": getattr(context.keypoint_provider, "name", "disabled"),
        }
        return PinSemanticsOutput(component.components, diagnostics)
