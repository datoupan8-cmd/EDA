"""P6-S1 experimental Pin Semantics Stage."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from ...pin_semantics_v4 import assign_pin_semantics_v4


class PinSemanticsStageV4:
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context) -> PinSemanticsOutput:
        pin_events = []
        for item, terminals in localization.terminals:
            pins, events = assign_pin_semantics_v4(item, terminals, component.roles)
            item.pins = pins
            pin_events.extend({"component": item.key, **event} for event in events)
        diagnostics = {
            "mode": "image-only",
            "pipeline": context.config.legacy_pipeline_name,
            "pin_semantics": "v4_p6_s1_numeric_context_experiment",
            "ocr_backend": getattr(context.ocr, "name", "rapidocr"),
            "ocr_diagnostics": dict(getattr(context.ocr, "last_diagnostics", {}) or {}),
            "terminal_candidates": sum(len(item.pins) for item in component.components),
            "exportable_pins": sum(pin.exportable for item in component.components for pin in item.pins),
            "contextual_numeric_reclassifications": sum(event.get("number_role_reclassified", False)
                                                         for event in pin_events),
            "pin_events": pin_events,
            "keypoint_provider": getattr(context.keypoint_provider, "name", "disabled"),
        }
        return PinSemanticsOutput(component.components, diagnostics)
