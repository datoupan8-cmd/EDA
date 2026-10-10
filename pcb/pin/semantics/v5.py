"""P6-S2 experimental side-aware global Pin Semantics Stage."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from ...pin_semantics_v5 import assign_pin_semantics_v5


class PinSemanticsStageV5:
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context) -> PinSemanticsOutput:
        events = []
        for item, terminals in localization.terminals:
            pins, rows = assign_pin_semantics_v5(item, terminals, component.roles)
            item.pins = pins
            events.extend({"component": item.key, **row} for row in rows)
        diagnostics = {
            "mode": "image-only", "pipeline": context.config.legacy_pipeline_name,
            "pin_semantics": "v5_p6_s2_side_aware_global_assignment_experiment",
            "ocr_backend": getattr(context.ocr, "name", "rapidocr"),
            "ocr_diagnostics": dict(getattr(context.ocr, "last_diagnostics", {}) or {}),
            "terminal_candidates": sum(len(item.pins) for item in component.components),
            "exportable_pins": sum(pin.exportable for item in component.components for pin in item.pins),
            "contextual_numeric_reclassifications": sum(row.get("number_role_reclassified", False) for row in events),
            "pin_events": events,
            "keypoint_provider": getattr(context.keypoint_provider, "name", "disabled"),
        }
        return PinSemanticsOutput(component.components, diagnostics)
