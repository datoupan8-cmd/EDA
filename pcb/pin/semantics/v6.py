"""P6.1 experimental side-ordered joint Pin Semantics Stage."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from ...pin_semantics_v6 import assign_pin_semantics_v6


class PinSemanticsStageV6:
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context) -> PinSemanticsOutput:
        events = []
        for item, terminals in localization.terminals:
            pins, rows = assign_pin_semantics_v6(item, terminals, component.roles)
            item.pins = pins
            events.extend({"component": item.key, **row} for row in rows)
        diagnostics = {
            "mode": "image-only", "pipeline": context.config.legacy_pipeline_name,
            "pin_semantics": "v6_p6_1_side_ordered_joint_alignment_experiment",
            "ocr_backend": getattr(context.ocr, "name", "rapidocr"),
            "ocr_diagnostics": dict(getattr(context.ocr, "last_diagnostics", {}) or {}),
            "terminal_candidates": sum(len(item.pins) for item in component.components),
            "exportable_pins": sum(pin.exportable for item in component.components for pin in item.pins),
            "contextual_numeric_reclassifications": sum(row.get("number_role_reclassified", False) for row in events),
            "ordered_joint_rows": sum(row.get("row_has_number_and_name", False) for row in events),
            "semantic_abstentions": sum(bool(row.get("semantic_abstention")) for row in events),
            "pin_events": events,
            "keypoint_provider": getattr(context.keypoint_provider, "name", "disabled"),
        }
        return PinSemanticsOutput(component.components, diagnostics)

