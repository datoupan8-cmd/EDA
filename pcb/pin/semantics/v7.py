"""P6.3 experimental semantics Stage with directional-candidate gating."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from ...pin_semantics_v7 import assign_pin_semantics_v7


class PinSemanticsStageV7:
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context) -> PinSemanticsOutput:
        events = []
        for item, terminals in localization.terminals:
            pins, rows = assign_pin_semantics_v7(item, terminals, component.roles)
            item.pins = pins
            events.extend({"component": item.key, **row} for row in rows)
        diagnostics = {
            "mode": "image-only",
            "pipeline": context.config.legacy_pipeline_name,
            "pin_semantics": "v7_p6_3_directional_candidate_confidence_gate_experiment",
            "ocr_backend": getattr(context.ocr, "name", "rapidocr"),
            "ocr_diagnostics": dict(getattr(context.ocr, "last_diagnostics", {}) or {}),
            "terminal_candidates": sum(len(item.pins) for item in component.components),
            "exportable_pins": sum(pin.exportable for item in component.components for pin in item.pins),
            "contextual_numeric_reclassifications": sum(row.get("number_role_reclassified", False) for row in events),
            "ordered_joint_rows": sum(row.get("row_has_number_and_name", False) for row in events),
            "directional_gate_checked": sum(row.get("directional_gate_applied", False) for row in events),
            "directional_gate_passed": sum(row.get("directional_gate_passed", False) for row in events),
            "directional_gate_abstentions": sum(
                row.get("semantic_abstention") == "directional_candidate_low_confidence" for row in events
            ),
            "semantic_abstentions": sum(bool(row.get("semantic_abstention")) for row in events),
            "pin_events": events,
            "keypoint_provider": getattr(context.keypoint_provider, "name", "disabled"),
        }
        return PinSemanticsOutput(component.components, diagnostics)
