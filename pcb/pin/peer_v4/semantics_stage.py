"""Local, unique pin-number assignment with explicit internal unknown IDs."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from .semantics import assign_pin_semantics_v4, assign_role_owners


class PinSemanticsStageV4:
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context) -> PinSemanticsOutput:
        pin_events = []
        owned_roles, owner_events = assign_role_owners(localization.terminals, component.roles)
        for index, (item, terminals) in enumerate(localization.terminals):
            pins, events = assign_pin_semantics_v4(item, terminals, owned_roles[index])
            item.pins = pins
            pin_events.extend({"component": item.key, **event} for event in events)
        diagnostics = {
            **localization.diagnostics,
            "mode": "image-only", "pipeline": context.config.legacy_pipeline_name,
            "pin_semantics_version": "v4", "pin_v3_frozen": False,
            "ocr_backend": getattr(context.ocr, "name", "rapidocr"),
            "ocr_diagnostics": dict(getattr(context.ocr, "last_diagnostics", {}) or {}),
            "terminal_candidates": sum(len(item.pins) for item in component.components),
            "exportable_pins": sum(pin.exportable for item in component.components for pin in item.pins),
            "unknown_pins_retained": sum(not pin.exportable for item in component.components for pin in item.pins),
            "promoted_numeric_values": sum(bool(event.get("promoted_numeric_value")) for event in pin_events),
            "reclassified_signal_labels": sum(bool(event.get("signal_label_reclassified")) for event in pin_events),
            "pin_token_owners": owner_events,
            "shared_pin_token_candidates": sum(event["candidate_component_count"] > 1 for event in owner_events),
            "pin_events": pin_events,
            "keypoint_provider": getattr(context.keypoint_provider, "name", "disabled"),
        }
        return PinSemanticsOutput(component.components, diagnostics)
