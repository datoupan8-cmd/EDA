"""P6.5: unchanged formal V3 semantics plus a D/LED name-only correction."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from .diode_names import apply_diode_pinname_mapping
from .v3 import PinSemanticsStageV3


class PinSemanticsStageV9(PinSemanticsStageV3):
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context) -> PinSemanticsOutput:
        output = super().run(component, localization, context)
        offset = 0
        changes = []
        events = output.diagnostics["pin_events"]
        for item, terminals in localization.terminals:
            end = offset + len(terminals)
            rows = events[offset:end]
            for change in apply_diode_pinname_mapping(item.type, item.pins, rows):
                changes.append({"component": item.key, "type": item.type, **change})
            offset = end
        if offset != len(events):
            raise ValueError("Component/event cardinality mismatch in diode name Stage")
        output.diagnostics.update({
            "pin_v3_frozen": False,
            "pin_geometry_frozen": True,
            "pin_semantics": "v9_p6_5_diode_pinname_only_experiment",
            "diode_pinname_mapping": {
                "rule": {"1": "K", "2": "A"},
                "polarity_recognition_enabled": False,
                "changed_pin_count": len(changes),
                "changes": changes,
            },
        })
        return output

