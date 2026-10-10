"""P1-2.3 Pin Localization V4: V3 candidates with tip-only remapping."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput
from .tip_placement_v4 import place_boundary_tip
from .v3 import PinLocalizationStageV3


class PinLocalizationStageV4:
    """Preserve V3 candidates and remap only boundary-wire tip extension."""

    def __init__(self, base_stage=None) -> None:
        self.base_stage = base_stage or PinLocalizationStageV3()

    def run(self, image, component: ComponentStageOutput, context) -> PinLocalizationOutput:
        baseline = self.base_stage.run(image, component, context)
        events = []
        for item, terminals in baseline.terminals:
            for index, terminal in enumerate(terminals):
                if terminal.get("method") != "boundary_wire_support":
                    continue
                old_base = terminal.get("base")
                old_side = terminal.get("side")
                old_method = terminal.get("method")
                old_tip = terminal.get("tip")
                new_tip, event = place_boundary_tip(terminal, image.shape)
                terminal["tip"] = new_tip
                if terminal.get("base") != old_base:
                    raise AssertionError("P1-2.3 changed terminal base")
                if terminal.get("side") != old_side:
                    raise AssertionError("P1-2.3 changed terminal side")
                if terminal.get("method") != old_method:
                    raise AssertionError("P1-2.3 changed terminal method")
                events.append({
                    "component": item.key,
                    "candidate_index": index,
                    "old_tip": old_tip,
                    **event,
                })
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update({
            "pin_localization": "v4_p1_2_3_tip_placement",
            "tip_placement_strategy": "scale_aware",
            "tip_placement_events": events,
            "candidate_count": sum(len(rows) for _, rows in baseline.terminals),
        })
        return PinLocalizationOutput(
            terminals=baseline.terminals,
            diagnostics=diagnostics,
            debug=dict(baseline.debug),
        )

