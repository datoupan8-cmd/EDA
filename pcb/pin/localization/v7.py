"""P3.1 experimental localization: P3 plus resumed-wire rescue."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput
from .continuity_v7 import verify_outward_continuity_v7
from .directional_v5 import build_raw_ink
from .v5 import PinLocalizationStageV5


class PinLocalizationStageV7:
    """Keep V3 candidates; recheck only P2 additions using image evidence."""

    def __init__(self, base_stage=None) -> None:
        self.base_stage = base_stage or PinLocalizationStageV5()

    def run(self, image, component: ComponentStageOutput, context) -> PinLocalizationOutput:
        baseline = self.base_stage.run(image, component, context)
        ink = build_raw_ink(image)
        rows = []
        events = []
        for item, terminals in baseline.terminals:
            kept = []
            for terminal in terminals:
                if terminal.get("method") != "directional_boundary_support":
                    kept.append(terminal)
                    continue
                accepted, evidence = verify_outward_continuity_v7(ink, terminal, image.shape)
                events.append({"component": item.key, "side": terminal["side"],
                               "base": terminal["base"], **evidence})
                if accepted:
                    kept.append(terminal)
            rows.append((item, kept))
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update({
            "pin_localization": "v7_p3_1_resumed_wire_experiment",
            "continuity_checked": len(events),
            "continuity_kept": sum(event["accepted"] for event in events),
            "continuity_events": events,
        })
        return PinLocalizationOutput(rows, diagnostics, dict(baseline.debug))
