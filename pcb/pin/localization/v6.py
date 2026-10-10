"""Experimental P3 continuity verification of P2's new candidates."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput
from .continuity_v6 import verify_outward_continuity
from .directional_v5 import build_raw_ink
from .v5 import PinLocalizationStageV5


class PinLocalizationStageV6:
    """Filter only P2 additions; preserve every V3 terminal bit-for-bit."""

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
                accepted, evidence = verify_outward_continuity(ink, terminal, image.shape)
                events.append({"component": item.key, "side": terminal["side"],
                               "base": terminal["base"], **evidence})
                if accepted:
                    kept.append(terminal)
            rows.append((item, kept))
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update({
            "pin_localization": "v6_p3_outward_continuity_experiment",
            "continuity_checked": len(events),
            "continuity_kept": sum(event["accepted"] for event in events),
            "continuity_events": events,
        })
        return PinLocalizationOutput(rows, diagnostics, dict(baseline.debug))
