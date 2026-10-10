"""Experimental P2 boundary detection; V3 remains the formal default."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput
from ...pin_detection import TWO_TERMINAL
from .directional_v5 import build_raw_ink, propose_boundary_terminals
from .v3 import PinLocalizationStageV3


class PinLocalizationStageV5:
    """Keep V3 candidates and append direction-aware boundary proposals."""

    def __init__(self, base_stage=None) -> None:
        self.base_stage = base_stage or PinLocalizationStageV3()

    def run(self, image, component: ComponentStageOutput, context) -> PinLocalizationOutput:
        baseline = self.base_stage.run(image, component, context)
        ink = build_raw_ink(image)
        records = []
        for item, terminals in baseline.terminals:
            if item.type in TWO_TERMINAL or item.type in {"crystal_4pin", "gnd"}:
                continue
            additions, detail = propose_boundary_terminals(ink, item, terminals)
            terminals.extend(additions)
            records.append({"component": item.key, "added_count": len(additions), **detail})
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update({
            "pin_localization": "v5_p2_directional_boundary_experiment",
            "directional_candidate_count": sum(row["added_count"] for row in records),
            "directional_by_component": records,
        })
        return PinLocalizationOutput(baseline.terminals, diagnostics, dict(baseline.debug))
