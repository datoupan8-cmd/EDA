"""Optional V4 localization, preserving V3 candidates and simple layouts."""
from __future__ import annotations

from collections import Counter

from ...core.interfaces import PinLocalizationOutput
from ...official_types import to_official_type
from .detection import pixel_evidence, terminal_candidates_v4


class PinLocalizationStageV4:
    def run(self, image, component, context):
        evidence = pixel_evidence(image, component.texts)
        rows, events = [], []
        for item in component.components:
            item.type = to_official_type(item.type)
            terminals = terminal_candidates_v4(image, item, component.texts,
                                               context.keypoint_provider, evidence=evidence)
            for terminal in terminals:
                for field in ("tip", "base"):
                    x, y = terminal[field]
                    terminal[field] = (min(context.width - 1.0, max(0.0, x)),
                                       min(context.height - 1.0, max(0.0, y)))
                events.append({"component": item.key, **terminal})
            rows.append((item, terminals))
        return PinLocalizationOutput(rows, {
            "pin_localization_version": "v4",
            "pin_tip_adjustments": sum(bool(event["tip_adjusted"]) for event in events),
            "pin_localization_evidence_counts": dict(Counter(event["localization_evidence"] for event in events)),
            "pin_localization_events": events,
        })
