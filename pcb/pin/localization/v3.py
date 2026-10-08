"""Wrapper around the unchanged point-first terminal locator."""
from __future__ import annotations

from ...core.interfaces import ComponentStageOutput, PinLocalizationOutput
from ...official_types import to_official_type
from ...pin_detection import terminal_candidates_v3


class PinLocalizationStageV3:
    def run(self, image, component: ComponentStageOutput, context) -> PinLocalizationOutput:
        rows = []
        for item in component.components:
            item.type = to_official_type(item.type)
            terminals = terminal_candidates_v3(
                image,
                item,
                component.texts,
                context.keypoint_provider,
            )
            for terminal in terminals:
                terminal["tip"] = (
                    min(context.width - 1.0, max(0.0, terminal["tip"][0])),
                    min(context.height - 1.0, max(0.0, terminal["tip"][1])),
                )
                terminal["base"] = (
                    min(context.width - 1.0, max(0.0, terminal["base"][0])),
                    min(context.height - 1.0, max(0.0, terminal["base"][1])),
                )
            rows.append((item, terminals))
        return PinLocalizationOutput(rows)
