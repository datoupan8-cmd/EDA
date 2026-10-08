"""Behavior-preserving adapter for the existing Component V3 frontend."""
from __future__ import annotations

from ..component_detection import detect_components_v3
from ..core.interfaces import ComponentStageOutput, TextStageOutput
from ..official_types import to_official_type
from ..vision import detect_scene


class ComponentStageV3:
    def run(self, image, text: TextStageOutput, context) -> ComponentStageOutput:
        components, roles, diagnostics, debug = detect_components_v3(image, text.texts, context.image_name)
        legacy_scene = detect_scene(image, context.ocr)
        existing = {component.key.upper() for component in components}
        fallback_events = []
        for candidate in legacy_scene.components:
            if candidate.key.upper() in existing or candidate.confidence < .10:
                continue
            candidate.type = to_official_type(candidate.type)
            if candidate.type == "other":
                continue
            candidate.source_id = "legacy_missing_key_fallback"
            candidate.pins = []
            candidate.observable_designator = candidate.observable_designator or candidate.key
            components.append(candidate)
            existing.add(candidate.key.upper())
            fallback_events.append({
                "component": candidate.key,
                "type": candidate.type,
                "bbox": candidate.bbox,
                "confidence": candidate.confidence,
            })
        diagnostics["legacy_missing_key_fallback"] = fallback_events
        diagnostics["legacy_fallback_count"] = len(fallback_events)
        return ComponentStageOutput(components, text.texts, roles, diagnostics, debug)
