"""Behavior-preserving wrapper around the current Component V4.1 frontend."""
from __future__ import annotations

from ..component_detection_v4 import detect_components_v4
from ..core.interfaces import ComponentStageOutput, TextStageOutput
from ..vision_v4 import VALUE_TYPES, component_refinement_regions


class ComponentStageV4:
    def run(self, image, text: TextStageOutput, context) -> ComponentStageOutput:
        texts = text.texts
        components, roles, diagnostics, debug = detect_components_v4(
            image,
            texts,
            context.image_name,
            context.detector,
            context.config.component_stage,
            context.config.text_rules,
        )
        # This block intentionally mirrors vision_v4.detect_scene_v4 verbatim:
        # the existing SelectiveLocalOCR behavior is part of Component V4.1.
        if hasattr(context.ocr, "recognize_component_regions"):
            regions = component_refinement_regions(image, components, diagnostics)
            initial_summary = {
                "component_count": len(components),
                "unresolved_count": sum(component.observable_designator is None for component in components),
                "missing_value_count": sum(component.type in VALUE_TYPES and component.value is None for component in components),
            }
            if regions:
                texts = context.ocr.recognize_component_regions(image, regions)
                components, roles, diagnostics, debug = detect_components_v4(
                    image,
                    texts,
                    context.image_name,
                    context.detector,
                    context.config.component_stage,
                    context.config.text_rules,
                )
            diagnostics["selective_local_refinement"] = {
                "enabled": True,
                "selected_component_count": len(regions),
                "regions": regions,
                "initial": initial_summary,
                "final": {
                    "component_count": len(components),
                    "unresolved_count": sum(component.observable_designator is None for component in components),
                    "missing_value_count": sum(component.type in VALUE_TYPES and component.value is None for component in components),
                },
            }
        return ComponentStageOutput(components, texts, roles, diagnostics, debug)
