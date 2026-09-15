"""Algorithm-free orchestration of registered PCB parsing stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schema import Scene
from .config import PipelineConfig
from .context import PipelineContext
from .interfaces import SubmissionStageOutput, TopologyStageOutput, WireStageOutput
from .registry import StageRegistry


@dataclass
class PipelineArtifacts:
    scene: Scene
    result: dict[str, Any]
    component_debug: dict[str, Any]
    wire: WireStageOutput
    topology: TopologyStageOutput
    submission: SubmissionStageOutput


class ModularPipeline:
    """Call selected stages in the stable public order; contain no algorithms."""

    def __init__(self, config: PipelineConfig, registry: StageRegistry) -> None:
        self.config = config
        self.text = registry.create("text", config.text)
        self.component = registry.create("component", config.component)
        self.pin_localization = registry.create("pin_localization", config.pin_localization)
        self.pin_semantics = registry.create("pin_semantics", config.pin_semantics)
        self.wire = registry.create("wire", config.wire)
        self.topology = registry.create("topology", config.topology)
        self.submission = registry.create("submission", config.submission)

    def run(self, image: Any, context: PipelineContext) -> PipelineArtifacts:
        text = self.text.run(image, context)
        component = self.component.run(image, text, context)
        localization = self.pin_localization.run(image, component, context)
        semantics = self.pin_semantics.run(component, localization, context)
        scene = Scene(
            context.width,
            context.height,
            semantics.components,
            component.texts,
            diagnostics=component.diagnostics,
        )
        scene.diagnostics.update(semantics.diagnostics)
        wire = self.wire.run(image, scene, context)
        topology = self.topology.run(scene, wire, context)
        if getattr(context.ocr, "last_diagnostics", None):
            scene.diagnostics["ocr"] = dict(context.ocr.last_diagnostics)
        submission = self.submission.run(scene, context)
        return PipelineArtifacts(scene, submission.data, component.debug, wire, topology, submission)
