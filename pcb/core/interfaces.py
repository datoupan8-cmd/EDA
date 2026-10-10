"""Stage contracts. These containers do not alter the existing domain schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schema import Component, Scene, Text


@dataclass
class TextStageOutput:
    texts: list[Text]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStageOutput:
    components: list[Component]
    texts: list[Text]
    roles: list[Any]
    diagnostics: dict[str, Any]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class PinLocalizationOutput:
    # A list, rather than a key-indexed dict, preserves order and supports the
    # legacy possibility of duplicate component keys before export validation.
    terminals: list[tuple[Component, list[dict[str, Any]]]]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class PinSemanticsOutput:
    components: list[Component]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class WireStageOutput:
    mask: Any
    color_debug: Any = None
    suppressed: Any = None
    corridor: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyStageOutput:
    skeleton: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubmissionStageOutput:
    data: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TextStage(Protocol):
    def run(self, image: Any, context: Any) -> TextStageOutput: ...


class ComponentStage(Protocol):
    def run(self, image: Any, text: TextStageOutput, context: Any) -> ComponentStageOutput: ...


class PinLocalizationStage(Protocol):
    def run(self, image: Any, component: ComponentStageOutput, context: Any) -> PinLocalizationOutput: ...


class PinSemanticsStage(Protocol):
    def run(self, component: ComponentStageOutput, localization: PinLocalizationOutput, context: Any) -> PinSemanticsOutput: ...


class ImagePinSemanticsStage(Protocol):
    """Optional image-aware interface for local OCR; legacy run stays valid."""
    def run_image(self, image: Any, component: ComponentStageOutput, localization: PinLocalizationOutput, context: Any) -> PinSemanticsOutput: ...


def run_pin_semantics(stage: Any, image: Any, component: ComponentStageOutput,
                      localization: PinLocalizationOutput, context: Any) -> PinSemanticsOutput:
    """Bridge the two explicit interfaces without knowing any algorithm name."""
    image_runner = getattr(stage, "run_image", None)
    if callable(image_runner):
        return image_runner(image, component, localization, context)
    return stage.run(component, localization, context)


class WireStage(Protocol):
    def run(self, image: Any, scene: Scene, context: Any) -> WireStageOutput: ...


class TopologyStage(Protocol):
    def run(self, scene: Scene, wire: WireStageOutput, context: Any) -> TopologyStageOutput: ...


class SubmissionStage(Protocol):
    def run(self, scene: Scene, context: Any) -> SubmissionStageOutput: ...
