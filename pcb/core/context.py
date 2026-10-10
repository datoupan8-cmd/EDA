"""Small runtime context shared by stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import PipelineConfig


@dataclass
class PipelineContext:
    image_name: str
    width: int
    height: int
    config: PipelineConfig
    ocr: Any
    detector: Any = None
    keypoint_provider: Any = None
    # Runtime resources only: devices, model paths and cache paths. Stage data
    # (components, terminals, pins, wires, nets) must remain explicit outputs.
    resources: dict[str, Any] = field(default_factory=dict)

    @property
    def variant(self) -> str:
        return self.config.variant

    @property
    def enable_flying_labels(self) -> bool:
        return self.config.enable_flying_labels
