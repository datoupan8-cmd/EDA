"""Small runtime context shared by stages."""
from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def variant(self) -> str:
        return self.config.variant

    @property
    def enable_flying_labels(self) -> bool:
        return self.config.enable_flying_labels
