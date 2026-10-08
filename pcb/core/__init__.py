"""Stable orchestration layer for independently replaceable PCB stages."""

from .config import PipelineConfig
from .context import PipelineContext
from .pipeline import ModularPipeline, PipelineArtifacts
from .registry import StageRegistry, build_default_registry

__all__ = [
    "ModularPipeline",
    "PipelineArtifacts",
    "PipelineConfig",
    "PipelineContext",
    "StageRegistry",
    "build_default_registry",
]
