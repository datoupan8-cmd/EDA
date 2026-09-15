"""Central Stage selection without changing algorithm parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    text: str = "current"
    component: str = "v4"
    pin_localization: str = "v3"
    pin_semantics: str = "v3"
    wire: str = "v3"
    topology: str = "v3"
    submission: str = "official"
    component_stage: str = "BEST"
    text_rules: str = "v4_1"
    variant: str = "baseline"
    enable_flying_labels: bool = False
    legacy_pipeline_name: str = "v4_component"

    @classmethod
    def current(
        cls,
        *,
        component_stage: str = "BEST",
        text_rules: str = "v4_1",
        variant: str = "baseline",
        enable_flying_labels: bool = False,
    ) -> "PipelineConfig":
        return cls(
            component_stage=component_stage,
            text_rules=text_rules,
            variant=variant,
            enable_flying_labels=enable_flying_labels,
        )

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "PipelineConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Pipeline config must be a JSON object")
        values = asdict(cls())
        unknown = set(payload) - set(values)
        if unknown:
            raise ValueError(f"Unknown pipeline config fields: {sorted(unknown)}")
        values.update(payload)
        if overrides:
            values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
