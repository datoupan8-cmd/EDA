"""Compose independently owned config fragments; never select algorithms here."""
from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig

OWNED_FIELDS = {
    "component": {"text", "component", "component_stage", "text_rules"},
    "pin": {"pin_localization", "pin_semantics"},
    "wire_topology": {"wire", "topology", "variant", "enable_flying_labels"},
}


def load_team_config(path: str | Path, root: str | Path | None = None) -> PipelineConfig:
    """Load one stable team manifest and one selection file per module owner.

    Fragment paths are repository-relative, not relative to the shell's cwd.
    Absolute machine paths, escaping symlinks and cross-owner fields fail early.
    A plain legacy PipelineConfig JSON remains accepted.
    """
    project = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    source = Path(path)
    source = source if source.is_absolute() else project / source
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "modules" not in payload:
        return PipelineConfig.load(source)
    if set(payload) != {"base", "modules"} or set(payload["modules"]) != set(OWNED_FIELDS):
        raise ValueError("Team manifest requires base and component/pin/wire_topology modules")

    def local(relative: str) -> Path:
        p = Path(relative)
        if p.is_absolute() or "\\" in relative or ":" in relative:
            raise ValueError("Config paths must be portable repository-relative paths")
        resolved = (project / p).resolve()
        if not resolved.is_relative_to(project):
            raise ValueError("Config path escapes repository")
        return resolved

    values = PipelineConfig.load(local(payload["base"])).as_dict()
    for owner, allowed in OWNED_FIELDS.items():
        fragment = json.loads(local(payload["modules"][owner]).read_text(encoding="utf-8"))
        if not isinstance(fragment, dict) or set(fragment) - allowed:
            raise ValueError(f"{owner} config may only change {sorted(allowed)}")
        values.update(fragment)
    return PipelineConfig(**values)
