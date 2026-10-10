"""Explicit in-process registry for Stage strategies."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class StageRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Callable[[], Any]]] = {}

    def register(self, kind: str, version: str, factory: Callable[[], Any]) -> None:
        versions = self._factories.setdefault(kind, {})
        if version in versions:
            raise ValueError(f"Stage already registered: {kind}.{version}")
        versions[version] = factory

    def create(self, kind: str, version: str) -> Any:
        try:
            factory = self._factories[kind][version]
        except KeyError as exc:
            available = sorted(self._factories.get(kind, {}))
            raise ValueError(f"Unknown {kind} Stage {version!r}; available={available}") from exc
        return factory()

    def versions(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(self._factories.get(kind, {})))

    def describe(self) -> dict[str, list[str]]:
        return {kind: sorted(versions) for kind, versions in sorted(self._factories.items())}


def build_default_registry() -> StageRegistry:
    """Each owner registers versions locally; core never names algorithms."""
    from ..text.registry import register_stages as text
    from ..component.registry import register_stages as component
    from ..pin.registry import register_stages as pin
    from ..wire_stage.registry import register_stages as wire
    from ..topology_stage.registry import register_stages as topology
    from ..output.registry import register_stages as submission

    registry = StageRegistry()
    for register in (text, component, pin, wire, topology, submission):
        register(registry)
    return registry
