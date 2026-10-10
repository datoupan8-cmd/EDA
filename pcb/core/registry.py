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
    from ..component.v3 import ComponentStageV3
    from ..component.v4 import ComponentStageV4
    from ..output.stage import OfficialSubmissionStage
    from ..pin.localization.v3 import PinLocalizationStageV3
    from ..pin.localization.v4 import PinLocalizationStageV4
    from ..pin.localization.v5 import PinLocalizationStageV5
    from ..pin.localization.v6 import PinLocalizationStageV6
    from ..pin.localization.v7 import PinLocalizationStageV7
    from ..pin.localization.v8 import PinLocalizationStageV8
    from ..pin.localization.box_skeleton_experiment import (
        BoxScanPreserveStage, BoxSkeletonMaskedStage, BoxSkeletonPreserveStage,
    )
    from ..pin.semantics.v3 import PinSemanticsStageV3
    from ..pin.semantics.v4 import PinSemanticsStageV4
    from ..pin.semantics.v5 import PinSemanticsStageV5
    from ..pin.semantics.v6 import PinSemanticsStageV6
    from ..pin.semantics.v7 import PinSemanticsStageV7
    from ..pin.semantics.v8 import PinSemanticsStageV8
    from ..pin.semantics.v9 import PinSemanticsStageV9
    from ..text.stage import CurrentTextStage
    from ..topology_stage.versions import TopologyStageV1, TopologyStageV2, TopologyStageV3
    from ..wire_stage.versions import WireStageV1, WireStageV2, WireStageV3

    registry = StageRegistry()
    registry.register("text", "current", CurrentTextStage)
    registry.register("component", "v3", ComponentStageV3)
    registry.register("component", "v4", ComponentStageV4)
    registry.register("pin_localization", "v3", PinLocalizationStageV3)
    registry.register("pin_localization", "box_scan_preserve", BoxScanPreserveStage)
    registry.register("pin_localization", "box_skeleton_masked", BoxSkeletonMaskedStage)
    registry.register("pin_localization", "box_skeleton_preserve", BoxSkeletonPreserveStage)
    registry.register("pin_localization", "v4", PinLocalizationStageV4)
    registry.register("pin_localization", "v5", PinLocalizationStageV5)
    registry.register("pin_localization", "v6", PinLocalizationStageV6)
    registry.register("pin_localization", "v7", PinLocalizationStageV7)
    registry.register("pin_localization", "v8", PinLocalizationStageV8)
    registry.register("pin_semantics", "v3", PinSemanticsStageV3)
    registry.register("pin_semantics", "v4", PinSemanticsStageV4)
    registry.register("pin_semantics", "v5", PinSemanticsStageV5)
    registry.register("pin_semantics", "v6", PinSemanticsStageV6)
    registry.register("pin_semantics", "v7", PinSemanticsStageV7)
    registry.register("pin_semantics", "v8", PinSemanticsStageV8)
    registry.register("pin_semantics", "v9", PinSemanticsStageV9)
    registry.register("wire", "v1", WireStageV1)
    registry.register("wire", "v2", WireStageV2)
    registry.register("wire", "v3", WireStageV3)
    registry.register("topology", "v1", TopologyStageV1)
    registry.register("topology", "v2", TopologyStageV2)
    registry.register("topology", "v3", TopologyStageV3)
    registry.register("submission", "official", OfficialSubmissionStage)
    return registry
