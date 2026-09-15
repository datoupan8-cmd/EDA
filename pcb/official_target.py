"""Reader for the official ``*_target.json`` format.

The official target uses a Cartesian-like image coordinate system: x increases to
the right and y increases upward. OpenCV uses y increasing downward. The point
conversion therefore uses ``image_height - y``. This is intentionally kept in a
small, tested module so the legacy CVJsonStd conversion is not part of the main
official-ground-truth path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable
from .coordinates import (
    target_to_opencv as _target_to_opencv,
    opencv_to_target as _opencv_to_target,
    target_bbox_to_opencv as _target_bbox_to_opencv,
    opencv_bbox_to_target as _opencv_bbox_to_target,
)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class Component:
    key: str
    name: str | None
    type: str
    value: str | None
    bbox: BBox


@dataclass(frozen=True)
class Pin:
    component_key: str
    key: str
    suffix: str
    pinname: str
    point: Point


@dataclass(frozen=True)
class Edge:
    key: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class Net:
    key: str
    hypergraph: str
    references: tuple[tuple[str, str], ...]
    edges: tuple[Edge, ...]


@dataclass
class Scene:
    source_path: Path
    components: dict[str, Component] = field(default_factory=dict)
    pins: dict[str, dict[str, Pin]] = field(default_factory=dict)
    nets: dict[str, Net] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def iter_pins(self) -> Iterable[Pin]:
        for component_pins in self.pins.values():
            yield from component_pins.values()


def parse_hypergraph(
    value: Any,
    components: Iterable[str] | None = None,
    pins: dict[str, dict[str, Pin]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Parse hyperGraph without assuming that pin suffixes contain no dots."""
    if value is None:
        return ()
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if not text.strip():
        return ()
    refs: list[tuple[str, str]] = []
    component_keys = sorted((str(x) for x in (components or ())), key=len, reverse=True)
    for token in text.split(","):
        token = token.strip()
        if not token or "." not in token:
            continue
        matches: list[tuple[str, str]] = []
        for component_key in component_keys:
            prefix = component_key + "."
            if token.startswith(prefix):
                suffix = token[len(prefix):]
                if pins is None or f"pin_{suffix}" in pins.get(component_key, {}):
                    matches.append((component_key, suffix))
        if matches:
            component_key, pin_suffix = matches[0]
        else:
            component_key, pin_suffix = token.split(".", 1)
        refs.append((component_key.strip(), pin_suffix.strip()))
    return tuple(refs)


def _point(raw: dict[str, Any]) -> Point:
    return Point(float(raw["x"]), float(raw["y"]))


def read_official_target(path: str | Path) -> Scene:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    scene = Scene(source_path=path, raw=raw)

    for key, item in (raw.get("components") or {}).items():
        bbox = item.get("bbox") or [0, 0, 0, 0]
        scene.components[str(key)] = Component(
            key=str(key),
            name=None if item.get("Name") is None else str(item.get("Name")),
            type=str(item.get("type") or ""),
            value=None if item.get("value") is None else str(item.get("value")),
            bbox=BBox(*(float(v) for v in bbox)),
        )

    for component_key, pin_items in (raw.get("pins") or {}).items():
        parsed: dict[str, Pin] = {}
        for pin_key, item in (pin_items or {}).items():
            pin_key = str(pin_key)
            suffix = pin_key[4:] if pin_key.startswith("pin_") else pin_key
            parsed[pin_key] = Pin(
                component_key=str(component_key),
                key=pin_key,
                suffix=suffix,
                pinname=str(item.get("pinname") or ""),
                point=_point(item.get("point") or {"x": 0, "y": 0}),
            )
        scene.pins[str(component_key)] = parsed

    for net_key, item in (raw.get("nets") or {}).items():
        edge_items = item.get("edges") or {}
        edges: list[Edge] = []
        if isinstance(edge_items, dict):
            iterator = edge_items.items()
        elif isinstance(edge_items, list):
            iterator = ((f"edge_{i + 1}", points) for i, points in enumerate(edge_items))
        else:
            iterator = ()
        for edge_key, points in iterator:
            edges.append(
                Edge(str(edge_key), tuple(_point(point) for point in (points or [])))
            )
        hypergraph = str(item.get("hyperGraph") or "")
        scene.nets[str(net_key)] = Net(
            key=str(net_key),
            hypergraph=hypergraph,
            references=parse_hypergraph(hypergraph, scene.components, scene.pins),
            edges=tuple(edges),
        )
    return scene


def target_to_opencv(x: float, y: float, image_height: float) -> tuple[float, float]:
    return _target_to_opencv(x, y, image_height)


def opencv_to_target(x: float, y: float, image_height: float) -> tuple[float, float]:
    return _opencv_to_target(x, y, image_height)


def target_bbox_to_opencv(bbox: BBox, image_height: float) -> BBox:
    return BBox(*_target_bbox_to_opencv((bbox.x_min,bbox.y_min,bbox.x_max,bbox.y_max),image_height))


def opencv_bbox_to_target(bbox: BBox, image_height: float) -> BBox:
    return BBox(*_opencv_bbox_to_target((bbox.x_min,bbox.y_min,bbox.x_max,bbox.y_max),image_height))
