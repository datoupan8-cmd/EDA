"""Network-private contracts. Do not replace the public Component/Pin/Scene."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

Point=tuple[float,float]
Pixel=tuple[int,int]  # Skeleton paths use (row, column), not (x,y).
Segment=tuple[Point,Point]


@dataclass(frozen=True)
class PinView:
    number: str
    name: str | None
    tip: Point
    base: Point | None
    side: str
    exportable: bool


@dataclass(frozen=True)
class ComponentView:
    key: str
    type: str
    bbox: tuple[float,float,float,float]
    body_bbox: tuple[float,float,float,float] | None
    pins: tuple[PinView,...]


@dataclass(frozen=True)
class TextView:
    text: str
    bbox: tuple[float,float,float,float]
    score: float

    @property
    def center(self):
        a,b,c,d=self.bbox
        return ((a+c)/2,(b+d)/2)


@dataclass(frozen=True)
class SceneView:
    width: int
    height: int
    components: tuple[ComponentView,...]
    texts: tuple[TextView,...]
    diagnostics: Mapping[str,Any]


def snapshot(scene):
    def pt(value):return tuple(value) if value is not None else None
    return SceneView(scene.width,scene.height,
        tuple(ComponentView(c.key,c.type,tuple(c.bbox),pt(c.body_bbox),
              tuple(PinView(p.number,p.name,pt(p.tip),pt(p.base),p.side,p.exportable) for p in c.pins))
              for c in scene.components),
        tuple(TextView(t.text,tuple(t.bbox),t.score) for t in scene.texts),
        MappingProxyType({'wire_stroke_width':scene.diagnostics.get('wire_stroke_width',1.)}))


@dataclass(frozen=True)
class WireGraph:
    width: int
    height: int
    paths: tuple[tuple[Pixel,...],...]
    adjacency: Mapping[Pixel,tuple[Pixel,...]]
    # Fixed legacy reference for logical snapping, never the line backend's
    # configurable final simplification. All NumPy arrays are private/read-only.
    reference_segments: tuple[tuple[Segment,...],...]
    mask: Any
    thickness: Any
    skeleton: Any
    # Optional half-open raw edge ranges aligned with reference_segments.
    # Last/defaulted for existing positional WireGraph constructors.
    reference_segment_edge_ranges: tuple[tuple[tuple[int, int] | None, ...], ...] | None = None


@dataclass(frozen=True)
class NetGroup:
    group_id: int
    pins: tuple[str,...]
    path_ids: tuple[int,...]
    contact_points: tuple[Point,...]=()


@dataclass(frozen=True)
class NetworkPlan:
    groups: tuple[NetGroup,...]


@dataclass(frozen=True)
class LogicalResult:
    plan: NetworkPlan
    diagnostics: dict[str,Any]


@dataclass(frozen=True)
class GroupLines:
    group_id: int
    segments: tuple[Segment,...]


@dataclass(frozen=True)
class GeometryResult:
    groups: tuple[GroupLines,...]
