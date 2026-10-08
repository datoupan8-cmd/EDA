from dataclasses import dataclass, field
from typing import Optional

# Internal geometry uses original-image pixels, top-left origin. Export flips y once.
Point = tuple[float, float]

@dataclass
class Text:
    text: str
    bbox: tuple[float, float, float, float]
    score: float = 1.0
    source_id: str = ''
    @property
    def center(self):
        a,b,c,d = self.bbox
        return ((a+c)/2, (b+d)/2)

@dataclass
class Pin:
    # ``number`` is the exported target reference suffix.  It is intentionally
    # separate from an OCR-observed physical number and a source internal ID.
    number: str
    name: Optional[str]
    tip: Point
    source_id: str = ''
    base: Optional[Point] = None
    inferred_number: bool = False
    number_confidence: float = 0.0
    name_confidence: float = 0.0
    side: str = ''
    exportable: bool = True
    observable_number: Optional[str] = None
    internal_key: Optional[str] = None
    number_source: str = 'unknown'

@dataclass
class Component:
    key: str
    type: str
    bbox: tuple[float,float,float,float]
    name: Optional[str] = None
    value: Optional[str] = None
    pins: list[Pin] = field(default_factory=list)
    source_id: str = ''
    body_bbox: Optional[tuple] = None
    net_label: Optional[str] = None
    confidence: float = 1.0
    observable_designator: Optional[str] = None
    internal_identifier: Optional[str] = None
    model: Optional[str] = None

@dataclass
class Net:
    pins: list[str]
    segments: list[tuple[Point,Point]]

@dataclass
class Scene:
    width: int
    height: int
    components: list[Component]
    texts: list[Text] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

class DSU:
    def __init__(self, n): self.p = list(range(n)); self.rank = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a,b):
        a,b = self.find(a),self.find(b)
        if a == b: return
        if self.rank[a] < self.rank[b]: a,b=b,a
        self.p[b]=a
        if self.rank[a]==self.rank[b]: self.rank[a]+=1
