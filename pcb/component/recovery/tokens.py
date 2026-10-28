"""Literal OCR observations shared by the fixed recovery rules."""
from dataclasses import asdict,dataclass
from typing import Any

@dataclass(frozen=True)
class OcrToken:
    text: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    source: str = "unknown"
    variant: str = "unknown"
    binding_id: str | None = None
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
