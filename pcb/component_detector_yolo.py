"""Adapter around the teammate YOLO detector. It returns symbol proposals only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .component_proposal_fusion import ComponentProposal, bbox_iou
from .official_types import to_official_type


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size > 0 and 0 <= overlap < tile_size are required")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def offset_bbox(bbox, left: float, top: float) -> tuple[float, float, float, float]:
    """Convert one tile-local xyxy box to global top-left image coordinates."""
    return (float(bbox[0] + left), float(bbox[1] + top), float(bbox[2] + left), float(bbox[3] + top))


def short_cache_key(image: np.ndarray, signature: bytes) -> str:
    """Return a 96-bit cache filename safe for deeply nested Windows paths."""
    return hashlib.sha256(image.tobytes() + signature).hexdigest()[:24]


def class_aware_nms(proposals: list[ComponentProposal], threshold: float) -> tuple[list[ComponentProposal], list[dict[str, Any]]]:
    """Suppress duplicate tile predictions of the same type and retain type conflicts for fusion."""
    kept: list[ComponentProposal] = []
    rejected: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda p: p.confidence, reverse=True):
        hit = next((old for old in kept if old.type == proposal.type and bbox_iou(old.bbox, proposal.bbox) >= threshold), None)
        if hit is None:
            kept.append(proposal)
        else:
            rejected.append({"proposal": proposal.as_dict(), "reason": "tile_nms", "kept": hit.as_dict(), "iou": bbox_iou(hit.bbox, proposal.bbox)})
    return kept, rejected


@dataclass(frozen=True)
class YoloConfig:
    tile_size: int = 1280
    overlap: int = 192
    confidence: float = .10
    nms_iou: float = .50
    max_det: int = 500


class YoloComponentDetector:
    """Load the supplied best.pt once and run tiled symbol-only inference."""

    def __init__(self, weights: str | Path, device: str = "auto", config: YoloConfig | None = None, cache_dir: str | Path | None = None):
        from ultralytics import YOLO
        import torch

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.device = "0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.config = config or YoloConfig()
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(self.weights))
        self.names = {int(k): to_official_type(v) for k, v in dict(self.model.names).items()}
        self.weight_sha256 = hashlib.sha256(self.weights.read_bytes()).hexdigest()

    def detect(self, image: np.ndarray) -> tuple[list[ComponentProposal], dict[str, Any]]:
        cache_path = None
        if self.cache_dir:
            signature = json.dumps({"weights_sha256": self.weight_sha256, "config": asdict(self.config)}, sort_keys=True).encode()
            key = short_cache_key(image, signature)
            cache_path = self.cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                proposals = [ComponentProposal(tuple(row["bbox"]), row["type"], row["confidence"], row["source"], row.get("metadata", {})) for row in cached["proposals"]]
                diagnostics = dict(cached["diagnostics"])
                diagnostics["cache_hit"] = True
                return proposals, diagnostics
        height, width = image.shape[:2]
        raw: list[ComponentProposal] = []
        tile_count = 0
        for top in tile_origins(height, self.config.tile_size, self.config.overlap):
            for left in tile_origins(width, self.config.tile_size, self.config.overlap):
                tile_count += 1
                tile = image[top:min(top + self.config.tile_size, height), left:min(left + self.config.tile_size, width)]
                result = self.model.predict(
                    tile,
                    imgsz=self.config.tile_size,
                    conf=self.config.confidence,
                    device=self.device,
                    verbose=False,
                    max_det=self.config.max_det,
                )[0]
                if result.boxes is None:
                    continue
                for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                    box = offset_bbox(xyxy, left, top)
                    raw.append(ComponentProposal(box, self.names[int(class_id)], float(confidence), "yolo", {"tile_left": left, "tile_top": top, "class_id": int(class_id)}))
        kept, rejected = class_aware_nms(raw, self.config.nms_iou)
        diagnostics = {
            "weights": str(self.weights.resolve()), "weights_sha256": self.weight_sha256, "device": self.device,
            "tile_size": self.config.tile_size, "overlap": self.config.overlap,
            "tile_count": tile_count, "raw_count": len(raw), "kept_count": len(kept),
            "nms_rejected": rejected, "class_names": self.names, "cache_hit": False,
        }
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"proposals": [p.as_dict() for p in kept], "diagnostics": diagnostics}, ensure_ascii=False), encoding="utf-8")
        return kept, diagnostics
