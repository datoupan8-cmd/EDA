"""OCR backends for Component V4.1.

The tiled backend intentionally follows the teammate implementation: every
640 px tile (96 px overlap) is passed to EasyOCR with 2x magnification. Tile
results are cached and converted back to full-image coordinates before NMS.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import cv2
import numpy as np

from .component_detector_yolo import tile_origins
from .runtime_paths import prepend_compatible_vendor_paths
from .schema import Text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
prepend_compatible_vendor_paths(PROJECT_ROOT)


@dataclass(frozen=True)
class TiledEasyOCRConfig:
    tile_size: int = 640
    overlap: int = 96
    magnification: float = 2.0
    text_threshold: float = 0.5
    low_text: float = 0.3
    link_threshold: float = 0.3
    min_size: int = 4
    dedup_iou: float = 0.3


def _bbox_iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def deduplicate_texts(tokens: list[Text], iou_threshold: float = 0.3) -> list[Text]:
    """Remove duplicate readings created by overlapping OCR tiles."""
    kept: list[Text] = []
    for token in sorted(tokens, key=lambda item: item.score, reverse=True):
        normalized = _normalized_text(token.text)
        duplicate = any(
            normalized == _normalized_text(other.text)
            and _bbox_iou(token.bbox, other.bbox) >= iou_threshold
            for other in kept
        )
        if not duplicate:
            kept.append(token)
    return sorted(kept, key=lambda item: (item.bbox[1], item.bbox[0], item.text))


class TiledEasyOCR:
    """EasyOCR adapter compatible with the existing V4 OCR interface."""

    name = "easyocr_tiled"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        *,
        device: str = "auto",
        allow_download: bool = False,
        config: TiledEasyOCRConfig | None = None,
        reader: Any | None = None,
    ) -> None:
        self.config = config or TiledEasyOCRConfig()
        self.cache = (Path(cache_dir) / "easyocr_tiles").resolve() if cache_dir else None
        requested_model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "models" / "easyocr"
        if not requested_model_dir.is_absolute():
            requested_model_dir = PROJECT_ROOT / requested_model_dir
        self.model_dir = requested_model_dir.resolve()
        self.device = device
        self.allow_download = allow_download
        self.reader = reader
        self.last_diagnostics: dict[str, object] = {}
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _gpu_enabled(self) -> bool:
        if self.device == "cpu":
            return False
        if self.device in {"cuda", "gpu", "0"}:
            return True
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _ensure_reader(self):
        if self.reader is not None:
            return self.reader
        import easyocr

        self.model_dir.mkdir(parents=True, exist_ok=True)
        user_network = self.model_dir / "user_network"
        user_network.mkdir(parents=True, exist_ok=True)
        self.reader = easyocr.Reader(
            ["en"],
            gpu=self._gpu_enabled(),
            model_storage_directory=str(self.model_dir),
            user_network_directory=str(user_network),
            download_enabled=self.allow_download,
            verbose=True,
        )
        return self.reader

    def _tile_cache_path(self, tile: np.ndarray) -> Path | None:
        payload = {
            "backend": "easyocr-1.7.2-tiled-v1",
            "shape": list(tile.shape),
            "tile_size": self.config.tile_size,
            "magnification": self.config.magnification,
            "thresholds": [self.config.text_threshold, self.config.low_text, self.config.link_threshold],
        }
        # A 96-bit filename is collision-safe for this cache and avoids the
        # Windows MAX_PATH failure seen after extracting the project deeply.
        digest = hashlib.sha256(tile.tobytes() + json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        return self.cache / f"{digest}.json" if self.cache else None

    def _read_tile(self, tile: np.ndarray) -> tuple[list[dict[str, object]], bool]:
        cache_path = self._tile_cache_path(tile)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), True
        reader = self._ensure_reader()
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        result = reader.readtext(
            rgb,
            detail=1,
            paragraph=False,
            batch_size=32,
            workers=0,
            decoder="greedy",
            min_size=self.config.min_size,
            canvas_size=max(1280, self.config.tile_size * 2),
            mag_ratio=self.config.magnification,
            text_threshold=self.config.text_threshold,
            low_text=self.config.low_text,
            link_threshold=self.config.link_threshold,
        )
        rows: list[dict[str, object]] = []
        for polygon, text, confidence in result:
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            if not text or not xs or not ys:
                continue
            rows.append({"text": str(text), "bbox": [min(xs), min(ys), max(xs), max(ys)], "score": float(confidence)})
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return rows, False

    def recognize(self, image: np.ndarray) -> list[Text]:
        start = time.perf_counter()
        height, width = image.shape[:2]
        x_origins = tile_origins(width, self.config.tile_size, self.config.overlap)
        y_origins = tile_origins(height, self.config.tile_size, self.config.overlap)
        raw: list[Text] = []
        cached_tiles = 0
        for top in y_origins:
            for left in x_origins:
                tile = image[top:min(top + self.config.tile_size, height), left:min(left + self.config.tile_size, width)]
                rows, was_cached = self._read_tile(tile)
                cached_tiles += int(was_cached)
                for row in rows:
                    x1, y1, x2, y2 = (float(value) for value in row["bbox"])
                    raw.append(Text(str(row["text"]), (x1 + left, y1 + top, x2 + left, y2 + top), float(row["score"])))
        result = deduplicate_texts(raw, self.config.dedup_iou)
        self.last_diagnostics = {
            "backend": self.name,
            "tile_size": self.config.tile_size,
            "overlap": self.config.overlap,
            "magnification": self.config.magnification,
            "tile_count": len(x_origins) * len(y_origins),
            "cached_tile_count": cached_tiles,
            "raw_token_count": len(raw),
            "deduplicated_token_count": len(result),
            "seconds": round(time.perf_counter() - start, 3),
        }
        return result

    def recognize_regions(
        self,
        image: np.ndarray,
        regions: list[dict[str, object]],
        *,
        upscale: float = 2.0,
        enhance: bool = True,
    ) -> list[Text]:
        """OCR selected component neighbourhoods and restore global coordinates.

        ``regions`` are chosen by the component frontend after its first
        association pass.  Each region is still read independently, but unlike
        full tiling this path spends EasyOCR time only on components whose
        designator/name/value is missing or uncertain.
        """
        start = time.perf_counter()
        height, width = image.shape[:2]
        raw: list[Text] = []
        cached_regions = 0
        processed: list[dict[str, object]] = []
        scale = max(1.0, float(upscale))
        for index, region in enumerate(regions):
            x1, y1, x2, y2 = (int(round(float(value))) for value in region["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            crop = image[y1:y2, x1:x2]
            if scale != 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if enhance:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
                gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
                crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            rows, was_cached = self._read_tile(crop)
            cached_regions += int(was_cached)
            for row in rows:
                rx1, ry1, rx2, ry2 = (float(value) for value in row["bbox"])
                raw.append(Text(
                    str(row["text"]),
                    (rx1 / scale + x1, ry1 / scale + y1, rx2 / scale + x1, ry2 / scale + y1),
                    float(row["score"]),
                    source_id=f"easyocr_local:{index}",
                ))
            processed.append({
                "component": region.get("component"),
                "bbox": [x1, y1, x2, y2],
                "reasons": list(region.get("reasons", [])),
                "cached": was_cached,
                "raw_token_count": len(rows),
            })
        result = deduplicate_texts(raw, self.config.dedup_iou)
        self.last_diagnostics = {
            "backend": "easyocr_component_regions",
            "region_count": len(processed),
            "cached_region_count": cached_regions,
            "raw_token_count": len(raw),
            "deduplicated_token_count": len(result),
            "upscale": scale,
            "enhance": enhance,
            "regions": processed,
            "seconds": round(time.perf_counter() - start, 3),
        }
        return result

    def recognize_crop(self, image: np.ndarray):
        """Compatibility helper for existing local pin/text refinement code."""
        if image is None or image.size == 0:
            return []
        reader = self._ensure_reader()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows = reader.readtext(rgb, detail=1, paragraph=False, mag_ratio=2.0, min_size=3)
        return [(str(text), float(confidence)) for _, text, confidence in rows]

    def read_pin_number(self, image: np.ndarray, box):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        patch = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
        if patch.size == 0:
            return None
        candidates = self.recognize_crop(patch)
        candidates.sort(key=lambda row: row[1], reverse=True)
        for text, confidence in candidates:
            compact = re.sub(r"\s+", "", text)
            if confidence >= 0.65 and re.fullmatch(r"\d{1,3}", compact):
                return compact
        return None


class HybridOCR:
    """Run every EasyOCR tile and merge it with the existing full-image OCR.

    This preserves the user-selected tiled strategy while retaining text that
    only the original RapidOCR frontend can read.
    """

    name = "hybrid_tiled_easyocr_rapidocr"

    def __init__(self, tiled: TiledEasyOCR, rapid) -> None:
        self.tiled = tiled
        self.rapid = rapid
        self.last_diagnostics: dict[str, object] = {}

    def recognize(self, image: np.ndarray) -> list[Text]:
        start = time.perf_counter()
        rapid_tokens = self.rapid.recognize(image)
        tiled_tokens = self.tiled.recognize(image)
        combined = deduplicate_texts([*rapid_tokens, *tiled_tokens], self.tiled.config.dedup_iou)
        self.last_diagnostics = {
            "backend": self.name,
            "rapid_token_count": len(rapid_tokens),
            "tiled_token_count": len(tiled_tokens),
            "merged_token_count": len(combined),
            "tiled": dict(self.tiled.last_diagnostics),
            "seconds": round(time.perf_counter() - start, 3),
        }
        return combined

    def recognize_crop(self, image: np.ndarray):
        rapid = self.rapid.recognize_crop(image)
        tiled = self.tiled.recognize_crop(image)
        best: dict[str, tuple[str, float]] = {}
        for text, confidence in [*rapid, *tiled]:
            key = _normalized_text(text)
            if key and (key not in best or confidence > best[key][1]):
                best[key] = (text, confidence)
        return sorted(best.values(), key=lambda row: row[1], reverse=True)

    def read_pin_number(self, image: np.ndarray, box):
        return self.rapid.read_pin_number(image, box) or self.tiled.read_pin_number(image, box)


class SelectiveLocalOCR:
    """Full-image RapidOCR plus a targeted EasyOCR component refinement pass."""

    name = "selective_local_easyocr_rapidocr"

    def __init__(self, tiled: TiledEasyOCR, rapid) -> None:
        self.tiled = tiled
        self.rapid = rapid
        self.initial_tokens: list[Text] = []
        self.last_diagnostics: dict[str, object] = {}

    def recognize(self, image: np.ndarray) -> list[Text]:
        start = time.perf_counter()
        self.initial_tokens = self.rapid.recognize(image)
        self.last_diagnostics = {
            "backend": self.name,
            "initial_backend": "rapidocr_full_image",
            "initial_token_count": len(self.initial_tokens),
            "local_pass_executed": False,
            "seconds": round(time.perf_counter() - start, 3),
        }
        return list(self.initial_tokens)

    def recognize_component_regions(
        self,
        image: np.ndarray,
        regions: list[dict[str, object]],
    ) -> list[Text]:
        start = time.perf_counter()
        local_tokens = self.tiled.recognize_regions(image, regions, upscale=2.0, enhance=True)
        merged = deduplicate_texts([*self.initial_tokens, *local_tokens], self.tiled.config.dedup_iou)
        self.last_diagnostics.update({
            "local_pass_executed": True,
            "selected_component_count": len(regions),
            "local_token_count": len(local_tokens),
            "merged_token_count": len(merged),
            "local": dict(self.tiled.last_diagnostics),
            "seconds": round(time.perf_counter() - start, 3) + float(self.last_diagnostics.get("seconds", 0.0)),
        })
        return merged

    def recognize_crop(self, image: np.ndarray):
        # Pin parsing is frozen in this component-only ablation.  Keep its
        # original RapidOCR behaviour so the measured change comes from the
        # selected component refinement regions.
        return self.rapid.recognize_crop(image)

    def read_pin_number(self, image: np.ndarray, box):
        return self.rapid.read_pin_number(image, box)
