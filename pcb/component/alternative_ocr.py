"""Pinned PP-OCRv5 ONNX frontend, independent of the old HybridOCR output."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import cv2

from .detector_yolo import tile_origins
from ..schema import Text
from ..core.interfaces import TextStageOutput
from ..text.ocr_backends import deduplicate_texts

ROOT = Path(__file__).resolve().parents[2]
MODEL_HASHES = {
    "ch_PP-OCRv5_det_mobile.onnx": "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    "ch_PP-OCRv5_rec_mobile.onnx": "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"}


class PPOcrV5Backend:
    name = "ppocrv5_onnx_mobile_tiled"

    def __init__(self, cache_dir):
        self.cache = Path(cache_dir).resolve()
        self.engine = None
        self.last_diagnostics = {}
        self.models = ROOT / "models/ocr_v5"
        for name, expected in MODEL_HASHES.items():
            if hashlib.sha256((self.models / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"OCR model checksum mismatch: {name}")
        self.signature = json.dumps({"runtime": "rapidocr-3.9.2", "models": MODEL_HASHES,
                                     "tiling": [768, 128, 2], "det_limit": 2048}, sort_keys=True).encode()

    def _ensure_engine(self):
        if self.engine is not None:
            return self.engine
        local = ROOT / ".runtime/alternative_frontend"
        if local.is_dir() and str(local) not in sys.path:
            sys.path.insert(0, str(local))
        from rapidocr import RapidOCR
        self.engine = RapidOCR(params=self.engine_parameters())
        return self.engine

    def engine_parameters(self):
        from rapidocr.utils.typings import OCRVersion, ModelType
        return {"Global.use_cls": False, "Global.model_root_dir": str(self.models),
                  "Global.log_level": "warning", "Global.max_side_len": 4000,
                  "Det.model_path": str(self.models / "ch_PP-OCRv5_det_mobile.onnx"),
                  "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
                  "Det.limit_side_len": 2048, "Det.limit_type": "max",
                  "Rec.model_path": str(self.models / "ch_PP-OCRv5_rec_mobile.onnx"),
                  "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
                  "Cls.model_path": str(self.models / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
                  "EngineConfig.onnxruntime.intra_op_num_threads": 4,
                  "EngineConfig.onnxruntime.inter_op_num_threads": 2}

    def _read(self, image, *, left=0, top=0, scale=1.):
        # RapidOCR retains per-call switches. A recognition-only local reread
        # must not disable detection for the next full OCR call.
        output = self._ensure_engine()(image, use_det=True, use_cls=False, use_rec=True)
        if output.txts is None or output.boxes is None:
            return []
        texts = []
        for polygon, value, confidence in zip(output.boxes, output.txts, output.scores):
            xs, ys = polygon[:, 0], polygon[:, 1]
            box = (float(xs.min()) / scale + left, float(ys.min()) / scale + top,
                   float(xs.max()) / scale + left, float(ys.max()) / scale + top)
            texts.append(Text(str(value), box, float(confidence), "ppocrv5"))
        return texts

    def recognize(self, image):
        key = hashlib.sha256(image.tobytes() + self.signature).hexdigest()[:24]
        path = self.cache / f"{key}.json"
        if path.is_file():
            rows = json.loads(path.read_text(encoding="utf-8"))
            self.last_diagnostics = {"backend": self.name, "cache_hit": True, "tokens": len(rows)}
            return [Text(**row) for row in rows]
        texts = self._read(image)
        height, width = image.shape[:2]
        count = 0
        for top in tile_origins(height, 768, 128):
            for left in tile_origins(width, 768, 128):
                patch = image[top:top + 768, left:left + 768]
                patch = cv2.resize(patch, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                texts.extend(self._read(patch, left=left, top=top, scale=2))
                count += 1
        texts = deduplicate_texts(texts, .3)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(text) for text in texts], ensure_ascii=False), encoding="utf-8")
        self.last_diagnostics = {"backend": self.name, "cache_hit": False, "tile_count": count,
                                 "tokens": len(texts), "models_sha256": MODEL_HASHES}
        return texts


class TextStagePPOcrV5:
    def __init__(self):
        self.backend = None

    def run(self, image, context):
        resources = context.resources.get("component", {})
        if self.backend is None:
            cache = Path(resources.get("cache_dir", ROOT / "runs/component_alternative_cache")) / "ocr_v5"
            self.backend = PPOcrV5Backend(cache)
        texts = self.backend.recognize(image)
        return TextStageOutput(texts, dict(self.backend.last_diagnostics))
