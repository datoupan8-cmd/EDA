"""Image-only local PARSeq rereading; experimental, Component-owned.

No case identifiers, target vocabulary, values or other modules are used.
The recognition model supports ASCII; engineering values stay with PP-OCR.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

from .alternative_ocr import ROOT
from .latest.text_assignment import BOX_FAMILY, NO_DESIGNATOR_TYPES, bbox_gap, designator_weight
from .latest.text_roles import DESIGNATOR_RE, VALUE_RE, classify_tokens
from .local_joint import select_regions
from .proposal_fusion import ComponentProposal, bbox_iou
from ..core.interfaces import TextStageOutput
from ..schema import Text


@dataclass(frozen=True)
class ParseqConfig:
    max_ocr_crops: int = 96
    max_new_crops: int = 64
    batch_size: int = 32
    min_confidence: float = .90
    min_character_confidence: float = .60
    version: str = "parseq_local_ascii_v1"


def clipped_box(box, shape, padding=2):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return (max(0, math.floor(x1)-padding), max(0, math.floor(y1)-padding),
            min(w, math.ceil(x2)+padding), min(h, math.ceil(y2)+padding))


def nearby_components(box, original):
    height = max(4., box[3]-box[1])
    return [c for c in original.components if c.type not in NO_DESIGNATOR_TYPES
            and bbox_gap(box, c.bbox) <= min(120., 5*height)]


def word_boxes(image, regions, font):
    """Propose short horizontal words from glyph geometry, without OCR labels."""
    output = []
    for region in regions:
        left, top, right, bottom = region["bbox"]
        gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        if gray.size == 0: continue
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        # Wires are removed only for proposal geometry, never from model pixels.
        lines = cv2.bitwise_or(
            cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(20, int(3*font))), np.uint8)),
            cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(20, int(3*font)), 1), np.uint8)))
        glyphs = cv2.subtract(ink, lines)
        _, _, stats, _ = cv2.connectedComponentsWithStats(glyphs, 8)
        chars = []
        for x, y, w, h, area in stats[1:]:
            if 4 <= h <= min(64, 2.5*font) and 1 <= w <= 2*h and area >= 3 and .08 <= area/(w*h) <= .95:
                chars.append((int(x), int(y), int(x+w), int(y+h)))
        rows = []
        for box in sorted(chars, key=lambda b: ((b[1]+b[3])/2, b[0])):
            cy, h = (box[1]+box[3])/2, box[3]-box[1]
            line = next((row for row in rows if abs(statistics.median((b[1]+b[3])/2 for b in row)-cy) <= .35*h
                         and .5 <= (statistics.median(b[3]-b[1] for b in row)/h) <= 2), None)
            if line is None: rows.append([box])
            else: line.append(box)
        for row in rows:
            groups = []
            for box in sorted(row):
                if not groups or box[0]-groups[-1][-1][2] > max(3, .65*(box[3]-box[1])):
                    groups.append([box])
                else: groups[-1].append(box)
            for group in groups:
                if not 2 <= len(group) <= 24: continue
                box = (min(b[0] for b in group)+left, min(b[1] for b in group)+top,
                       max(b[2] for b in group)+left, max(b[3] for b in group)+top)
                ratio = (box[2]-box[0])/max(1, box[3]-box[1])
                if .7 <= ratio <= 15 and not any(bbox_iou(box, old) >= .6 for old in output): output.append(box)
    return output


def select_crops(image, original, config=ParseqConfig()):
    heights = [t.bbox[3]-t.bbox[1] for t in original.texts if 4 <= t.bbox[3]-t.bbox[1] <= 64]
    font = float(statistics.median(heights)) if heights else 12.
    existing = []
    for index, token in enumerate(original.texts):
        box = clipped_box(token.bbox, image.shape)
        h, w = box[3]-box[1], box[2]-box[0]
        near = nearby_components(box, original)
        if not near or not (5 <= h <= 96 and .6 <= w/h <= 18) or not 1 <= len(token.text.strip()) <= 25:
            continue
        if not token.text.isascii(): continue
        missing = any(c.observable_designator is None for c in near)
        priority = (int(missing), int(token.score < .95), -min(bbox_gap(box, c.bbox) for c in near), -index)
        existing.append((priority, {"bbox": box, "source": "ocr_box", "token_index": index}))
    existing = [row for _, row in sorted(existing, key=lambda row: row[0], reverse=True)[:config.max_ocr_crops]]
    regions = select_regions(image.shape, original)
    extra = []
    for box in word_boxes(image, regions, font):
        near = nearby_components(box, original)
        if not near or any(bbox_iou(box, t.bbox) >= .55 for t in original.texts): continue
        priority = (int(any(c.observable_designator is None for c in near)),
                    -min(bbox_gap(box, c.bbox) for c in near), box)
        extra.append((priority, {"bbox": clipped_box(box, image.shape), "source": "cc_word", "token_index": None}))
    extra = [row for _, row in sorted(extra, key=lambda row: row[0], reverse=True)[:config.max_new_crops]]
    return existing+extra


class ParseqReader:
    def __init__(self, cache_dir, device="cuda", config=ParseqConfig()):
        self.cache, self.device, self.config = Path(cache_dir), device, config
        self.weights = ROOT / "models/component/parseq/parseq-bb5792a6.pt"
        # Ship the exact upstream inference sources with the program; .runtime
        # remains a development fallback for older checkouts only.
        bundled = ROOT / "pcb/component/_vendor/parseq"
        self.source = bundled if bundled.is_dir() else ROOT / ".runtime/parseq_upstream"
        manifest = json.loads((self.source / "source_manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["files"].items():
            if hashlib.sha256((self.source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("PARSeq source checksum mismatch: " + relative)
        sha = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        if sha != "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00":
            raise ValueError("PARSeq weight checksum mismatch")
        self.signature = json.dumps({"weights": sha, "source": manifest["files"], "device": device,
            "config": asdict(config), "preprocess": "RGB,PIL_BICUBIC_32x128,[-1,1]", "timm": "0.9.16"}, sort_keys=True).encode()
        self.model = self.tokenizer = None

    def ensure_model(self):
        if self.model is not None: return
        import torch, yaml, timm
        if timm.__version__ != "0.9.16":
            raise ValueError("PARSeq requires timm==0.9.16; install the project requirements")
        sys.path.insert(0, str(self.source))
        from strhub.models.parseq.model import PARSeq
        from strhub.data.utils import Tokenizer
        from PIL import Image
        charset = yaml.safe_load((self.source / "configs/charset/94_full.yaml").read_text(encoding="utf-8"))["model"]["charset_train"]
        self.tokenizer = Tokenizer(charset)
        self.model = PARSeq(len(self.tokenizer), 25, [32,128], [4,8], 384,6,4,12,12,4,1,True,1,.1)
        self.model.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True), strict=True)
        self.model.eval().to(self.device)
        torch.set_num_threads(4)

    def recognize(self, image, crops):
        fingerprint = json.dumps(crops, sort_keys=True).encode()
        key = hashlib.sha256(image.tobytes()+self.signature+fingerprint).hexdigest()[:24]
        path = self.cache / (key+".json")
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["readings"], {**data["diagnostics"], "cache_hit": True}
        import torch
        from PIL import Image
        self.ensure_model()
        readings = []
        if self.device.startswith("cuda"): torch.cuda.synchronize()
        start = time.perf_counter()
        for begin in range(0, len(crops), self.config.batch_size):
            batch = crops[begin:begin+self.config.batch_size]; tensors = []
            for row in batch:
                left, top, right, bottom = row["bbox"]
                rgb = cv2.cvtColor(image[top:bottom,left:right], cv2.COLOR_BGR2RGB)
                resized = np.asarray(Image.fromarray(rgb).resize((128,32), Image.Resampling.BICUBIC)).copy()
                tensors.append(torch.from_numpy(resized).permute(2,0,1).float().div(127.5).sub(1.))
            with torch.inference_mode():
                probabilities = self.model(self.tokenizer, torch.stack(tensors).to(self.device)).softmax(-1)
                labels, scores = self.tokenizer.decode(probabilities)
            for row, label, score in zip(batch, labels, scores):
                score = score.cpu()
                readings.append({**row, "text": label, "confidence": float(score.clamp_min(1e-12).log().mean().exp()),
                                 "minimum_probability": float(score.min()), "has_eos": len(score) == len(label)+1})
        if self.device.startswith("cuda"): torch.cuda.synchronize()
        diagnostics = {"backend": "PARSeq_official", "crops": len(crops), "seconds": time.perf_counter()-start,
                       "cache_hit": False, "GT_used_in_inference": False}
        self.cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"readings": readings, "diagnostics": diagnostics}), encoding="utf-8")
        temporary.replace(path)
        return readings, diagnostics


def accepted_readings(original, readings, config=ParseqConfig(), *, recover=False):
    texts, decisions = list(original.texts), []
    for row in readings:
        if row["source"] == "cc_word" and not recover: continue
        value = row["text"].strip()
        if (not row["has_eos"] or row["confidence"] < config.min_confidence
                or row["minimum_probability"] < config.min_character_confidence): continue
        if not value.isascii(): continue
        # Numeric prefixes occur in real model names (e.g. 2N3904), but an
        # engineering quantity such as 100k/100nF is never model evidence.
        if VALUE_RE.fullmatch(value): continue
        match = DESIGNATOR_RE.fullmatch(value)
        is_designator = bool(match and not match.group(2) and len(value) <= 10)
        is_model = bool(any(c.isalpha() for c in value) and sum(c.isdigit() for c in value) >= 3
                        and 4 <= len(value) <= 25 and all(c.isalnum() or c in "_.+-/" for c in value))
        if not (is_designator or is_model): continue
        token = Text(value, tuple(row["bbox"]), row["confidence"], "parseq_local")
        near = nearby_components(token.bbox, original)
        if is_designator:
            roles = classify_tokens([token], expanded_rules=True)
            if not roles or not any(designator_weight(ComponentProposal(c.bbox,c.type,c.confidence,"fixed",{}),roles[0]) >= .055 for c in near): continue
        elif not any(c.type in BOX_FAMILY for c in near): continue
        index = row["token_index"]
        if index is not None:
            previous = original.texts[index]
            if previous.text.strip() == value: continue
            # Confidence scales differ. Only uncertain original readings can be
            # replaced, and PARSeq must be distinctly more certain.
            if previous.score >= .95 or row["confidence"] < previous.score+.05: continue
            texts[index] = token
        else:
            if any(bbox_iou(token.bbox, old.bbox) >= .3 and old.score >= .95 for old in texts): continue
            texts.append(token)
        decisions.append({**row, "accepted_as": "designator" if is_designator else "model"})
    return texts, decisions


def refine_component(original, texts, image, context):
    """Rerun identical field logic on fixed proposals; keep shared OCR private."""
    from .value_groups_v2 import ComponentStageOrderedValues
    class FixedDetector:
        def detect(self, _image):
            return original.debug["yolo_proposals"], original.diagnostics["yolo"]
    output = ComponentStageOrderedValues().run(image, TextStageOutput(texts), replace(context, detector=FixedDetector(), ocr=None))
    assert [(c.type,c.bbox) for c in original.components] == [(c.type,c.bbox) for c in output.components]
    output.texts, output.roles = original.texts, original.roles
    return output
