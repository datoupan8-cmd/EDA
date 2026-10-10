"""Image-only adapter for the frozen L1/L3/L36/L37/F research combination.

Only configured models, public library and content-addressed OCR/YOLO caches
are read. No dataset/annotation/prediction/snapshot is needed by this module.
Existing algorithms are invoked serially, including their restored adapters.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from pcb.component_detector_yolo import YoloComponentDetector
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import ComponentStageOutput
from pcb.core.pipeline import PipelineArtifacts
from pcb.core.registry import build_default_registry
from pcb.ocr_backends import HybridOCR, TiledEasyOCR
from pcb.schema import Scene
from pcb.submission import validate_strict
from pcb.vision import OCR
from pin_contextual_roles_e6 import refine_scene, assert_invariants
from pin_learned_locator_l1 import PinPointNet, POLICY as POINT_POLICY, locate_boxes
from pin_library_semantics_l37 import stage as library_stage
from pin_side_joint_l36 import stage as joint_stage
from pin_skeleton_followup import BoxBBoxSemantics
from pin_word_reader_l3 import WholeWordOCR
from wire_frame_guard import suppress_box_frames

LIBRARY_SHA256 = "9c2bec334d41119f29a62cd6ff3fb36ae28fa28cbb7aeb3155875a355b38d322"
PIN_WEIGHT_SHA256 = "a488d0dc4897440ab7c0241c9044e1f260d02f8e2151ad80e39bcb121b4eec69"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw_terminals(localization) -> list[dict[str, Any]]:
    """Preserve candidate order, methods and coordinate representation."""
    return [{"component": c.key, "type": c.type, "terminals": copy.deepcopy(rows)}
            for c, rows in localization.terminals]


@dataclass
class CombinedOutputs:
    baseline: PipelineArtifacts
    candidate: PipelineArtifacts
    frontend: ComponentStageOutput
    raw: list[dict[str, Any]]
    words: dict[str, Any]
    trace: dict[str, Any]


class CombinedPipeline:
    """Wrap current frontend and frozen research stages; do not change core."""

    def __init__(self, config: PipelineConfig, model, device: str,
                 reader: WholeWordOCR, library: dict, registry=None) -> None:
        self.config, self.model, self.device = config, model, device
        self.reader, self.library = reader, library
        self.registry = registry or build_default_registry()

    def finish(self, image, scene: Scene, context: PipelineContext, frames: bool) -> PipelineArtifacts:
        wire = self.registry.create("wire", self.config.wire).run(image, scene, context)
        if frames:
            wire = suppress_box_frames(image, scene, wire)
        topology = self.registry.create("topology", self.config.topology).run(scene, wire, context)
        if getattr(context.ocr, "last_diagnostics", None):
            scene.diagnostics["ocr"] = copy.deepcopy(context.ocr.last_diagnostics)
        submission = self.registry.create("submission", self.config.submission).run(scene, context)
        validate_strict(submission.data, (context.width, context.height))
        return PipelineArtifacts(scene, submission.data, {}, wire, topology, submission)

    def run(self, image, context: PipelineContext) -> CombinedOutputs:
        started = time.perf_counter()
        text = self.registry.create("text", self.config.text).run(image, context)
        front = self.registry.create("component", self.config.component).run(image, text, context)
        frontend_seconds = time.perf_counter() - started
        baseline_front = copy.deepcopy(front)
        v3_loc = self.registry.create("pin_localization", self.config.pin_localization).run(image, baseline_front, context)
        v3_sem = self.registry.create("pin_semantics", self.config.pin_semantics).run(baseline_front, v3_loc, context)
        base = Scene(context.width, context.height, v3_sem.components, front.texts,
                     diagnostics=copy.deepcopy(front.diagnostics))
        base.diagnostics.update(v3_sem.diagnostics)
        baseline = self.finish(image, base, context, False)

        own = copy.deepcopy(front)
        original = self.registry.create("pin_localization", "v3").run(image, own, context)
        loc = locate_boxes(image, own.components, original, self.model, self.device)
        raw = raw_terminals(loc)
        sem = BoxBBoxSemantics().run(own, loc, context)
        scene = Scene(context.width, context.height, sem.components, front.texts,
                      diagnostics=copy.deepcopy(front.diagnostics))
        scene.diagnostics.update(sem.diagnostics)
        scene, role_trace = refine_scene(scene, raw, front.roles)
        blocks = {r["component"]: r["terminals"] for r in raw}
        words = {c.key: self.reader.read_owner(image, c, blocks[c.key], front.texts)
                 for c in scene.components if c.type == "box"}
        candidate, joint_trace = joint_stage(scene, raw, words, front.roles)
        candidate, library_trace = library_stage(candidate, self.library)
        assert_invariants(scene, candidate, raw)
        after = self.finish(image, candidate, context, True)
        if baseline.result["components"] != after.result["components"]:
            raise AssertionError("Combination changed Component output")
        for c in baseline.scene.components:
            if c.type != "box" and baseline.result["pins"][c.key] != after.result["pins"][c.key]:
                raise AssertionError("Combination changed nonbox exported Pin")
        return CombinedOutputs(baseline, after, front, raw, words,
                               {"roles": role_trace, "joint": joint_trace, "library": library_trace,
                                "frontend_seconds": frontend_seconds,
                                "total_seconds": time.perf_counter() - started,
                                "image_only": True, "GT_read_in_inference": False})


def build_runtime(root: Path, library_path: Path, cache: Path,
                  words_cache: Path, device: str = "cuda", yolo_cache: Path | None = None):
    """Load unchanged existing assets; all paths supplied or repository-relative."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Original CUDA condition unavailable; no silent protocol change")
    weights = root / "reports/pin_learned_locator_l1/best.pt"
    if digest(weights) != PIN_WEIGHT_SHA256 or digest(library_path) != LIBRARY_SHA256:
        raise AssertionError("Frozen point model/public library version changed")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    from dataclasses import asdict
    if checkpoint["policy"] != asdict(POINT_POLICY):
        raise AssertionError("L1 policy changed")
    model = PinPointNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    config = PipelineConfig.load(root / "configs/current.json")
    if (config.pin_localization, config.pin_semantics, config.wire, config.topology) != ("v3", "v3", "v3", "v3"):
        raise AssertionError("Formal baseline changed; declare a new comparison")
    library = json.loads(library_path.read_text(encoding="utf-8"))
    reader = WholeWordOCR(words_cache)
    ocr = HybridOCR(TiledEasyOCR(cache, root / "models/easyocr", device=device, allow_download=False), OCR(cache))
    detector = YoloComponentDetector(root / "models/component_yolo11n_continue_v2_best.pt", device="0" if device == "cuda" else device, cache_dir=yolo_cache or root / "runs/yolo_cache")
    return CombinedPipeline(config, model, device, reader, library), ocr, detector
