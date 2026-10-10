"""Standard Stage wrappers for the frozen L1/L3/E6/L36/L37 Pin combination.

Only import and runtime resource plumbing differs from the validated L39
implementation. All thresholds, weights, formulas and output semantics stay.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ...core.interfaces import PinSemanticsOutput
from ...schema import Scene
from ..localization.v3 import PinLocalizationStageV3
from .bbox_semantics import BoxBBoxSemantics
from .contextual_roles import refine_scene, assert_invariants
from .joint_alignment import stage as joint_stage
from .library_semantics import stage as library_stage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIN_WEIGHT_SHA256 = "a488d0dc4897440ab7c0241c9044e1f260d02f8e2151ad80e39bcb121b4eec69"
LIBRARY_SHA256 = "9c2bec334d41119f29a62cd6ff3fb36ae28fa28cbb7aeb3155875a355b38d322"


def checked(path, expected):
    p = Path(path)
    with p.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Frozen asset missing/changed: {p}. Run git lfs pull; do not replace weights.")
    return p


def options(context):
    return context.resources.get("pin", {})


def raw_terminals(localization):
    return [{"component": c.key, "type": c.type, "terminals": copy.deepcopy(rows)}
            for c, rows in localization.terminals]


class L39Localization:
    """V3 candidates -> unchanged L1 box locator; nonboxes stay unchanged."""
    def __init__(self, model=None, device=None):
        self.model, self.device = model, device

    def run(self, image, component, context):
        import torch
        from .locator import PinPointNet, POLICY, locate_boxes
        cfg = options(context)
        self.device = self.device or cfg.get("device", "cuda")
        if self.model is None:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; explicitly select --device cpu for CPU execution")
            path = checked(cfg.get("weights", PROJECT_ROOT / "reports/pin_learned_locator_l1/best.pt"), PIN_WEIGHT_SHA256)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["policy"] != asdict(POLICY):
                raise ValueError("Frozen L1 policy changed")
            self.model = PinPointNet().to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
        original = PinLocalizationStageV3().run(image, component, context)
        return locate_boxes(image, component.components, original, self.model, self.device)


class L39Semantics:
    """Explicit image input supports local OCR without hiding image in Context."""
    def __init__(self, reader=None, library=None):
        self.reader, self.library = reader, library

    def run(self, component, localization, context):
        raise TypeError("L39 requires image input; use ModularPipeline or run_pin_semantics")

    def run_image(self, image, component, localization, context):
        from .word_reader import WholeWordOCR
        cfg = options(context)
        if self.library is None:
            path = checked(cfg.get("library", PROJECT_ROOT / "assets/cpntLibrary.json"), LIBRARY_SHA256)
            self.library = json.loads(path.read_text(encoding="utf-8"))
        if self.reader is None:
            self.reader = WholeWordOCR(Path(cfg.get("words_cache", PROJECT_ROOT / "runs/team_cache/pin_words")))
        raw = raw_terminals(localization)
        sem = BoxBBoxSemantics().run(component, localization, context)
        scene = Scene(context.width, context.height, sem.components, component.texts,
                      diagnostics=copy.deepcopy(component.diagnostics))
        scene.diagnostics.update(sem.diagnostics)
        scene, role_trace = refine_scene(scene, raw, component.roles)
        blocks = {r["component"]: r["terminals"] for r in raw}
        words = {c.key: self.reader.read_owner(image, c, blocks[c.key], component.texts)
                 for c in scene.components if c.type == "box"}
        candidate, joint_trace = joint_stage(scene, raw, words, component.roles)
        candidate, library_trace = library_stage(candidate, self.library)
        assert_invariants(scene, candidate, raw)
        candidate.diagnostics["pin_l39_stage_trace"] = {
            "raw_terminals": raw, "roles": role_trace, "joint": joint_trace, "library": library_trace,
            "image_only": True, "GT_read_in_inference": False,
        }
        return PinSemanticsOutput(candidate.components, candidate.diagnostics)
