"""The validated larger detector with unchanged ordered value assignment."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .detector_yolo import YoloComponentDetector, YoloConfig
from .value_groups_v2 import ComponentStageOrderedValues

MODEL_DIR = Path(__file__).resolve().parents[2] / "models/component/yolo11m_finetune120"


class ComponentStageYolo11m(ComponentStageOrderedValues):
    """Choose owned model weights while retaining runtime device/cache choices.

    Explicit opaque detector implementations remain injectable for contracts
    and external integrations. The standard YOLO runtime selects this profile.
    """
    def __init__(self, detector_factory=YoloComponentDetector):
        super().__init__()
        self.detector_factory = detector_factory
        self.detector = None
        self.runtime_key = None

    def resolve_detector(self, runtime):
        if runtime is not None and not isinstance(runtime, YoloComponentDetector):
            return runtime, "provided_detector"
        device = getattr(runtime, "device", "auto")
        config = getattr(runtime, "config", YoloConfig())
        cache = getattr(runtime, "cache_dir", None)
        key = (device, config, cache)
        if self.detector is None or key != self.runtime_key:
            manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
            weights = MODEL_DIR / "best.pt"
            if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
                raise ValueError("YOLO11m Component checkpoint checksum mismatch")
            candidate = self.detector_factory(weights, device=device, config=config, cache_dir=cache)
            if runtime is not None and candidate.names != runtime.names:
                raise ValueError("YOLO11m class order differs from provided YOLO runtime")
            self.detector, self.runtime_key = candidate, key
        return self.detector, "yolo11m_finetune120"

    def run(self, image, text, context):
        detector, profile = self.resolve_detector(context.detector)
        output = super().run(image, text, replace(context, detector=detector))
        output.diagnostics["component_detector_profile"] = profile
        return output
