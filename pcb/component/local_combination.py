"""The fast Component combination: YOLO11m, ordered fields, local PARSeq."""
from pathlib import Path

from .alternative_ocr import ROOT
from .parseq_adapted import AdaptedParseqReader
from .parseq_local import accepted_readings, refine_component, select_crops
from .yolo_capacity import ComponentStageYolo11m

MODEL_DIR = ROOT / "models/component/parseq_schematic120"


class ComponentStageYolo11mLocalPARSeq(ComponentStageYolo11m):
    """Reread uncertain nearby identifiers/models without touching shared OCR."""
    def __init__(self, reader_factory=AdaptedParseqReader, **kwargs):
        super().__init__(**kwargs)
        self.reader_factory = reader_factory
        self.reader = None
        self.reader_key = None

    def resolve_reader(self, context):
        settings = context.resources.get("component", {})
        cache = Path(settings.get("cache_dir", ROOT / "runs/component_alternative_cache")) / "local_parseq"
        device = str(getattr(self.detector, "device", getattr(context.detector, "device", "auto")))
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.isdigit():
            device = "cuda:" + device
        key = (cache, device)
        if self.reader is None or self.reader_key != key:
            self.reader = self.reader_factory(cache, MODEL_DIR / "best.pt", MODEL_DIR / "training_manifest.json", device=device)
            self.reader_key = key
        return self.reader

    def run(self, image, text, context):
        original = super().run(image, text, context)
        reader = self.resolve_reader(context)
        crops = select_crops(image, original, reader.config)
        readings, diagnostics = reader.recognize(image, crops)
        texts, decisions = accepted_readings(original, readings, reader.config, recover=True)
        output = refine_component(original, texts, image, context) if decisions else original
        output.diagnostics["component_detector_profile"] = original.diagnostics["component_detector_profile"]
        output.diagnostics["component_local_recognition"] = {
            **diagnostics, "accepted": decisions, "shared_texts_and_roles_preserved": True,
            "geometry_types_preserved": True, "engineering_values_replaced": False,
            "policy": "uncertain_ASCII_designator_or_model_only", "GT_used_in_inference": False}
        return output
