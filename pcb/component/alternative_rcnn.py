"""Independent two-stage detector, with the existing text contract held constant."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .proposal_fusion import ComponentProposal
from .detector_yolo import tile_origins
from .latest.text_roles import classify_tokens
from .latest.text_assignment import (assign_designators, assign_values, build_components,
                                     recover_near_box_designators)
from .contextual_values import refine_contextual_values
from ..core.interfaces import ComponentStageOutput
from ..official_types import OFFICIAL_COMPONENT_TYPES

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("__background__", *sorted(OFFICIAL_COMPONENT_TYPES))


def build_rcnn(*, pretrained=None):
    """FPN stride 4..64 plus small-symbol anchors; no automatic downloads."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
        min_size=768, max_size=1024, box_detections_per_img=500,
        box_score_thresh=.05, rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, len(CLASSES))
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)), aspect_ratios=((.5, 1., 2.),) * 5)
    # Keep pretrained low-level features fixed; learn domain-specific upper layers.
    for name, parameter in model.backbone.body.named_parameters():
        if not name.startswith(("layer2.", "layer3.", "layer4.")):
            parameter.requires_grad_(False)
    return model


@dataclass(frozen=True)
class RcnnConfig:
    tile_size: int = 768
    overlap: int = 192
    confidence: float = .10
    nms_iou: float = .50
    max_det: int = 500


class RcnnDetector:
    def __init__(self, weights, *, device="cuda", cache_dir=None):
        import torch
        self.weights = Path(weights).resolve()
        self.weight_sha256 = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        if tuple(payload["classes"]) != CLASSES:
            raise ValueError("Checkpoint taxonomy does not match the frozen output contract")
        self.training_ids = payload["training_case_ids"]
        if self.training_ids != [f"{n:04d}" for n in range(1, 151) if n % 5]:
            raise ValueError("Only the frozen development training split is accepted")
        self.config = RcnnConfig()
        self.device = device
        self.cache = Path(cache_dir).resolve() if cache_dir else None
        self.model = build_rcnn()
        self.model.load_state_dict(payload["model"])
        self.model.to(device).eval()

    def detect(self, image):
        import torch
        from torchvision.ops import batched_nms
        signature = json.dumps({"weights": self.weight_sha256, "config": asdict(self.config)}, sort_keys=True).encode()
        key = hashlib.sha256(image.tobytes() + signature).hexdigest()[:24]
        cache = self.cache / f"{key}.json" if self.cache else None
        if cache and cache.is_file():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            result = [ComponentProposal(tuple(row["bbox"]), row["type"], row["confidence"],
                                        "faster_rcnn", {}) for row in saved["proposals"]]
            return result, {**saved["diagnostics"], "cache_hit": True}
        height, width = image.shape[:2]
        boxes, scores, labels = [], [], []
        tile_count = 0
        with torch.inference_mode():
            for top in tile_origins(height, self.config.tile_size, self.config.overlap):
                for left in tile_origins(width, self.config.tile_size, self.config.overlap):
                    patch = image[top:top + self.config.tile_size, left:left + self.config.tile_size]
                    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255).to(self.device)
                    with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                        output = self.model([tensor])[0]
                    for box, score, label in zip(output["boxes"].float().cpu().tolist(),
                                                 output["scores"].float().cpu().tolist(),
                                                 output["labels"].cpu().tolist()):
                        if score < self.config.confidence:
                            continue
                        boxes.append([box[0] + left, box[1] + top, box[2] + left, box[3] + top])
                        scores.append(score)
                        labels.append(label)
                    tile_count += 1
        selected = batched_nms(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                              torch.tensor(scores), torch.tensor(labels, dtype=torch.int64),
                              self.config.nms_iou).tolist() if boxes else []
        proposals = [ComponentProposal(tuple(boxes[i]), CLASSES[labels[i]], scores[i], "faster_rcnn", {})
                     for i in selected]
        diagnostics = {"detector": "FasterRCNN_ResNet50_FPN", "config": asdict(self.config),
                       "weights_sha256": self.weight_sha256, "tile_count": tile_count,
                       "proposal_count": len(proposals), "cache_hit": False}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"proposals": [p.as_dict() for p in proposals],
                                          "diagnostics": diagnostics}), encoding="utf-8")
        return proposals, diagnostics


def assemble_rcnn_component_output(proposals, texts, detector_diagnostics):
    """Hold text interpretation constant to isolate the new visual detector."""
    roles = classify_tokens(texts, [p.bbox for p in proposals if p.type in {"box", "block"}])
    roles, recovered = recover_near_box_designators(proposals, roles)
    assigned, assignment = assign_designators(proposals, roles, global_assignment=True)
    components, provenance = build_components(proposals, assigned)
    diagnostics = {"frontend": "faster_rcnn_independent_frontend", "yolo_used": False,
                   "detector": detector_diagnostics, "component_provenance": provenance,
                   "designator_assignment": assignment, "near_box_designator_recovery": recovered,
                   "value_assignment": assign_values(components, roles, provenance, designators=assigned),
                   "component_final": len(components), "GT_used_in_inference": False}
    return refine_contextual_values(ComponentStageOutput(components, texts, roles, diagnostics,
                                                         {"fused_proposals": proposals}), pin_boxes=True)


class ComponentStageFasterRcnn:
    def __init__(self):
        self.detector = None

    def run(self, image, text, context):
        resources = context.resources.get("component", {})
        if self.detector is None:
            device = resources.get("device", "cpu" if getattr(context.detector, "device", "cpu") == "cpu" else "cuda")
            self.detector = RcnnDetector(ROOT / "models/component_fasterrcnn/final.pth", device=device,
                                        cache_dir=resources.get("cache_dir", ROOT / "runs/component_rcnn_cache"))
        proposals, diagnostics = self.detector.detect(image)
        return assemble_rcnn_component_output(proposals, text.texts, diagnostics)
