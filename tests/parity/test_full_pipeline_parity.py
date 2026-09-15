from __future__ import annotations

import unittest

import numpy as np

from pcb.component_proposal_fusion import ComponentProposal
from pcb.core import ModularPipeline, PipelineConfig, PipelineContext, build_default_registry
from pcb.schema import Text
from pcb.submission import export
from pcb.topology_v3 import build_topology_v3
from pcb.vision_v4 import detect_scene_v4
from pcb.wire_v3 import extract_wire_v3


class StaticOCR:
    name = "static"

    def __init__(self):
        self.last_diagnostics = {"backend": self.name}

    def recognize(self, image):
        return [Text("R1", (43, 30, 57, 40), .95), Text("10k", (43, 65, 57, 75), .9)]


class StaticDetector:
    def detect(self, image):
        return [ComponentProposal((40, 45, 60, 55), "r", .99, "test")], {"cache_hit": True}


class FullPipelineParity(unittest.TestCase):
    def test_legacy_and_modular_predictions_are_identical(self):
        image = np.full((100, 100, 3), 255, np.uint8)
        legacy_ocr = StaticOCR()
        legacy, _ = detect_scene_v4(image, legacy_ocr, StaticDetector(), "fixture.png", "BEST", text_rules="v4_1")
        mask, _, _, _ = extract_wire_v3(image, legacy)
        build_topology_v3(legacy, mask, merge_labels=False)
        legacy.diagnostics["ocr"] = dict(legacy_ocr.last_diagnostics)
        expected = export(legacy)

        config = PipelineConfig.current()
        context = PipelineContext("fixture.png", 100, 100, config, StaticOCR(), StaticDetector())
        artifacts = ModularPipeline(config, build_default_registry()).run(image, context)
        self.assertEqual(artifacts.result, expected)
        self.assertEqual(artifacts.scene.diagnostics, legacy.diagnostics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
