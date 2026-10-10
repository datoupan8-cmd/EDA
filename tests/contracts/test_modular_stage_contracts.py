from __future__ import annotations

import unittest

import numpy as np

from pcb.component_proposal_fusion import ComponentProposal
from pcb.core import ModularPipeline, PipelineConfig, PipelineContext, build_default_registry
from pcb.schema import Text
from pcb.submission import validate_strict


class StaticOCR:
    name = "static"

    def __init__(self):
        self.last_diagnostics = {"backend": self.name}

    def recognize(self, image):
        return [Text("R1", (43, 30, 57, 40), .95), Text("10k", (43, 65, 57, 75), .9)]


class StaticDetector:
    def detect(self, image):
        return [ComponentProposal((40, 45, 60, 55), "r", .99, "test")], {"cache_hit": True}


def run_fixture():
    image = np.full((100, 100, 3), 255, np.uint8)
    config = PipelineConfig.current()
    context = PipelineContext("fixture.png", 100, 100, config, StaticOCR(), StaticDetector())
    return ModularPipeline(config, build_default_registry()).run(image, context)


class ModularContractTests(unittest.TestCase):
    def test_registry_contains_only_real_stage_versions(self):
        registry = build_default_registry()
        self.assertTrue({"v3", "v4", "peer_latest", "baseline_v41"}.issubset(registry.versions("component")))
        self.assertEqual(registry.versions("pin_localization"), ("box_scan_preserve", "box_skeleton_masked", "box_skeleton_preserve", "l39", "peer_v4", "v3", "v4", "v5", "v6", "v7", "v8"))
        self.assertEqual(registry.versions("pin_semantics"), ("l39", "peer_v4", "v3", "v4", "v5", "v6", "v7", "v8", "v9"))
        self.assertEqual(registry.versions("wire"), ("v1", "v2", "v3", "v3_frame_guard"))
        self.assertEqual(registry.versions("topology"), ("v1", "v2", "v3"))

    def test_component_contract(self):
        artifacts = run_fixture()
        component = artifacts.scene.components[0]
        self.assertEqual(component.key, "R1")
        self.assertEqual(component.type, "r")
        self.assertEqual(component.bbox, (40.0, 45.0, 60.0, 55.0))

    def test_pin_localization_and_semantics_contract(self):
        artifacts = run_fixture()
        pins = artifacts.scene.components[0].pins
        self.assertEqual([(pin.number, pin.name, pin.side) for pin in pins], [("1", "1", "left"), ("2", "2", "right")])
        self.assertEqual(pins[0].tip, (30.0, 50.0))
        self.assertEqual(pins[0].base, (40.0, 50.0))

    def test_wire_can_feed_topology(self):
        artifacts = run_fixture()
        self.assertEqual(artifacts.wire.mask.shape, (100, 100))
        self.assertEqual(artifacts.topology.skeleton.shape, (100, 100))
        self.assertIsInstance(artifacts.scene.nets, list)

    def test_submission_contract(self):
        artifacts = run_fixture()
        self.assertTrue(validate_strict(artifacts.result, (100, 100)))

    def test_context_does_not_hide_stage_payloads(self):
        config = PipelineConfig.current()
        context = PipelineContext("fixture.png", 100, 100, config, StaticOCR(), StaticDetector())
        self.assertFalse(hasattr(context, "components"))
        self.assertFalse(hasattr(context, "pins"))
        self.assertFalse(hasattr(context, "nets"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
