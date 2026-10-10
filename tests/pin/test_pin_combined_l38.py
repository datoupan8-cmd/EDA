"""Contracts for image-only combination; no model training or dataset access."""
from __future__ import annotations
import copy
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import numpy as np
import pin_combined_l38 as combo
from tools.run_pin_combined_l38 import Guard, differences
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import ComponentStageOutput, PinLocalizationOutput, TextStageOutput
from pcb.core.registry import build_default_registry
from pcb.schema import Component
from pcb.submission import validate_strict


class FakeWords:
    def read_owner(self, image, component, terminals, texts):
        return {"words": [], "regions": []}


class FrontRegistry:
    def __init__(self, front):
        self.front = front
        self.original = build_default_registry()

    def create(self, kind, version):
        if kind == "text":
            return SimpleNamespace(run=lambda image, ctx: TextStageOutput([]))
        if kind == "component":
            return SimpleNamespace(run=lambda image, text, ctx: copy.deepcopy(self.front))
        return self.original.create(kind, version)


class CombinedTests(unittest.TestCase):
    def setUp(self):
        self.front = ComponentStageOutput([
            Component("U1", "box", (10., 10., 50., 50.), body_bbox=(10., 10., 50., 50.)),
            Component("R1", "r", (65., 25., 75., 45.), body_bbox=(65., 25., 75., 45.))], [], [], {})
        self.context = PipelineContext("unit.png", 100, 100, PipelineConfig(), None)

    def result(self):
        pipeline = combo.CombinedPipeline(self.context.config, None, "cpu", FakeWords(), {}, FrontRegistry(self.front))
        with patch.object(combo, "locate_boxes", side_effect=lambda image, components, baseline, model, device: baseline):
            return pipeline.run(np.full((100, 100, 3), 255, np.uint8), self.context)

    def test_candidate_and_baseline_schema(self):
        r = self.result()
        self.assertTrue(validate_strict(r.baseline.result, (100, 100)))
        self.assertTrue(validate_strict(r.candidate.result, (100, 100)))

    def test_components_unchanged(self):
        r = self.result()
        self.assertEqual(r.baseline.result["components"], r.candidate.result["components"])

    def test_nonbox_export_unchanged(self):
        r = self.result()
        self.assertEqual(r.baseline.result["pins"]["R1"], r.candidate.result["pins"]["R1"])

    def test_frontend_not_mutated(self):
        before = copy.deepcopy(asdict(self.front))
        self.result()
        self.assertEqual(before, asdict(self.front))

    def test_body_bbox_and_order_preserved(self):
        r = self.result()
        self.assertEqual([c.key for c in r.frontend.components], [x["component"] for x in r.raw])
        self.assertEqual((10., 10., 50., 50.), r.candidate.scene.components[0].body_bbox)

    def test_raw_records_are_independent(self):
        c = Component("R1", "r", (1., 1., 5., 6.))
        loc = PinLocalizationOutput([(c, [{"tip": (2., 3.), "side": "left", "method": "v3"}])])
        out = combo.raw_terminals(loc)
        out[0]["terminals"][0]["side"] = "right"
        self.assertEqual("left", loc.terminals[0][1][0]["side"])

    def test_candidate_geometry_contract(self):
        r = self.result()
        blocks = {x["component"]: x["terminals"] for x in r.raw}
        for c in r.candidate.scene.components:
            self.assertEqual(len(c.pins), len(blocks[c.key]))
            for p, t in zip(c.pins, blocks[c.key]):
                self.assertEqual(p.tip, t["tip"])
                self.assertEqual(p.base, t["base"])
                self.assertEqual(p.side, t["side"])

    def test_target_read_guard(self):
        g = Guard(install=False)
        with g.inferring(), self.assertRaises(PermissionError):
            g.audit("open", ("C:/data/200_train_cases/0014/a_target.json", "r"))

    def test_sealed_and_quicktest_guards(self):
        for path in ["C:/data/200_train_cases/0151/a.png", "C:/10GT-0930/a.png", "C:/Golden/a.json", "C:/EDA_Pin_Crossing_QuickTest/a.py"]:
            with self.subTest(path=path), self.assertRaises(PermissionError):
                Guard(install=False).audit("open", (path, "r"))

    def test_archived_intermediates_cannot_feed_inference(self):
        for path in ["C:/tmp/pin_sina_frontend/0014.json", "C:/reports/x/predictions/0014/result.json", "C:/reports/x/words/0014.json", "C:/reports/x/raw_terminals.json"]:
            g = Guard(install=False)
            with self.subTest(path=path), g.inferring(), self.assertRaises(PermissionError):
                g.audit("open", (path, "r"))

    def test_content_cache_is_allowed(self):
        g = Guard(install=False)
        with g.inferring():
            g.audit("open", ("C:/reports/pin_word_decoder_l3/ocr_cache/abcd.json", "r"))
            g.audit("open", ("C:/runs/ocr_cache_v4_1/easyocr_tiles/abcd.json", "r"))
        self.assertEqual(0, g.blocked)

    def test_exact_parity_diagnostic_does_not_hide_number_changes(self):
        self.assertEqual([], differences({"x": (1, 2)}, {"x": [1, 2]}))
        self.assertTrue(differences({"pin_1": "GND"}, {"pin_2": "GND"}))


if __name__ == "__main__":
    unittest.main()
