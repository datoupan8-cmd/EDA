from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pcb.ocr_backends import HybridOCR, SelectiveLocalOCR, TiledEasyOCR, TiledEasyOCRConfig, deduplicate_texts
from pcb.schema import Component, Text
from pcb.vision_v4 import component_refinement_regions


class FakeReader:
    def __init__(self):
        self.calls = 0

    def readtext(self, image, **kwargs):
        self.calls += 1
        return [([[10, 10], [30, 10], [30, 20], [10, 20]], "R1", 0.9)]


class TiledEasyOCRTests(unittest.TestCase):
    def test_every_tile_is_processed_and_coordinates_are_global(self):
        reader = FakeReader()
        backend = TiledEasyOCR(
            config=TiledEasyOCRConfig(tile_size=640, overlap=96),
            reader=reader,
        )
        tokens = backend.recognize(np.full((500, 1000, 3), 255, dtype=np.uint8))
        self.assertEqual(reader.calls, 2)
        self.assertEqual(backend.last_diagnostics["tile_count"], 2)
        self.assertEqual(tokens[0].bbox, (10.0, 10.0, 30.0, 20.0))
        self.assertEqual(tokens[1].bbox, (370.0, 10.0, 390.0, 20.0))

    def test_overlapping_equal_tokens_are_deduplicated(self):
        tokens = [
            Text("R1", (100, 100, 130, 115), 0.8),
            Text("r1", (102, 101, 132, 116), 0.95),
            Text("R2", (102, 101, 132, 116), 0.7),
        ]
        kept = deduplicate_texts(tokens)
        self.assertEqual(len(kept), 2)
        self.assertTrue(any(token.text == "r1" for token in kept))

    def test_tile_cache_prevents_a_second_reader_call(self):
        reader = FakeReader()
        with tempfile.TemporaryDirectory() as folder:
            backend = TiledEasyOCR(cache_dir=Path(folder), reader=reader)
            image = np.full((200, 300, 3), 255, dtype=np.uint8)
            backend.recognize(image)
            backend.recognize(image)
            self.assertEqual(reader.calls, 1)
            self.assertEqual(backend.last_diagnostics["cached_tile_count"], 1)

    def test_fresh_run_recreates_missing_cache_directory(self):
        reader = FakeReader()
        with tempfile.TemporaryDirectory() as folder:
            backend = TiledEasyOCR(cache_dir=Path(folder) / "new_cache", reader=reader)
            backend.cache.rmdir()
            backend.recognize(np.full((100, 120, 3), 255, dtype=np.uint8))
            self.assertTrue(backend.cache.is_dir())
            self.assertEqual(len(list(backend.cache.glob("*.json"))), 1)

    def test_relative_model_directory_is_resolved_from_project(self):
        backend = TiledEasyOCR(model_dir=Path("models/easyocr"), reader=FakeReader())
        expected = (Path(__file__).resolve().parents[1] / "models" / "easyocr").resolve()
        self.assertEqual(backend.model_dir, expected)

    def test_cache_filename_is_short_enough_for_deep_windows_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = TiledEasyOCR(cache_dir=folder, reader=FakeReader())
            path = backend._tile_cache_path(np.zeros((20, 20, 3), dtype=np.uint8))
            self.assertEqual(len(path.stem), 24)

    def test_hybrid_keeps_unique_tokens_from_both_backends(self):
        class StaticOCR:
            def __init__(self, tokens):
                self.tokens = tokens
            def recognize(self, image):
                return list(self.tokens)
            def recognize_crop(self, image):
                return []
            def read_pin_number(self, image, box):
                return None

        tiled = StaticOCR([Text("R1", (10, 10, 30, 20), .8)])
        tiled.config = TiledEasyOCRConfig()
        tiled.last_diagnostics = {"tile_count": 1}
        hybrid = HybridOCR(tiled, StaticOCR([Text("C2", (40, 10, 60, 20), .9)]))
        result = hybrid.recognize(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual({token.text for token in result}, {"R1", "C2"})
        self.assertEqual(hybrid.last_diagnostics["tiled"]["tile_count"], 1)

    def test_region_ocr_restores_coordinates_after_upscale(self):
        reader = FakeReader()
        backend = TiledEasyOCR(reader=reader)
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        tokens = backend.recognize_regions(
            image,
            [{"component": "R1", "bbox": (100, 50, 180, 100), "reasons": ["missing_value"]}],
            upscale=2.0,
            enhance=False,
        )
        self.assertEqual(reader.calls, 1)
        self.assertEqual(tokens[0].bbox, (105.0, 55.0, 115.0, 60.0))
        self.assertEqual(backend.last_diagnostics["region_count"], 1)

    def test_only_incomplete_components_get_refinement_regions(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        components = [
            Component("R1", "r", (20, 20, 40, 35), name="R1", value="10k", observable_designator="R1"),
            Component("UNRESOLVED_0001", "c", (100, 80, 120, 100)),
        ]
        diagnostics = {"component_provenance": [
            {"component": "R1", "proposal_index": 0, "name_source": "designator_default"},
            {"component": "UNRESOLVED_0001", "proposal_index": 1, "name_source": "unknown"},
        ], "designator_assignment": {"accepted": [{"proposal_index": 0, "score": .5}]}}
        regions = component_refinement_regions(image, components, diagnostics)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["component"], "UNRESOLVED_0001")
        self.assertIn("missing_designator", regions[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
