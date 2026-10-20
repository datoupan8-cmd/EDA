"""Regression checks for independent crop and model input sizes, without weights."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from pcb.component.detector_yolo import YoloConfig, YoloComponentDetector


class Array:
    def __init__(self, rows): self.rows = rows
    def cpu(self): return self
    def tolist(self): return self.rows


class Model:
    def __init__(self): self.calls = []
    def predict(self, tile, **kwargs):
        self.calls.append((tile.shape, kwargs))
        boxes = SimpleNamespace(xyxy=Array([[1, 2, 5, 6]]), conf=Array([.9]), cls=Array([0]))
        return [SimpleNamespace(boxes=boxes)]


def detector(config, cache=None):
    value = object.__new__(YoloComponentDetector)
    value.config, value.device, value.model = config, "cpu", Model()
    value.names, value.weights, value.weight_sha256 = {0: "r"}, Path("fixture.pt"), "fixture"
    value.cache_dir = Path(cache) if cache else None
    return value


class IndependentYoloSizeTests(unittest.TestCase):
    def setUp(self): self.image = np.zeros((100, 180, 3), dtype=np.uint8)

    def test_input_size_changes_without_changing_crops_or_global_boxes(self):
        a = detector(YoloConfig(tile_size=96, overlap=16, inference_size=128))
        b = detector(YoloConfig(tile_size=96, overlap=16, inference_size=192))
        pa, da = a.detect(self.image); pb, db = b.detect(self.image)
        self.assertEqual([p.as_dict() for p in pa], [p.as_dict() for p in pb])
        self.assertEqual([s for s, _ in a.model.calls], [s for s, _ in b.model.calls])
        self.assertEqual(da["tile_count"], 6)
        self.assertEqual(db["tile_count"], 6)
        self.assertTrue(all(kw["imgsz"] == 128 for _, kw in a.model.calls))
        self.assertTrue(all(kw["imgsz"] == 192 for _, kw in b.model.calls))
        self.assertIn((81., 6., 85., 10.), [p.bbox for p in pa])

    def test_crop_size_changes_without_changing_input_size(self):
        a = detector(YoloConfig(tile_size=96, overlap=16))
        b = detector(YoloConfig(tile_size=128, overlap=16))
        _, da = a.detect(self.image); _, db = b.detect(self.image)
        self.assertEqual((da["tile_count"], db["tile_count"]), (6, 2))
        self.assertEqual({kw["imgsz"] for _, kw in a.model.calls + b.model.calls}, {1280})

    def test_cache_separates_both_sizes_and_reuses_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            variants = [(96, 128), (96, 192), (128, 128)]
            for tile, infer in variants:
                d = detector(YoloConfig(tile_size=tile, overlap=16, inference_size=infer), tmp)
                _, first = d.detect(self.image)
                count = len(d.model.calls)
                _, second = d.detect(self.image)
                self.assertFalse(first["cache_hit"])
                self.assertTrue(second["cache_hit"])
                self.assertEqual(len(d.model.calls), count)
                self.assertEqual(second["inference_size"], infer)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 3)

    def test_defaults_and_existing_positional_arguments(self):
        self.assertEqual(YoloConfig().tile_size, 1536)
        self.assertEqual(YoloConfig().inference_size, 1280)
        c = YoloConfig(960, 192, .1, .5, 500)
        self.assertEqual((c.tile_size, c.inference_size), (960, 1280))

    def test_invalid_input_size_rejected_before_loading_model(self):
        for size in (0, -32, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                YoloConfig(inference_size=size)
        with self.assertRaises(ValueError): YoloConfig(tile_size=192, overlap=192)


if __name__ == "__main__": unittest.main()
