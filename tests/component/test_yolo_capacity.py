import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pcb.component.detector_yolo import YoloComponentDetector, YoloConfig
from pcb.component.yolo_capacity import ComponentStageYolo11m
from pcb.component.value_groups_v2 import ComponentStageOrderedValues
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import TextStageOutput


class YoloCapacityStageTests(unittest.TestCase):
    def fixture(self, root):
        weights = root / "best.pt"; weights.write_bytes(b"fixture-weights")
        (root / "manifest.json").write_text(json.dumps({"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}),encoding="utf-8")
        runtime = object.__new__(YoloComponentDetector)
        runtime.device, runtime.config, runtime.cache_dir = "cpu", YoloConfig(), root / "cache"
        runtime.names = {0: "r"}
        context = PipelineContext("fixture", 100, 50, PipelineConfig(), None, detector=runtime)
        return weights, runtime, context

    def test_new_weights_keep_device_crop_config_cache_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); weights,runtime,context=self.fixture(root)
            candidate=SimpleNamespace(names=runtime.names)
            factory=Mock(return_value=candidate); stage=ComponentStageYolo11m(factory)
            image=np.zeros((50,100,3),dtype=np.uint8);text=TextStageOutput([])
            result=SimpleNamespace(diagnostics={})
            with patch("pcb.component.yolo_capacity.MODEL_DIR",root), patch.object(ComponentStageOrderedValues,"run",return_value=result) as parent:
                self.assertIs(stage.run(image,text,context),result)
                stage.run(image,text,context)
            factory.assert_called_once_with(weights,device="cpu",config=runtime.config,cache_dir=runtime.cache_dir)
            self.assertIs(context.detector,runtime)
            self.assertIs(parent.call_args.args[2].detector,candidate)
            self.assertIs(parent.call_args.args[0],image)
            self.assertIs(parent.call_args.args[1],text)
            self.assertEqual(result.diagnostics["component_detector_profile"],"yolo11m_finetune120")

    def test_modified_model_rejected_before_constructor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); weights,_,context=self.fixture(root);weights.write_bytes(b"changed")
            factory=Mock();stage=ComponentStageYolo11m(factory)
            with patch("pcb.component.yolo_capacity.MODEL_DIR",root),self.assertRaises(ValueError):
                stage.resolve_detector(context.detector)
            factory.assert_not_called()

    def test_class_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,context=self.fixture(root)
            stage=ComponentStageYolo11m(Mock(return_value=SimpleNamespace(names={0:"c"})))
            with patch("pcb.component.yolo_capacity.MODEL_DIR",root),self.assertRaises(ValueError):
                stage.resolve_detector(context.detector)

    def test_opaque_detector_injection_remains_supported(self):
        detector=object();factory=Mock();stage=ComponentStageYolo11m(factory)
        actual,profile=stage.resolve_detector(detector)
        self.assertIs(actual,detector);self.assertEqual(profile,"provided_detector")
        factory.assert_not_called()


if __name__=="__main__":unittest.main()
