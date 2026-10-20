from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pcb.component.local_combination import ComponentStageYolo11mLocalPARSeq
from pcb.component.yolo_capacity import ComponentStageYolo11m
from pcb.component.parseq_local import ParseqConfig
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import TextStageOutput


class LocalCombinationTests(unittest.TestCase):
    def fixture(self, device="cpu"):
        runtime = SimpleNamespace(device=device)
        context = PipelineContext("fixture", 100, 100, PipelineConfig(), object(), detector=runtime,
                                  resources={"component": {"cache_dir": "fixture-cache"}})
        return context

    def test_runtime_device_and_cache_are_kept_and_reader_reused(self):
        factory = Mock(return_value=object())
        stage = ComponentStageYolo11mLocalPARSeq(factory)
        context = self.fixture("0")
        first = stage.resolve_reader(context)
        self.assertIs(stage.resolve_reader(context), first)
        self.assertEqual(factory.call_args.kwargs["device"], "cuda:0")
        self.assertEqual(factory.call_args.args[0], Path("fixture-cache/local_parseq"))
        self.assertEqual(context.detector.device, "0")

    def test_component_only_refinement_and_shared_contract_preserved(self):
        context = self.fixture()
        original = SimpleNamespace(diagnostics={"component_detector_profile": "yolo11m_finetune120"}, texts=[object()], roles=[object()])
        refined = SimpleNamespace(diagnostics={}, texts=original.texts, roles=original.roles)
        reader = SimpleNamespace(config=ParseqConfig(), recognize=Mock(return_value=(["reading"], {"cache_hit": False})))
        stage = ComponentStageYolo11mLocalPARSeq(Mock(return_value=reader))
        image, text = object(), TextStageOutput([])
        with patch.object(ComponentStageYolo11m, "run", return_value=original), \
             patch("pcb.component.local_combination.select_crops", return_value=["crop"]), \
             patch("pcb.component.local_combination.accepted_readings", return_value=(["private"], ["edit"])), \
             patch("pcb.component.local_combination.refine_component", return_value=refined) as refine:
            self.assertIs(stage.run(image, text, context), refined)
        refine.assert_called_once_with(original, ["private"], image, context)
        self.assertEqual(text.texts, [])
        self.assertIs(refined.texts, original.texts)
        self.assertIs(refined.roles, original.roles)

    def test_no_eligible_edit_skips_rematching(self):
        context = self.fixture()
        original = SimpleNamespace(diagnostics={"component_detector_profile": "yolo11m_finetune120"})
        reader = SimpleNamespace(config=ParseqConfig(), recognize=Mock(return_value=([], {})))
        stage = ComponentStageYolo11mLocalPARSeq(Mock(return_value=reader))
        with patch.object(ComponentStageYolo11m, "run", return_value=original), \
             patch("pcb.component.local_combination.select_crops", return_value=[]), \
             patch("pcb.component.local_combination.accepted_readings", return_value=([], [])), \
             patch("pcb.component.local_combination.refine_component") as refine:
            self.assertIs(stage.run(object(), TextStageOutput([]), context), original)
        refine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
