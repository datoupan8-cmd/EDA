import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from pcb.core.registry import build_default_registry
from pcb.pin.registry import build_l45_registry
from pcb.core.team_config import load_team_config
from pcb.pin.l45_release import L45RegressionSemantics, MODEL_SHA256, ROOT
from pcb.pin.rc_pair_l45 import RCPointSemantics


class ReleaseSelectionTests(unittest.TestCase):
    def test_opt_in_changes_only_the_pin_semantics_selection(self):
        before = load_team_config("configs/team.json").as_dict()
        after = load_team_config("configs/pin/team_l45.json").as_dict()
        self.assertEqual(before["pin_semantics"], "l39")
        self.assertEqual(after.pop("pin_semantics"), "l45_regression")
        before.pop("pin_semantics")
        self.assertEqual(after, before)
        self.assertNotIn("l45_regression", build_default_registry().versions("pin_semantics"))
        stage = build_l45_registry().create("pin_semantics", "l45_regression")
        self.assertIsInstance(stage.implementation, RCPointSemantics)
        self.assertEqual(stage.implementation.mode, "regression")
        self.assertFalse(stage.implementation.projection)
        self.assertEqual(stage.implementation.resources, ROOT / "models/pin/l45_rc_pair")

    def test_pin_model_cache_follows_explicit_cache_root_without_context_mutation(self):
        stage = object.__new__(L45RegressionSemantics)
        stage.implementation = Mock()
        context = SimpleNamespace(resources={"pin": {"device": "cpu", "words_cache": ROOT / "runs/custom/pin_words"}, "other": {"unchanged": True}})
        original = copy.deepcopy(context.resources)
        stage.run_image("image", "component", "localization", context)
        passed = stage.implementation.run_image.call_args.args[-1]
        self.assertEqual(passed.resources["pin"]["model_cache"], ROOT / "runs/custom/pin_model_control_l42")
        self.assertEqual(context.resources, original)
        self.assertIs(passed.resources["other"], context.resources["other"])

    def test_explicit_model_cache_is_retained(self):
        stage = object.__new__(L45RegressionSemantics)
        stage.implementation = Mock()
        context = SimpleNamespace(resources={"pin": {"model_cache": ROOT / "runs/explicit"}})
        stage.run_image("image", "component", "localization", context)
        self.assertIs(stage.implementation.run_image.call_args.args[-1], context)
