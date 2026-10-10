"""Dataset-free contracts and exact old/new Pin-combination parity."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import ComponentStageOutput, TextStageOutput, run_pin_semantics
from pcb.core.pipeline import ModularPipeline
from pcb.core.registry import build_default_registry, StageRegistry
from pcb.core.team_config import load_team_config
from pcb.pin.frozen_l39.stages import L39Localization, L39Semantics
from pcb.pin.frozen_l39.text_roles import classify_tokens
from pcb.schema import Component, Text
from tools.run_team_pipeline import parse_ids, select_inputs, assert_development_path
from tools.team_access_guard import Guard
from tools.check_module_scope import check_paths


class FakeWords:
    def read_owner(self, image, component, terminals, texts):
        return {"words": [], "regions": []}


def canonical(value):
    return json.dumps(value, sort_keys=True, default=lambda x: x.tolist())


class TeamSlotTests(unittest.TestCase):
    def test_module_scope_keeps_algorithms_config_and_registration_separate(self):
        self.assertEqual(check_paths("pin", ["pcb/pin/registry.py", "configs/pin/current.json", "tests/pin/test_new.py"]), [])
        self.assertEqual(check_paths("component", ["pcb/component/latest/detection.py", "pcb/text/ocr_backends.py"]), [])
        self.assertEqual(check_paths("wire_topology", ["pcb/wire_stage/registry.py", "pcb/topology_stage/v3.py"]), [])
        self.assertEqual(check_paths("pin", ["pcb/core/pipeline.py", "configs/component/current.json"]),
                         ["configs/component/current.json", "pcb/core/pipeline.py"])

    def test_default_selects_both_latest_modules(self):
        config = load_team_config("configs/team.json")
        self.assertEqual(config.component, "peer_latest")
        self.assertEqual((config.pin_localization, config.pin_semantics), ("l39", "l39"))
        self.assertEqual(config.wire, "v3_frame_guard")

    def test_old_and_peer_versions_stay_selectable(self):
        registry = build_default_registry()
        for kind in ("pin_localization", "pin_semantics"):
            self.assertIn("v3", registry.versions(kind))
            self.assertIn("v4", registry.versions(kind))
            self.assertIn("peer_v4", registry.versions(kind))
            self.assertIn("l39", registry.versions(kind))
            self.assertNotEqual(type(registry.create(kind, "v4")), type(registry.create(kind, "peer_v4")))

    def test_duplicate_version_registration_is_rejected(self):
        registry = StageRegistry()
        registry.register("component", "team_test", object)
        with self.assertRaises(ValueError):
            registry.register("component", "team_test", object)

    def test_semantics_bridge_preserves_legacy_interface(self):
        class Legacy:
            def run(self, component, localization, context):
                return (component, localization, context)
        self.assertEqual(run_pin_semantics(Legacy(), "image", 1, 2, 3), (1, 2, 3))

    def test_semantics_bridge_passes_explicit_image(self):
        class ImageAware:
            def run_image(self, image, component, localization, context):
                return (image, component, localization, context)
        self.assertEqual(run_pin_semantics(ImageAware(), 0, 1, 2, 3), (0, 1, 2, 3))

    def test_context_keeps_algorithm_outputs_explicit(self):
        c = PipelineContext("fixture", 10, 10, PipelineConfig(), None)
        for name in ("image", "components", "terminals", "pins", "wires", "nets"):
            self.assertFalse(hasattr(c, name))

    def test_only_pin_config_can_select_pin_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.json").write_text("{}")
            for owner in ("component", "pin", "wire_topology"):
                (root / f"{owner}.json").write_text("{}")
            manifest = {"base": "base.json", "modules": {x: f"{x}.json" for x in ("component", "pin", "wire_topology")}}
            (root / "team.json").write_text(json.dumps(manifest))
            (root / "pin.json").write_text('{"pin_localization":"l39","pin_semantics":"l39"}')
            a = load_team_config(root / "team.json", root)
            (root / "pin.json").write_text('{"pin_localization":"v3","pin_semantics":"v3"}')
            b = load_team_config(root / "team.json", root)
            self.assertEqual(a.component, b.component)
            self.assertEqual(a.wire, b.wire)
            self.assertEqual(a.topology, b.topology)
            (root / "component.json").write_text('{"pin_semantics":"v3"}')
            with self.assertRaises(ValueError):
                load_team_config(root / "team.json", root)

    def test_config_rejects_machine_paths_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for base in ("C:/Users/person/file.json", "../outside.json"):
                (root / "team.json").write_text(json.dumps({"base": base, "modules":
                    {x: "unused.json" for x in ("component", "pin", "wire_topology")}}))
                with self.assertRaises(ValueError):
                    load_team_config(root / "team.json", root)

    def test_development_whitelist(self):
        self.assertEqual(parse_ids("1,14,87"), [1, 14, 87])
        for value in ("151", "0", "1,1", "1,x"):
            with self.assertRaises((ValueError, PermissionError)):
                parse_ids(value)
        with self.assertRaises(PermissionError):
            select_inputs(Path("C:/Golden/a.png"), None, None)

    def test_dataset_root_is_allowed_only_as_a_dataset(self):
        root = Path("C:/data/200_train_cases")
        assert_development_path(root, allow_dataset_root=True)
        with self.assertRaises(PermissionError):
            assert_development_path(root)

    def test_named_target_cannot_enter_inference(self):
        guard = Guard(install=False)
        with guard.inferring(), self.assertRaises(PermissionError):
            guard.audit("open", ("C:/data/200_train_cases/0014/x_target_named.json", "r"))

    def test_gt_never_enters_inference(self):
        guard = Guard(install=False)
        with guard.inferring(), self.assertRaises(PermissionError):
            guard.audit("open", ("C:/data/200_train_cases/0014/x_target.json", "r"))

    def test_new_slots_exactly_match_frozen_combination(self):
        import pin_combined_l38 as legacy
        from pcb.pin.frozen_l39 import locator
        front = ComponentStageOutput([
            Component("U1", "box", (20., 20., 65., 70.), body_bbox=(20., 20., 65., 70.)),
            Component("R1", "r", (75., 35., 85., 45.), body_bbox=(75., 35., 85., 45.))],
            [Text("1", (12, 31, 17, 39), .9), Text("GND", (23, 31, 40, 39), .9)], [], {})
        front.roles = classify_tokens(front.texts, [front.components[0].bbox])
        image = np.full((100, 110, 3), 255, np.uint8)
        image[35, 8:21] = 0
        image[20:71, 20] = 0
        image[20:71, 65] = 0
        image[20, 20:66] = 0
        image[70, 20:66] = 0

        class TextFixture:
            def run(self, image, context):
                return TextStageOutput(front.texts)
        class ComponentFixture:
            def run(self, image, texts, context):
                return copy.deepcopy(front)
        registry = build_default_registry()
        registry.register("text", "fixture", TextFixture)
        registry.register("component", "fixture", ComponentFixture)
        registry.register("pin_localization", "fixture_l39", lambda: L39Localization(model=object(), device="cpu"))
        registry.register("pin_semantics", "fixture_l39", lambda: L39Semantics(FakeWords(), {}))
        original = PipelineConfig(text="fixture", component="fixture")
        context = PipelineContext("fixture.png", 110, 100, original, None)
        current = replace(original, pin_localization="fixture_l39", pin_semantics="fixture_l39", wire="v3_frame_guard")
        def identity(image, components, baseline, model, device):
            return baseline
        with patch.object(legacy, "locate_boxes", side_effect=identity), patch.object(locator, "locate_boxes", side_effect=identity):
            old = legacy.CombinedPipeline(original, None, "cpu", FakeWords(), {}, registry).run(image, context).candidate
            new = ModularPipeline(current, registry).run(image, replace(context, config=current))
        self.assertEqual(canonical(old.result), canonical(new.result))
        self.assertEqual(canonical(old.scene.diagnostics['pin_events']), canonical(new.scene.diagnostics['pin_events']))
        np.testing.assert_array_equal(old.wire.mask, new.wire.mask)


if __name__ == "__main__":
    unittest.main()
