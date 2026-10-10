"""Contracts for image-grounded public-library numbering, without GT."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest

from pcb.schema import Component, Pin, Scene, Text
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from pin_library_semantics_l37 import (
    library_candidates, model_matches, name_numbers, stage,
)


def fixture(pins, library=None):
    c = Component("U1", "box", (20, 20, 80, 80), name="U1",
                  body_bbox=(20, 20, 80, 80), pins=pins)
    scene = Scene(100, 100, [c], [Text("TEST123", (22, 81, 65, 90), .99)])
    return scene, library or {"TEST123": {"pins": [
        {"number": "1", "name": "A", "pin_id": "e800"},
        {"number": "2", "name": "B", "pin_id": "e900"}]}}


def pin(number, name, exportable=True, confidence=.99):
    return Pin(number, name, (10., 30.), base=(20., 30.), side="left",
               exportable=exportable, name_confidence=confidence)


class LibrarySemanticsTests(unittest.TestCase):
    def test_image_model_required(self):
        scene, lib = fixture([pin("UNK1", "A", False)])
        scene.texts = []
        new, trace = stage(scene, lib)
        self.assertEqual(asdict(new.components[0]), asdict(scene.components[0]))
        self.assertEqual(trace["accepted_changes"], [])

    def test_exact_model_no_fuzzy_repair(self):
        scene, lib = fixture([pin("UNK1", "A", False)])
        scene.texts[0].text = "TESTI23"
        self.assertFalse(model_matches(scene.components, scene.texts, lib)[0])

    def test_documented_suffix_keeps_variants_and_exact_key_preferred(self):
        lib = {"TEST123_C99": {}, "TEST123_C88": {}}
        self.assertEqual(library_candidates("TEST123", lib), ("TEST123_C88", "TEST123_C99"))
        lib["TEST123"] = {}
        self.assertEqual(library_candidates("TEST123", lib), ("TEST123",))

    def test_pin_id_not_used_as_number(self):
        scene, lib = fixture([pin("UNK1", "A", False)])
        new, trace = stage(scene, lib)
        p = new.components[0].pins[0]
        self.assertEqual(p.number, "1")
        self.assertTrue(p.inferred_number)
        self.assertIsNone(p.observable_number)
        self.assertFalse(trace["public_library_pin_id_used"])

    def test_aggregate_conflict_not_resolved_by_frequency(self):
        scene, lib = fixture([pin("UNK1", "A", False)])
        lib["TEST123"]["pins"].extend([{"number": "1", "name": "A"}]*8 + [{"number": "3", "name": "A"}])
        self.assertEqual(name_numbers(("TEST123",), lib, "A"), ("1", "3"))
        self.assertFalse(stage(scene, lib)[1]["accepted_changes"])

    def test_simultaneous_number_release_chain(self):
        scene, lib = fixture([pin("2", "A"), pin("UNK2", "B", False)])
        new, trace = stage(scene, lib)
        self.assertEqual([p.number for p in new.components[0].pins], ["1", "2"])
        self.assertEqual(len(trace["accepted_changes"]), 2)

    def test_reserved_number_keeps_original(self):
        scene, lib = fixture([pin("1", "UNRELATED"), pin("UNK2", "A", False)])
        new, trace = stage(scene, lib)
        self.assertEqual(asdict(new.components[0]), asdict(scene.components[0]))
        self.assertEqual(trace["pin_reviews"][1]["status"], "number_reserved_by_unchanged_pin")

    def test_duplicate_proposals_abstain(self):
        scene, lib = fixture([pin("UNK1", "A", False), pin("UNK2", "A", False)])
        new, trace = stage(scene, lib)
        self.assertEqual(asdict(new.components[0]), asdict(scene.components[0]))
        self.assertEqual(len(trace["accepted_changes"]), 0)

    def test_rejected_number_release_propagates(self):
        scene, lib = fixture([pin("2", "A"), pin("3", "B"), pin("1", "UNRELATED")])
        new, trace = stage(scene, lib)
        self.assertEqual(asdict(new.components[0]), asdict(scene.components[0]))
        self.assertEqual(len(trace["accepted_changes"]), 0)

    def test_geometry_names_order_component_and_inputs_preserved(self):
        scene, lib = fixture([pin("UNK1", "a_", False)])
        original, original_lib = asdict(scene), copy.deepcopy(lib)
        new, _ = stage(scene, lib)
        self.assertEqual(asdict(scene), original)
        self.assertEqual(lib, original_lib)
        c, old = new.components[0], scene.components[0]
        self.assertEqual((c.key, c.type, c.bbox, c.body_bbox, c.name, c.value),
                         (old.key, old.type, old.bbox, old.body_bbox, old.name, old.value))
        self.assertEqual((c.pins[0].tip, c.pins[0].base, c.pins[0].side, c.pins[0].name),
                         (old.pins[0].tip, old.pins[0].base, old.pins[0].side, old.pins[0].name))
        self.assertEqual(c.pins[0].number, "1")
        self.assertIn("public_library", c.pins[0].number_source)

    def test_unresolved_and_nonbox_unchanged(self):
        scene, lib = fixture([pin("UNK1", "A", False)])
        for key, typ in [("UNRESOLVED_0001", "box"), ("R1", "r")]:
            scene.components[0].key, scene.components[0].type = key, typ
            new, _ = stage(scene, lib)
            self.assertEqual(asdict(new.components[0]), asdict(scene.components[0]))

    def test_model_ownership_ties_abstain(self):
        scene, lib = fixture([pin("UNK1", "A", False)])
        second = copy.deepcopy(scene.components[0]); second.key = "U2"
        scene.components.append(second)
        self.assertEqual(stage(scene, lib)[1]["models"], {})

    def test_low_name_and_model_confidence_not_accepted(self):
        scene, lib = fixture([pin("UNK1", "A", False, .3)])
        self.assertEqual(stage(scene, lib)[1]["accepted_changes"], [])
        scene.components[0].pins[0].name_confidence = .99
        scene.texts[0].score = .3
        self.assertEqual(stage(scene, lib)[1]["accepted_changes"], [])

    def test_existing_event_provenance_retained(self):
        scene, lib = fixture([pin("UNK1", "A", False)])
        scene.diagnostics["pin_events"] = [{"component": "U1", "number": "UNK1", "number_token": None,
                                           "name_token": "A", "side": "left", "tip": [10, 30]}]
        new, _ = stage(scene, lib)
        event = new.diagnostics["pin_events"][0]
        self.assertEqual(event["name_token"], "A")
        self.assertIsNone(event["number_token"])
        self.assertEqual(event["number"], "1")


if __name__ == "__main__":
    unittest.main()
