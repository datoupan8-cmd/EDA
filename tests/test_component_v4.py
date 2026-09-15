from __future__ import annotations

import unittest
from pathlib import Path

from main import official_images
from pcb.data_policy import assert_allowed_case_id, split_for_case
from pcb.component_detector_yolo import class_aware_nms, offset_bbox, short_cache_key, tile_origins
from pcb.component_proposal_fusion import ComponentProposal, fuse_proposals
from pcb.component_text_assignment import (
    assign_designators,
    assign_names,
    assign_values,
    build_components,
    type_designator_compatible,
)
from pcb.schema import Scene, Text
from pcb.submission import export, validate_strict
from pcb.text_detection import classify_tokens, value_compatible


class ComponentV4Tests(unittest.TestCase):
    def test_yolo_bbox_coordinate_offset(self):
        self.assertEqual(offset_bbox((1, 2, 11, 22), 100, 200), (101.0, 202.0, 111.0, 222.0))

    def test_tile_origins_cover_trailing_edge(self):
        self.assertEqual(tile_origins(2500, 1000, 100), [0, 900, 1500])

    def test_yolo_cache_key_is_short_for_deep_windows_paths(self):
        import numpy as np
        self.assertEqual(len(short_cache_key(np.zeros((8, 8, 3), dtype=np.uint8), b"config")), 24)

    def test_class_aware_nms_merges_same_type_only(self):
        rows = [
            ComponentProposal((0, 0, 20, 20), "r", .9, "yolo"),
            ComponentProposal((1, 1, 21, 21), "r", .8, "yolo"),
            ComponentProposal((1, 1, 21, 21), "c", .7, "yolo"),
        ]
        kept, rejected = class_aware_nms(rows, .5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(rejected), 1)

    def test_type_designator_compatibility(self):
        roles = classify_tokens([Text("R12", (0, 0, 10, 8), .9), Text("C2", (20, 0, 30, 8), .9)])
        self.assertTrue(type_designator_compatible("r", roles[0]))
        self.assertFalse(type_designator_compatible("c", roles[0]))

    def test_audited_resistor_aliases_are_supported(self):
        for value in ("RC1", "RES2", "RP3", "RT4", "RV5", "RVC6"):
            role = classify_tokens([Text(value, (0, 0, 20, 8), .9)])[0]
            self.assertEqual(role.role, "DESIGNATOR")
            self.assertTrue(type_designator_compatible("r", role))

    def test_unseen_r_prefix_is_not_accepted(self):
        role = classify_tokens([Text("REF5040", (0, 0, 30, 8), .9)])[0]
        self.assertNotEqual(role.role, "DESIGNATOR")

    def test_transistor_and_diode_prefix_families(self):
        q = classify_tokens([Text("Q3", (0, 0, 20, 8), .9)])[0]
        d = classify_tokens([Text("D4", (0, 0, 20, 8), .9)])[0]
        self.assertTrue(type_designator_compatible("bjt", q))
        self.assertTrue(type_designator_compatible("mosfet", q))
        self.assertTrue(type_designator_compatible("led", d))
        self.assertTrue(type_designator_compatible("esd", d))

    def test_expanded_value_units_are_type_specific(self):
        self.assertTrue(value_compatible("crystal_2pin", "16MHz"))
        self.assertTrue(value_compatible("fuse", "2A"))
        self.assertTrue(value_compatible("v", "3.3V"))
        self.assertFalse(value_compatible("c", "2A"))
        self.assertFalse(value_compatible("r", "10uF"))

    def test_designator_is_not_assigned_twice(self):
        proposals = [
            ComponentProposal((0, 15, 12, 25), "r", .9, "yolo"),
            ComponentProposal((14, 15, 26, 25), "r", .8, "yolo"),
        ]
        roles = classify_tokens([Text("R1", (6, 0, 18, 9), .95)])
        assigned, _ = assign_designators(proposals, roles, global_assignment=True)
        self.assertEqual(len(assigned), 1)

    def test_value_is_not_a_designator(self):
        roles = classify_tokens([Text("10k", (0, 0, 12, 8), .9)])
        self.assertEqual(roles[0].role, "VALUE")

    def test_pin_name_is_not_component_name(self):
        proposals = [ComponentProposal((0, 0, 100, 100), "box", .9, "yolo")]
        components, provenance = build_components(proposals, {})
        roles = classify_tokens([Text("GPIO4", (20, 20, 60, 30), .9)])
        assign_names(components, roles, provenance)
        self.assertIsNone(components[0].name)

    def test_net_label_is_not_designator(self):
        roles = classify_tokens([Text("SDA", (0, 0, 20, 8), .9)])
        self.assertNotEqual(roles[0].role, "DESIGNATOR")

    def test_value_assignment_is_one_to_one(self):
        proposals = [
            ComponentProposal((0, 15, 12, 25), "r", .9, "yolo"),
            ComponentProposal((20, 15, 32, 25), "r", .9, "yolo"),
        ]
        components, provenance = build_components(proposals, {})
        roles = classify_tokens([Text("10k", (8, 0, 20, 8), .95)])
        assign_values(components, roles, provenance)
        self.assertEqual(sum(component.value == "10k" for component in components), 1)

    def test_spatial_fusion_removes_geometry_copy(self):
        rows = [
            ComponentProposal((0, 0, 20, 20), "c", .9, "yolo"),
            ComponentProposal((1, 1, 21, 21), "c", .58, "capacitor_geometry"),
        ]
        kept, rejected = fuse_proposals(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].source, "yolo")
        self.assertEqual(len(rejected), 1)

    def test_unresolved_detection_is_not_fake_designator(self):
        components, _ = build_components([ComponentProposal((1, 1, 8, 8), "r", .9, "yolo")], {})
        self.assertTrue(components[0].key.startswith("UNRESOLVED_"))
        self.assertIsNone(components[0].observable_designator)

    def test_v4_component_exports_official_schema(self):
        components, _ = build_components([ComponentProposal((10, 10, 30, 30), "r", .9, "yolo")], {})
        result = export(Scene(100, 100, components))
        self.assertTrue(validate_strict(result, (100, 100)))

    def test_holdout_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            official_images(Path("unused"), 1, 151)

    def test_training_policy_seals_case_151(self):
        self.assertEqual(split_for_case(5), "dev")
        self.assertEqual(split_for_case(6), "train")
        with self.assertRaises(PermissionError):
            assert_allowed_case_id(151)


if __name__ == "__main__":
    unittest.main()
