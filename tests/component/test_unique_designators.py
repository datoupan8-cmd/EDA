import unittest

from pcb.component.latest.text_assignment import assign_designators, build_components
from pcb.component.latest.text_roles import classify_tokens
from pcb.component.proposal_fusion import ComponentProposal
from pcb.component.unique_designators import assign_unique_designators
from pcb.schema import Text


class UniqueDesignatorTests(unittest.TestCase):
    def test_duplicate_readings_cannot_consume_same_reference_twice(self):
        proposals = [ComponentProposal((10, 10, 30, 22), "r", .95, "yolo", {}),
                     ComponentProposal((55, 10, 75, 22), "r", .80, "yolo", {})]
        roles = classify_tokens([Text("R1", (48, 12, 64, 22), .99),
                                 Text("R1", (47, 11, 65, 23), .98)])
        legacy, _ = assign_designators(proposals, roles, global_assignment=True)
        self.assertEqual(len(legacy), 2)
        legacy_components, _ = build_components(proposals, legacy)
        self.assertEqual(legacy_components[0].key, "R1")
        assigned, trace = assign_unique_designators(proposals, roles)
        self.assertEqual(set(assigned), {1})
        self.assertEqual(assigned[1].designator, "R1")
        self.assertEqual(trace["unique_identifier_count"], 1)
        self.assertEqual(trace["duplicate_ocr_candidates"], 1)

    def test_distinct_references_keep_type_compatible_global_assignment(self):
        proposals = [ComponentProposal((10, 10, 30, 22), "r", .95, "yolo", {}),
                     ComponentProposal((55, 10, 75, 22), "c", .95, "yolo", {})]
        roles = classify_tokens([Text("R1", (10, 0, 25, 8), .99),
                                 Text("C2", (55, 0, 70, 8), .99)])
        assigned, _ = assign_unique_designators(proposals, roles)
        self.assertEqual({index: role.designator for index, role in assigned.items()}, {0: "R1", 1: "C2"})

    def test_empty_input_and_pin_readings_do_not_create_reference_identifiers(self):
        proposals = [ComponentProposal((10, 10, 30, 22), "r", .95, "yolo", {})]
        roles = classify_tokens([Text("100", (10, 0, 25, 8), .99)])
        self.assertEqual(assign_unique_designators(proposals, roles)[0], {})
        self.assertEqual(assign_unique_designators([], roles)[0], {})


if __name__ == "__main__":
    unittest.main()
