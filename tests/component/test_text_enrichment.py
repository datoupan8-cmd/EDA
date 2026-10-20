import copy
from dataclasses import asdict
import unittest

from pcb.component.latest.text_roles import classify_tokens
from pcb.component.latest.text_assignment import assign_values
from pcb.component.text_enrichment import enrich_fields
from pcb.schema import Component, Pin, Text


class ComponentTextEnrichmentTests(unittest.TestCase):
    def test_default_changes_only_component_text_selection_and_preserves_baseline(self):
        from pcb.core.team_config import load_team_config
        from pcb.core.registry import build_default_registry
        baseline = load_team_config("configs/component/team_peer_latest.json").as_dict()
        current = load_team_config("configs/team.json").as_dict()
        self.assertEqual(baseline.pop("component"), "peer_latest")
        self.assertEqual(current.pop("component"), "ordered_yolo11m_local_parseq_p01_v1")
        self.assertEqual(baseline.pop("text"), "current")
        self.assertEqual(current.pop("text"), "ppocrv5_beam")
        self.assertEqual(current, baseline)
        self.assertIn("peer_latest", build_default_registry().versions("component"))
        self.assertIn("contextual_values_v2", build_default_registry().versions("component"))
        self.assertIn("ordered_value_groups_v2", build_default_registry().versions("component"))
        self.assertIn("ordered_values_yolo11m_v1", build_default_registry().versions("component"))
        self.assertIn("ordered_yolo11m_local_parseq_v1", build_default_registry().versions("component"))
        self.assertIn("ordered_yolo11m_local_parseq_p01_v1", build_default_registry().versions("component"))

    def apply(self, components, texts, **options):
        roles = classify_tokens(texts, [c.bbox for c in components if c.type == "box"])
        return enrich_fields(components, roles, {}, **options)[0]

    def test_printed_capacitance_shorthand_is_preserved_literally(self):
        for value in ("27p", "100n", "4.7u", "10µ", "10μ", "0.1m"):
            with self.subTest(value=value):
                owner = Component("C1", "c", (40, 40, 60, 60), name="C1")
                output = self.apply([owner], [Text(value, (65, 45, 85, 55), .99)], names=False)
                self.assertEqual(output[0].value, value)
                self.assertIsNone(owner.value)

    def test_shorthand_is_not_assigned_to_resistors_or_from_bare_numbers(self):
        owners = [Component("R1", "r", (40, 40, 60, 60)), Component("C1", "c", (100, 40, 120, 60))]
        output = self.apply(owners, [Text("100n", (65, 45, 85, 55), .99), Text("100", (125, 45, 145, 55), .99)], names=False)
        self.assertIsNone(output[0].value)
        # A separate nearby capacitor may use the explicit 100n token; the
        # unitless number itself must never become its capacitance.
        self.assertNotEqual(output[1].value, "100")

    def test_unitless_value_alone_does_not_fill_capacitor(self):
        output = self.apply([Component("C1", "c", (40, 40, 60, 60))], [Text("100", (65, 45, 85, 55), .99)], names=False)
        self.assertIsNone(output[0].value)

    def test_one_physical_value_token_is_assigned_once(self):
        owners = [Component("C1", "c", (40, 40, 60, 60)), Component("C2", "c", (80, 40, 100, 60))]
        output = self.apply(owners, [Text("100n", (65, 45, 75, 55), .99)], names=False)
        self.assertEqual(sum(c.value is not None for c in output), 1)

    def test_value_token_claimed_by_existing_stage_is_not_reused(self):
        owners = [Component("R1", "r", (40, 40, 60, 60)), Component("C2", "c", (80, 40, 100, 60))]
        roles = classify_tokens([Text("100m", (65, 45, 75, 55), .99)])
        trace = assign_values(owners, roles, [{"component": c.key} for c in owners])
        self.assertEqual(owners[0].value, "100m")
        output, _ = enrich_fields(owners, roles, {"value_assignment": trace}, names=False)
        self.assertIsNone(output[1].value)

    def test_existing_value_and_model_name_are_preserved(self):
        owners = [Component("C1", "c", (40, 40, 60, 60), value="47n"),
                  Component("U1", "box", (100, 100, 200, 200), name="LM324")]
        output = self.apply(owners, [Text("100n", (65, 45, 85, 55), .99), Text("STM32F103C8T6", (130, 80, 170, 90), .99)])
        self.assertEqual(output[0].value, "47n")
        self.assertEqual(output[1].name, "LM324")

    def test_visible_model_replaces_designator_name_without_other_changes(self):
        owner = Component("U1", "box", (100, 100, 200, 200), name="U1", body_bbox=(100, 100, 200, 200),
                          pins=[Pin("1", "IN", (90, 120), base=(100, 120), side="left")])
        before = copy.deepcopy(asdict(owner))
        output = self.apply([owner], [Text("STM32F103C8T6", (130, 80, 170, 90), .99)], shorthand=False)
        after = asdict(output[0])
        self.assertEqual(after.pop("name"), "STM32F103C8T6")
        before.pop("name")
        self.assertEqual(after, before)
        self.assertEqual(owner.name, "U1")

    def test_signal_names_and_missing_ocr_do_not_generate_model_names(self):
        owner = Component("U1", "box", (100, 100, 200, 200), name="U1")
        for texts in ([], [Text("GPIO1", (130, 110, 170, 120), .99)]):
            self.assertEqual(self.apply([owner], texts, shorthand=False)[0].name, "U1")

    def test_pin_signals_and_package_annotations_are_not_chip_models(self):
        owner = Component("U1", "box", (100, 100, 200, 200), name="U1")
        for label in ("PC15-OSC32_OUT", "ADC7", "KEY2", "SOP8", "CAP/2.5V"):
            with self.subTest(label=label):
                self.assertEqual(self.apply([owner], [Text(label, (130, 110, 170, 120), .99)],
                                           shorthand=False, known_models=frozenset())[0].name, "U1")

    def test_public_model_evidence_can_resolve_value_or_pin_number_roles(self):
        owner = Component("U1", "box", (100, 100, 200, 200), name="U1")
        for model in ("2.4GHz", "L298N"):
            with self.subTest(model=model):
                output = self.apply([owner], [Text(model, (130, 110, 170, 120), .99)],
                                    shorthand=False, known_models=frozenset({model.upper()}))
                self.assertEqual(output[0].name, model)

    def test_ball_number_shaped_token_without_model_context_is_not_promoted(self):
        owner = Component("U1", "box", (100, 100, 200, 200), name="U1")
        self.assertEqual(self.apply([owner], [Text("AL358", (130, 110, 170, 120), .99)], shorthand=False)[0].name, "U1")


if __name__ == "__main__":
    unittest.main()
