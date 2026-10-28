import copy
import unittest

from pcb.topology_stage.identity_cause_audit import (
    geometry_matches, body_cause, inspect_case, aggregate,
)


def component(key, kind='r', box=(10, 10, 30, 30), value=None):
    return {'key': key, 'type': kind, 'bbox': list(box), 'value': value, 'confidence': .95}


def scene(text='R1', role='DESIGNATOR'):
    return {'height': 100, 'components': [], 'texts': [
        {'text': text, 'bbox': [11, 1, 28, 8], 'score': .99}],
        'diagnostics': {'text_roles': [
            {'index': 0, 'text': text, 'bbox': [11, 1, 28, 8], 'confidence': .99,
             'role': role, 'normalized': text.upper()}]}}


def cause(target, final, before, source, power=False):
    fm = geometry_matches([target], final)[0]
    bm = geometry_matches([target], before)[0]
    return body_cause(target, final, before, fm, bm, source, power)


class IdentityCauseAudit(unittest.TestCase):
    def test_geometry_ignores_names_and_type_and_does_not_reuse_a_box(self):
        gt = [component('R1'), component('R2', box=(50, 50, 70, 70))]
        pred = [component('R1', kind='c', box=(50, 50, 70, 70)), component('R2')]
        self.assertEqual([r['index'] for r in geometry_matches(gt, pred)], [1, 0])
        ambiguous = geometry_matches([gt[0], copy.deepcopy(gt[0])], [pred[1]])
        self.assertEqual([r['status'] for r in ambiguous], ['ambiguous', 'ambiguous'])
        self.assertEqual(sum(r['index'] is not None for r in ambiguous), 1)

    def test_case_mismatch_is_separate_from_missing_ocr_and_keeps_raw_case(self):
        target, pred = component('r1'), component('R1')
        result = cause(target, [pred], [pred], scene('R1'))
        self.assertEqual(result['cause'], 'case_only_key')
        self.assertEqual(result['ocr']['literal_count'], 1)
        self.assertEqual(result['ocr']['exact_case_literal_count'], 0)
        result = cause(target, [pred], [pred], scene('r1'))
        self.assertEqual(result['ocr']['exact_case_literal_count'], 1)

    def test_correct_candidate_vs_role_gate_vs_missing_literal(self):
        target, pred = component('R1'), component('UNRESOLVED_1')
        self.assertEqual(cause(target, [pred], [pred], scene())['cause'],
                         'candidate_assignment_or_key_collision')
        self.assertEqual(cause(target, [pred], [pred], scene(role='PIN_NUMBER'))['cause'],
                         'literal_seen_but_not_designator_candidate')
        self.assertEqual(cause(target, [pred], [pred], scene('R2'))['cause'],
                         'correct_literal_not_in_archived_OCR')

    def test_filtered_correct_identity_is_not_called_an_ocr_error(self):
        target = component('R1')
        self.assertEqual(cause(target, [], [target], scene())['cause'], 'correct_identity_filtered')
        self.assertEqual(cause(target, [], [component('UNRESOLVED_1')], scene())['cause'],
                         'body_filtered_with_wrong_or_unresolved_identity')
        self.assertEqual(cause(target, [], [], scene())['cause'], 'no_prefilter_box_at_diagnostic_IoU')

    def test_nonempty_power_error_is_distinct_from_empty_and_case(self):
        target = component('p', 'v', value='3.3V')
        for value, expected in [(None, 'power_value_empty'), ('5V', 'power_value_wrong_nonempty'),
                                ('3.3v', 'power_value_case_or_space_only')]:
            pred = component('PRED_PWR', 'v', value=value)
            result = cause(target, [pred], [pred], scene('3.3V', 'VALUE'), power=True)
            self.assertEqual(result['cause'], expected)

    def test_filtered_secondary_counts_preserve_units_and_role_evidence(self):
        source = scene('R1', role='PIN_NUMBER')
        source['diagnostics']['candidate_audit_before_filter'] = {'components': [component('UNRESOLVED_1')]}
        pred = {'components': {}, 'pins': {}, 'nets': {}}
        gt = {'components': {'R1': {'type': 'r', 'bbox': [10, 70, 30, 90], 'value': None}},
              'pins': {}, 'nets': {'n': {'hyperGraph': '(R1.1)', 'edges': {}}}}
        previous = {'identity': {'exclusive_losses': {'component_identity_missing': {'component': 1}}}}
        result = aggregate({'0001': inspect_case(source, pred, gt, previous)})
        self.assertEqual(result['filtered_body_identity_state']['owners'], {'unresolved': 1})
        self.assertEqual(result['filtered_body_text_evidence']['members'],
                         {'literal_seen_but_not_designator_candidate': 1})
        self.assertEqual(result['missing_nonrail_root_situations'],
                         {'literal_seen_but_not_designator_candidate': 1})
        self.assertEqual(result['filtered_literal_role_evidence_not_exclusive'], {'PIN_NUMBER:': 1})

    def test_conservation_weights_pins_but_counts_repeated_rail_once(self):
        native = [component('U1', 'box'), component('V1', 'v', box=(50, 50, 60, 60))]
        source = scene('U1')
        source['components'] = native
        source['diagnostics']['candidate_audit_before_filter'] = {'components': native}
        # Convert top-origin native boxes to bottom-origin export schema.
        exported = {c['key']: {**c, 'bbox': [c['bbox'][0], 100-c['bbox'][3], c['bbox'][2], 100-c['bbox'][1]]}
                    for c in native}
        target_cs = {'u1': {**exported['U1'], 'key': 'u1'},
                     'p1': {**exported['V1'], 'value': '3.3V'},
                     'p2': {**exported['V1'], 'bbox': [70, 40, 80, 50], 'value': '3.3V'}}
        pred = {'components': exported, 'pins': {}, 'nets': {}}
        gt = {'components': target_cs, 'pins': {}, 'nets': {
            'n': {'hyperGraph': '(u1.1,u1.2,p1.1,p2.1)', 'edges': {}}}}
        previous = {'identity': {'exclusive_losses': {'component_identity_missing': {'ic': 2, 'rail': 1}}}}
        row = inspect_case(source, pred, gt, previous)
        self.assertEqual(row['original_missing_member_count'], 3)
        self.assertEqual(row['missing_nonrail_members'], {'case_only_key': 2})
        self.assertEqual(row['missing_owner_count'], 1)
        self.assertEqual(row['missing_rail_count'], 1)
        self.assertEqual(row['missing_rails'][0]['symbol_count'], 2)
        self.assertEqual(aggregate({'0001': row})['original_missing_member_count'], 3)


if __name__ == '__main__':
    unittest.main()
