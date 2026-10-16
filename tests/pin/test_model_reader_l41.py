from dataclasses import asdict
import copy
import unittest
import numpy as np

from pcb.schema import Component, Pin, Scene, Text
from pcb.pin.model_reader_l41 import apply, evidence_for_owner, owner_roi, unrotate_bbox
from pcb.pin.local_ocr import rotate_points


LIBRARY = {'TEST123': {'pins': [{'number': '7', 'name': 'SIG'}]},
           'OTHER456': {'pins': [{'number': '9', 'name': 'SIG'}]}}


class FakeReader:
    def __init__(self, tokens=()):
        self.tokens = list(tokens)
        self.calls = self.cache_hits = 0
        self.model_signature = {'det': 'fixed', 'rec': 'fixed'}

    def read_owner(self, image, component, texts):
        self.calls += 4
        return copy.deepcopy(self.tokens)


def fixture(local=None):
    pin = Pin('1', 'SIG', (-10, 20), base=(0, 20), side='left', name_confidence=.99)
    box = Component('U1', 'box', (0, 0, 100, 100), pins=[pin], body_bbox=(0, 0, 100, 100))
    diag = {'L37_public_library': {'models': {}}}
    if local:
        diag['pin_l39_stage_trace'] = {'joint': {'role_preparation': [
            {'component': 'U1', 'canonical_tokens': [asdict(t) for t in local]}]}}
    return Scene(300, 300, [box], [], diagnostics=diag), np.full((300, 300, 3), 255, np.uint8)


class ModelReaderTests(unittest.TestCase):
    def test_existing_local_model_can_unlock_frozen_number_lookup(self):
        a, image = fixture([Text('TEST123', (10, 10, 60, 20), .99)])
        before = copy.deepcopy(asdict(a))
        reader = FakeReader()
        b = apply(image, a, reader, LIBRARY, {})
        self.assertEqual(b.components[0].pins[0].number, '7')
        self.assertEqual(reader.calls, 0)
        self.assertEqual(asdict(a), before)
        self.assertEqual(b.texts, a.texts)
        self.assertEqual(b.components[0].pins[0].tip, a.components[0].pins[0].tip)
        self.assertEqual(b.components[0].pins[0].name, a.components[0].pins[0].name)
        self.assertIn('U1', b.diagnostics['L41_model_reading']['new_models'])

    def test_whole_component_read_can_unlock_same_lookup(self):
        a, image = fixture()
        reader = FakeReader([Text('TEST123', (10, 10, 60, 20), .99)])
        b = apply(image, a, reader, LIBRARY, {})
        self.assertEqual(reader.calls, 4)
        self.assertEqual(b.components[0].pins[0].number, '7')
        self.assertFalse(b.diagnostics['L41_model_reading']['global_OCR_changed'])

    def test_existing_trusted_model_is_never_overridden(self):
        a, image = fixture()
        a.diagnostics['L37_public_library']['models'] = {'U1': {'library_keys': ['TEST123'], 'model_confidence': .99}}
        reader = FakeReader([Text('OTHER456', (10, 10, 60, 20), .99)])
        b = apply(image, a, reader, LIBRARY, {})
        self.assertEqual(reader.calls, 0)
        self.assertEqual(asdict(a.components[0]), asdict(b.components[0]))

    def test_conflicting_model_texts_abstain(self):
        a, image = fixture()
        reader = FakeReader([Text(k, (10, 10, 70, 20), .99) for k in LIBRARY])
        b = apply(image, a, reader, LIBRARY, {})
        self.assertEqual(b.diagnostics['L41_model_reading']['new_models'], {})
        self.assertEqual(asdict(a.components[0]), asdict(b.components[0]))

    def test_neighbor_model_cannot_be_owned_by_source_crop(self):
        a, _ = fixture()
        a.components.append(Component('U2', 'box', (150, 0, 250, 100), body_bbox=(150, 0, 250, 100)))
        tokens, events = evidence_for_owner(a, 'U1', [Text('TEST123', (170, 10, 220, 20), .99)], LIBRARY, 'new')
        self.assertEqual(tokens, [])
        self.assertEqual(events[0]['status'], 'other_owner_or_ambiguous')

    def test_library_missing_or_low_confidence_is_not_guessed(self):
        a, image = fixture()
        reader = FakeReader([Text('TEST12?', (10, 10, 70, 20), .99), Text('TEST123', (10, 10, 70, 20), .79)])
        b = apply(image, a, reader, LIBRARY, {})
        self.assertEqual(asdict(a.components[0]), asdict(b.components[0]))

    def test_nonbox_and_unresolved_not_read(self):
        a, image = fixture()
        a.components[0].type = 'r'
        a.components.append(Component('UNRESOLVED_0001', 'box', (150, 0, 250, 100)))
        reader = FakeReader()
        b = apply(image, a, reader, LIBRARY, {})
        self.assertEqual(reader.calls, 0)
        self.assertEqual(asdict(a.components[0]), asdict(b.components[0]))

    def test_quarter_turn_coordinates_return_to_original(self):
        roi = (30, 40, 80, 70)
        pad, sx, sy, pw, ph = 12, 3., 3., 174, 114
        polygon = np.array([[42., 42.], [102., 42.], [102., 72.], [42., 72.]])
        for angle in (0, 90, 180, 270):
            rotated = rotate_points(polygon, pw, ph, angle)
            b = unrotate_bbox(rotated, pw, ph, angle, sx, sy, pad, roi)
            np.testing.assert_allclose(b, (40, 50, 60, 60))

    def test_roi_preserves_full_intersecting_title_without_transitive_expansion(self):
        a, image = fixture()
        texts = [Text('LONGMODEL123', (90, 95, 190, 105), .99), Text('NEIGHBOR456', (185, 95, 290, 105), .99)]
        self.assertEqual(owner_roi(a.components[0], texts, image.shape), (0, 0, 192, 124))

    def test_repeated_application_preserves_new_trusted_model(self):
        a, image = fixture()
        reader = FakeReader([Text('TEST123', (10, 10, 60, 20), .99)])
        b = apply(image, a, reader, LIBRARY, {})
        calls = reader.calls
        c = apply(image, b, reader, LIBRARY, {})
        self.assertEqual(reader.calls, calls)
        self.assertEqual(asdict(b.components[0]), asdict(c.components[0]))


if __name__ == '__main__':
    unittest.main()
