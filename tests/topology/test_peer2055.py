import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

from pcb.topology_stage import peer_protocol2055 as peer
from tests.topology.test_protocol_202610 import doc, component, net


class Peer2055(unittest.TestCase):
    def test_scoring_definitions_match_original_archive_fingerprint(self):
        root = Path(__file__).resolve().parents[2]
        expected = json.loads((root/'docs/network/peer2055_provenance.json').read_text('utf-8'))
        tree = ast.parse(Path(peer.__file__).read_text('utf-8'))
        definitions = [ast.dump(n, include_attributes=False) for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        digest = hashlib.sha256('\n'.join(definitions).encode()).hexdigest()
        self.assertEqual(digest, expected['semantic_definition_sha256'])
        self.assertTrue(expected['assignment_semantics_equal'])

    def test_network_score_averages_network_overlap_instead_of_membership_counts(self):
        g = doc()
        component(g, 'U1', 'box')
        component(g, 'U2', 'box')
        net(g, 'a', 'U1.1', 'U1.2', 'U1.3')
        net(g, 'b', 'U2.1')
        p = copy.deepcopy(g)
        net(p, 'a', 'U1.1', 'U1.2', 'U1.4')
        score = peer.evaluate(p, g)
        self.assertAlmostEqual(score['metrics']['NetHypergraph']['f1'], 5/6)
        self.assertNotAlmostEqual(score['metrics']['NetHypergraph']['f1'], .75)

    def test_supply_missing_value_behavior_is_preserved_and_explicit(self):
        self.assertEqual(peer.rail({'type': 'v', 'value': None}), peer.rail({'type': 'v', 'value': 'VCC'}))
        self.assertNotEqual(peer.rail({'type': 'gnd'}), peer.rail({'type': 'v', 'value': 'GND'}))
        self.assertEqual(peer.evaluate(doc(), doc())['metrics']['Pin']['f1'], 0)


if __name__ == '__main__':
    unittest.main()
