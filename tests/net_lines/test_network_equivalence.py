import unittest

from tests.net_lines.network_equivalence import edge_graph_summary, normalize_submission


def _result(networks):
    """Build only the exported nets shape used by the public result schema."""
    return {
        "nets": {
            net_id: {
                "hyperGraph": "(" + ",".join(members) + ")",
                "edges": {
                    f"edge_{index}": [
                        {"x": a[0], "y": a[1]},
                        {"x": b[0], "y": b[1]},
                    ]
                    for index, (a, b) in enumerate(edges, 1)
                },
            }
            for net_id, members, edges in networks
        }
    }


class NetworkEquivalenceTests(unittest.TestCase):
    def test_collinear_split_merge_and_order_are_semantically_equal(self):
        merged = _result([("net_7", ["U1.1", "U2.1"], [((0, 0), (6, 0))])])
        split_reversed = _result([("other_id", ["U2.1", "U1.1"], [((6, 0), (3, 0)), ((3, 0), (0, 0))])])
        self.assertEqual(normalize_submission(merged), normalize_submission(split_reversed))

    def test_turn_is_not_removed_by_normalization(self):
        turn = _result([("net_1", ["U1.1"], [((0, 0), (2, 0)), ((2, 0), (2, 2))])])
        straight = _result([("net_1", ["U1.1"], [((0, 0), (2, 2))])])
        normalized = normalize_submission(turn)
        self.assertNotEqual(normalized, normalize_submission(straight))
        edges = next(iter(normalized.values()))
        self.assertIn((2.0, 0.0), {point for edge in edges for point in edge})

    def test_membership_change_is_not_hidden_by_same_geometry(self):
        first = _result([("net_1", ["U1.1", "U2.1"], [((0, 0), (4, 0))])])
        reassigned = _result([("net_1", ["U1.1", "U2.2"], [((0, 0), (4, 0))])])
        self.assertNotEqual(normalize_submission(first), normalize_submission(reassigned))

    def test_disconnected_geometry_is_a_real_difference(self):
        connected = _result([("net_1", ["U1.1", "U2.1"], [((0, 0), (2, 0)), ((2, 0), (4, 0))])])
        disconnected = _result([("net_1", ["U1.1", "U2.1"], [((0, 0), (2, 0)), ((4, 0), (6, 0))])])
        connected_edges = next(iter(normalize_submission(connected).values()))
        disconnected_edges = next(iter(normalize_submission(disconnected).values()))
        self.assertNotEqual(connected_edges, disconnected_edges)
        self.assertEqual(edge_graph_summary(connected_edges)["component_count"], 1)
        self.assertEqual(edge_graph_summary(disconnected_edges)["component_count"], 2)

    def test_branch_junction_is_preserved(self):
        branch = _result([("net_1", ["U1.1"], [((0, 0), (2, 0)), ((2, 0), (4, 0)), ((2, 0), (2, 2))])])
        edges = next(iter(normalize_submission(branch).values()))
        self.assertEqual(edge_graph_summary(edges)["degree_3_plus_count"], 1)
        self.assertEqual(len(edges), 3)

    def test_zero_length_contact_edge_is_preserved(self):
        with_contact = _result([("net_1", ["U1.1"], [((0, 0), (2, 0)), ((3, 3), (3, 3))])])
        normalized = next(iter(normalize_submission(with_contact).values()))
        self.assertIn(((3.0, 3.0), (3.0, 3.0)), normalized)
        self.assertEqual(edge_graph_summary(normalized)["component_count"], 2)

    def test_networks_are_normalized_independently(self):
        two_networks = _result([
            ("net_a", ["U1.1"], [((0, 0), (2, 0))]),
            ("net_b", ["U2.1"], [((2, 0), (4, 0))]),
        ])
        one_network = _result([("net_c", ["U1.1", "U2.1"], [((0, 0), (2, 0)), ((2, 0), (4, 0))])])
        self.assertNotEqual(normalize_submission(two_networks), normalize_submission(one_network))

    def test_repeated_normalization_is_deterministic(self):
        output = _result([("net_1", ["U1.1", "U2.1"], [((0, 0), (2, 0)), ((2, 0), (4, 0))])])
        self.assertEqual(normalize_submission(output), normalize_submission(output))


if __name__ == "__main__":
    unittest.main()
