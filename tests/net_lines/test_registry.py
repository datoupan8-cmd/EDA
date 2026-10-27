import unittest
from types import MappingProxyType
from unittest.mock import patch

from pcb.core.registry import StageRegistry
from pcb.net_hypergraph import registry as hypergraph_registry
from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.net_lines.path_collinear_v1 import CollinearPathLines
from pcb.net_lines.path_v1 import PathLines
from pcb.net_lines.registry import register_stages
from pcb.network_shared.contracts import GeometryResult, GroupLines, NetGroup, NetworkPlan, WireGraph
from pcb.network_shared.selection import backends, selected_versions


def graph_and_plan():
    graph = WireGraph(
        width=10,
        height=10,
        paths=(((2, 3), (4, 5)),),
        adjacency=MappingProxyType({}),
        reference_segments=((),),
        mask=memoryview(b"mask").toreadonly(),
        thickness=memoryview(b"thickness").toreadonly(),
        skeleton=memoryview(b"skeleton").toreadonly(),
    )
    return graph, NetworkPlan((NetGroup(42, (), (0,)),))


class NetLineRegistryTests(unittest.TestCase):
    def test_path_version_is_registered_created_and_callable(self):
        registry = StageRegistry()
        register_stages(registry)

        self.assertEqual(
            registry.versions("net_lines"),
            ("legacy_v2", "path_collinear_v1", "path_v1"),
        )
        self.assertIsInstance(registry.create("net_lines", "legacy_v2"), LegacyLines)
        lines = registry.create("net_lines", "path_v1")
        self.assertIsInstance(lines, PathLines)

        graph, plan = graph_and_plan()
        self.assertEqual(
            lines.run(graph, plan),
            GeometryResult((GroupLines(42, (((3.0, 2.0), (5.0, 4.0)),)),)),
        )

    def test_backend_selector_can_select_path_version_without_changing_default(self):
        self.assertEqual(selected_versions()["net_lines"], "legacy_v2")

        def register_test_hypergraph(registry):
            registry.register("net_hypergraph", "test_stub", object)

        with patch.object(hypergraph_registry, "register_stages", register_test_hypergraph):
            _, lines = backends(
                {"net_hypergraph": "test_stub", "net_lines": "path_v1"}
            )

        self.assertIsInstance(lines, PathLines)
        graph, plan = graph_and_plan()
        self.assertEqual(
            lines.run(graph, plan),
            GeometryResult((GroupLines(42, (((3.0, 2.0), (5.0, 4.0)),)),)),
        )

    def test_registered_path_version_is_repeatable(self):
        registry = StageRegistry()
        register_stages(registry)
        lines = registry.create("net_lines", "path_v1")
        graph, plan = graph_and_plan()

        first = lines.run(graph, plan)
        second = lines.run(graph, plan)

        self.assertEqual(second, first)

    def test_collinear_candidate_is_registered_created_and_callable(self):
        registry = StageRegistry()
        register_stages(registry)
        backend = registry.create("net_lines", "path_collinear_v1")
        self.assertIsInstance(backend, CollinearPathLines)

        graph, plan = graph_and_plan()
        self.assertEqual(
            backend.run(graph, plan),
            GeometryResult((GroupLines(42, (((3.0, 2.0), (5.0, 4.0)),)),)),
        )

    def test_explicit_selection_preserves_default_and_existing_backend_outputs(self):
        self.assertEqual(selected_versions()["net_lines"], "legacy_v2")
        graph, plan = graph_and_plan()

        def register_test_hypergraph(registry):
            registry.register("net_hypergraph", "test_stub", object)

        results = {}
        with patch.object(hypergraph_registry, "register_stages", register_test_hypergraph):
            for version, expected_type in (
                ("legacy_v2", LegacyLines),
                ("path_v1", PathLines),
                ("path_collinear_v1", CollinearPathLines),
            ):
                _, backend = backends(
                    {"net_hypergraph": "test_stub", "net_lines": version}
                )
                self.assertIsInstance(backend, expected_type)
                results[version] = backend.run(graph, plan)

        self.assertEqual(results["legacy_v2"], LegacyLines().run(graph, plan))
        self.assertEqual(results["path_v1"], PathLines().run(graph, plan))
        self.assertEqual(
            results["path_collinear_v1"], CollinearPathLines().run(graph, plan)
        )
        self.assertEqual(selected_versions()["net_lines"], "legacy_v2")

    def test_unknown_version_has_clear_available_versions_error(self):
        def register_test_hypergraph(registry):
            registry.register("net_hypergraph", "test_stub", object)

        with patch.object(hypergraph_registry, "register_stages", register_test_hypergraph):
            with self.assertRaisesRegex(
                ValueError,
                r"Unknown net_lines Stage 'missing'; available=\['legacy_v2', 'path_collinear_v1', 'path_v1'\]",
            ):
                backends({"net_hypergraph": "test_stub", "net_lines": "missing"})

    def test_candidate_registration_does_not_change_other_backend_results(self):
        graph, plan = graph_and_plan()
        legacy_before = LegacyLines().run(graph, plan)
        path_before = PathLines().run(graph, plan)

        registry = StageRegistry()
        register_stages(registry)
        registry.create("net_lines", "path_collinear_v1").run(graph, plan)

        self.assertEqual(LegacyLines().run(graph, plan), legacy_before)
        self.assertEqual(PathLines().run(graph, plan), path_before)

    def test_collinear_candidate_is_repeatable(self):
        registry = StageRegistry()
        register_stages(registry)
        backend = registry.create("net_lines", "path_collinear_v1")
        graph, plan = graph_and_plan()

        first = backend.run(graph, plan)
        second = backend.run(graph, plan)

        self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()
