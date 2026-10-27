import unittest

import cv2
import numpy as np

from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import TopologyStageOutput, WireStageOutput
from pcb.network_shared.selection import selected_versions
from pcb.schema import Component, Pin, Scene
from pcb.submission import export
from pcb.topology_stage.versions import TopologyStageV3


def run_topology_version(version):
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(mask, (10, 20), (40, 20), 255, 1)
    cv2.line(mask, (60, 60), (80, 60), 255, 1)
    cv2.line(mask, (80, 60), (80, 80), 255, 1)
    tips = ((10, 20), (10, 20), (40, 20), (60, 60), (80, 80))
    components = [
        Component(f"U{index}", "box", (0, 0, 5, 5), pins=[Pin("1", "1", tip)])
        for index, tip in enumerate(tips, start=1)
    ]
    scene = Scene(100, 100, components, diagnostics={"wire_stroke_width": 1.0})
    versions = {"net_hypergraph": "legacy_v2", "net_lines": version}
    stage = TopologyStageV3(network_versions=versions)
    context = PipelineContext("synthetic.png", 100, 100, PipelineConfig(), None)
    output = stage.run(scene, WireStageOutput(mask), context)
    return stage, scene, output


class PathLinesPipelineTests(unittest.TestCase):
    def test_path_v1_runs_through_topology_stage_and_geometry_validation(self):
        stage, scene, output = run_topology_version("path_v1")
        versions = {"net_hypergraph": "legacy_v2", "net_lines": "path_v1"}

        self.assertIsInstance(output, TopologyStageOutput)
        self.assertEqual(stage.network_versions, versions)
        self.assertEqual(selected_versions()["net_lines"], "legacy_v2")
        self.assertEqual(len(scene.nets), 2)

        contact_net = next(net for net in scene.nets if "U1.1" in net.pins)
        corner_net = next(net for net in scene.nets if "U4.1" in net.pins)
        self.assertCountEqual(contact_net.pins, ("U1.1", "U2.1", "U3.1"))
        self.assertCountEqual(corner_net.pins, ("U4.1", "U5.1"))
        self.assertIn(((10.0, 20.0), (10.0, 20.0)), contact_net.segments)
        self.assertTrue(any(a[1] == b[1] and a[0] != b[0]
                            for a, b in corner_net.segments))
        self.assertTrue(any(a[0] == b[0] and a[1] != b[1]
                            for a, b in corner_net.segments))
        self.assertEqual(
            scene.diagnostics["net_edges"],
            sum(len(net.segments) for net in scene.nets),
        )

    def test_path_collinear_v1_runs_through_topology_and_public_export(self):
        stage, scene, output = run_topology_version("path_collinear_v1")
        candidate_nets = export(scene)
        self.assertIsInstance(output, TopologyStageOutput)
        self.assertEqual(stage.network_versions["net_lines"], "path_collinear_v1")
        self.assertEqual(selected_versions()["net_lines"], "legacy_v2")
        self.assertEqual(set(candidate_nets), {"components", "pins", "nets"})
        self.assertEqual(len(candidate_nets["nets"]), 2)
        self.assertEqual(len(scene.nets), 2)
        self.assertCountEqual(
            [net["hyperGraph"] for net in candidate_nets["nets"].values()],
            ["(U1.1,U2.1,U3.1)", "(U4.1,U5.1)"],
        )
        self.assertTrue(
            all(set(net) == {"hyperGraph", "edges"}
                for net in candidate_nets["nets"].values())
        )
        self.assertEqual(
            scene.diagnostics["net_edges"],
            sum(len(net.segments) for net in scene.nets),
        )


if __name__ == "__main__":
    unittest.main()
