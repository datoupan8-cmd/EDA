"""Dataset-free acceptance checks for the imported NetLine integration."""
import contextlib
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pcb.core.context import PipelineContext
from pcb.core.interfaces import WireStageOutput
from pcb.core.team_config import load_team_config
from pcb.network_shared.line_runtime import (
    configure_lines, provenance_snapshot, runtime_versions,
)
from pcb.network_shared.selection import selected_versions
from pcb.submission import export, validate_strict
from pcb.wire_stage.v1 import simplify, simplify_with_indices
from tests.topology.split_cases import cases
from tools.run_team_pipeline import main, runtime_registry


VERSIONS = ("legacy_v2", "path_v1", "path_collinear_v1")
TOPOLOGIES = ("v2", "v3", "v3_power_terminals_v1", "v3_power_terminals_v2")


def run_case(topology, version, *, provenance=False, case="elbow", stage=None):
    config = replace(load_team_config("configs/team.json"), topology=topology)
    _, scene, mask = next(row for row in cases() if row[0] == case)
    upstream = asdict(scene)
    stage = stage or runtime_registry(config).create("topology", topology)
    logical_backend = stage.hypergraph
    configure_lines(stage, version, provenance=provenance)
    context = PipelineContext("synthetic.png", 100, 100, config, None)
    output = stage.run(scene, WireStageOutput(mask, debug={"power_symbol_crops": {}}), context)
    return stage, scene, output, upstream, logical_backend


class NetLineIntegration(unittest.TestCase):
    def test_candidates_preserve_membership_components_and_pins_in_all_runtime_topologies(self):
        for topology in TOPOLOGIES:
            for case in ("straight", "elbow", "cross_dot", "direct_contact"):
                _, baseline, _, _, _ = run_case(topology, "legacy_v2", case=case)
                for version in VERSIONS:
                    with self.subTest(topology=topology, case=case, version=version):
                        stage, scene, _, upstream, logical = run_case(topology, version, case=case)
                        self.assertIs(stage.hypergraph, logical)
                        self.assertEqual([net.pins for net in scene.nets], [net.pins for net in baseline.nets])
                        self.assertEqual(asdict(scene)["components"], upstream["components"])
                        self.assertEqual(asdict(scene)["texts"], upstream["texts"])
                        self.assertEqual(stage.network_versions["net_lines"], version)
                        validate_strict(export(scene), (100, 100))

    def test_optional_sidecar_preserves_public_results_and_aligns_every_output_segment(self):
        for topology in TOPOLOGIES:
            for version in VERSIONS:
                with self.subTest(topology=topology, version=version):
                    _, baseline, base_output, _, _ = run_case(topology, version)
                    stage, scene, output, _, _ = run_case(topology, version, provenance=True)
                    self.assertEqual(asdict(scene), asdict(baseline))
                    np.testing.assert_array_equal(output.skeleton, base_output.skeleton)
                    record = json.loads(json.dumps(provenance_snapshot(stage)))
                    self.assertEqual(record["version"], version)
                    self.assertEqual(record["coordinate_space"], "original_image_top_left_xy")
                    for geometry, trace in zip(record["geometry"]["groups"], record["provenance"]["groups"]):
                        self.assertEqual(geometry["group_id"], trace["group_id"])
                        self.assertEqual(len(geometry["segments"]), len(trace["segments"]))
                        for index, segment in enumerate(trace["segments"]):
                            self.assertEqual(segment["segment_index"], index)
                            self.assertEqual(segment["endpoints"], geometry["segments"][index])

    def test_cli_lists_per_run_override_without_rewriting_any_default(self):
        config = load_team_config("configs/team.json")
        defaults = selected_versions()
        for version in (None, *VERSIONS):
            stream = io.StringIO()
            options = [] if version is None else ["--net-lines-version", version]
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(["--list-stages", *options]), 0)
            record = json.loads(stream.getvalue())
            self.assertEqual(record["config"], config.as_dict())
            self.assertEqual(record["network_versions"], dict(defaults, net_lines=version or defaults["net_lines"]))
            self.assertEqual(selected_versions(), defaults)

    def test_invalid_version_fails_before_model_setup(self):
        with patch("tools.run_team_pipeline.build_runtime") as build, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--net-lines-version", "not_registered", "--list-stages"])
        self.assertEqual(error.exception.code, 2)
        build.assert_not_called()

    def test_unrelated_topology_rejects_both_netline_options(self):
        config = replace(load_team_config("configs/team.json"), topology="v1")
        for options in (["--net-lines-version", "path_v1"], ["--save-net-lines-provenance"]):
            with patch("tools.run_team_pipeline.load_team_config", return_value=config), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(options+["--list-stages"])
            self.assertEqual(error.exception.code, 2)
        self.assertIsNone(runtime_versions("v1"))

    def test_recorder_replaces_previous_image_and_refuses_stale_trace_after_failure(self):
        stage, _, _, _, _ = run_case("v3", "legacy_v2", provenance=True, case="elbow")
        previous = copy.deepcopy(provenance_snapshot(stage))
        config = replace(load_team_config("configs/team.json"), topology="v3")
        _, scene, mask = next(row for row in cases() if row[0] == "straight")
        context = PipelineContext("second.png", 100, 100, config, None)
        stage.run(scene, WireStageOutput(mask), context)
        self.assertNotEqual(provenance_snapshot(stage)["geometry"], previous["geometry"])
        with patch.object(stage.lines.backend, "run_with_provenance", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                stage.run(scene, WireStageOutput(mask), context)
        with self.assertRaisesRegex(ValueError, "No NetLine provenance"):
            provenance_snapshot(stage)

    def test_cli_writes_optional_sidecar_with_real_topology_and_preserves_official_json(self):
        # Image and all scene data are generated here; only upstream model setup
        # is replaced. The actual CLI I/O, topology, validators and export run.
        def build(config, device, ocr_device, cache, **options):
            stage = runtime_registry(config).create("topology", config.topology)
            configure_lines(stage, options.get("net_lines_version"),
                            provenance=options.get("save_net_lines_provenance", False))

            def run(image, context):
                _, scene, mask = next(row for row in cases() if row[0] == "elbow")
                wire = WireStageOutput(mask, debug={"power_symbol_crops": {}})
                stage.run(scene, wire, context)
                return SimpleNamespace(result=export(scene), scene=scene, wire=wire)
            return SimpleNamespace(topology=stage, run=run), None, None, "cpu"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/"synthetic.png"
            cv2.imwrite(str(image), np.full((100, 100, 3), 255, np.uint8))
            outputs = []
            for trace in (False, True):
                folder = root/("with_trace" if trace else "plain")
                options = ["--image", str(image), "--output", str(folder), "--cache-dir", str(root/"cache"),
                           "--device", "cpu", "--net-lines-version", "path_collinear_v1"]
                if trace:
                    options.append("--save-net-lines-provenance")
                with patch("tools.run_team_pipeline.build_runtime", side_effect=build), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(options), 0)
                result = json.loads((folder/"synthetic/result.json").read_text(encoding="utf8"))
                report = json.loads((folder/"report.json").read_text(encoding="utf8"))
                self.assertEqual(set(result), {"components", "pins", "nets"})
                self.assertEqual(report["network_versions"]["net_lines"], "path_collinear_v1")
                self.assertEqual(report["net_lines_provenance"], trace)
                self.assertEqual(report["target_opens"], 0)
                sidecar = folder/"synthetic/net_lines_provenance.json"
                self.assertEqual(sidecar.is_file(), trace)
                if trace:
                    self.assertEqual(json.loads(sidecar.read_text(encoding="utf8"))["version"], "path_collinear_v1")
                outputs.append(result)
            self.assertEqual(*outputs)

    def test_indexed_simplification_keeps_legacy_geometry_across_loops_and_random_paths(self):
        rng = random.Random(20261027)
        paths = [((0, 0), (0, 5), (5, 5), (0, 0)), ((2, 2), (2, 2))]
        for _ in range(200):
            path = [(20, 20)]
            for _ in range(rng.randrange(1, 100)):
                dy, dx = rng.choice(((0, 0), (0, 1), (1, 0), (1, 1), (-1, -1), (1, 2)))
                path.append((path[-1][0]+dy, path[-1][1]+dx))
            paths.append(path)
        for path in paths:
            for epsilon in (0., .5, 1., 2.):
                indexed = simplify_with_indices(path, epsilon)
                self.assertEqual(indexed.segments, tuple(simplify(path, epsilon)))
                self.assertEqual(len(indexed.segments), len(indexed.segment_edge_ranges))

    def test_float32_rounded_vertices_have_unknown_source_instead_of_guessed_indices(self):
        result = simplify_with_indices(((0, 16777217), (0, 16777220)))
        self.assertIsNone(result.source_vertex_indices)
        self.assertEqual(result.segment_edge_ranges, (None,))


if __name__ == "__main__":
    unittest.main()
