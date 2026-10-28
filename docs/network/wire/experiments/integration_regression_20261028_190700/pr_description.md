## Summary
Integrates the pushed Wire v4–v9 experimental backends with NetLine `legacy_v2`, `path_v1`, and `path_collinear_v1`, plus the optional `reference_segment_edge_ranges` provenance sidecar. The shared source-range change is isolated to `WireGraph`, Graph construction, and Wire v1 indexed simplification. Existing APIs remain compatible.

## Validation
- Wire tests: 34/34 passed.
- NetLine tests: 62/62 passed, including reference-edge provenance.
- `tests.topology.test_network_split.NetworkSplit`: 12/12 passed.
- Authorized 0001–0013 cached integration replay: all 13 samples passed for default `v3_frame_guard + legacy_v2`, `v9_branch_salvage_recall + legacy_v2`, and `v9_branch_salvage_recall + path_collinear_v1`.
- `git diff --check`: passed.
- No Component/Pin model inference was rerun.

The detailed results and reproducible scripts are in `docs/network/wire/experiments/integration_regression_20261028_190700/`.

## Results and limitations
- Wire v9 geometry coverage is 50.86% at Wire mask, 48.29% at Graph paths and GeometryResult, and 42.40% at Scene.nets. The collinear backend gives 42.41% at Scene.nets. These are development-set diagnostics, not official competition scores.
- Exact-ID connectivity is unchanged across all three scenarios: TP=2, FP=54, FN=324; 370 matched unambiguous Pins. This does not establish improved electrical correctness.
- Experimental Wire backends remain non-default. Defaults remain Wire `v3_frame_guard` and NetLine `legacy_v2`.
- The existing assembler filters groups without Pins. The expected export policy for such geometry remains for Integration/Submission and Topology owners to decide.
- Graph path coverage loss, Pin identity/matching limits, and Topology membership behavior remain open owner questions.
- This PR is an integration checkpoint for review, not final algorithm acceptance or a claim of improved official score.
