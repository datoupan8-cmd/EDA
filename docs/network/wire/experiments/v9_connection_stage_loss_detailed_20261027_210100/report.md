# V9 connectivity and downstream stage-loss replay

Reused the authorized 0001–0013 images and saved Component/Pin predictions. Wire, Graph, Topology, NetLine and assembly ran in memory; Component/Pin inference was not rerun. GT was read only after predictions for scoring.

Active network versions: `{'net_hypergraph': 'legacy_v2', 'net_lines': 'legacy_v2'}`. The NetLine backend was `legacy_v2`.

## Exact-ID connectivity comparison

Only GT pin identifiers that exactly match a predicted `component.pin` and are not ambiguous across GT nets enter pair scoring. Unmatched predicted/GT pins and ambiguous GT IDs are listed separately and are not counted as correct. Precision/recall are pairwise over that eligible universe.

| Wire | TP correct pairs | FN required pairs missed | FP incorrect pairs | Recall | Precision | FP / predicted | matched unambiguous pins |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3_frame_guard | 2 | 324 | 54 | 0.61% | 3.57% | 96.43% | 370 |
| v8_baseline_anchored_recall | 2 | 324 | 54 | 0.61% | 3.57% | 96.43% | 370 |
| v9_branch_salvage_recall | 2 | 324 | 54 | 0.61% | 3.57% | 96.43% | 370 |

V9 vs V8: 0 newly correct pairs, 0 newly incorrect pairs, 0 previously correct pairs lost.

## Geometry by pipeline stage

### v3_frame_guard

| Stage | mean coverage @5px | edges >=80% |
|---|---:|---:|
| wire_mask | 11.46% | 16/744 |
| graph_paths | 8.41% | 9/744 |
| graph_reference_segments | 8.41% | 9/744 |
| plan_selected_paths | 8.41% | 9/744 |
| plan_selected_reference_segments | 8.41% | 9/744 |
| geometry_result | 8.41% | 9/744 |
| assembled_scene_nets | 7.59% | 8/744 |

### v8_baseline_anchored_recall

| Stage | mean coverage @5px | edges >=80% |
|---|---:|---:|
| wire_mask | 44.06% | 273/744 |
| graph_paths | 41.35% | 266/744 |
| graph_reference_segments | 41.35% | 266/744 |
| plan_selected_paths | 41.35% | 266/744 |
| plan_selected_reference_segments | 41.35% | 266/744 |
| geometry_result | 41.35% | 266/744 |
| assembled_scene_nets | 36.53% | 234/744 |

### v9_branch_salvage_recall

| Stage | mean coverage @5px | edges >=80% |
|---|---:|---:|
| wire_mask | 50.86% | 318/744 |
| graph_paths | 48.29% | 311/744 |
| graph_reference_segments | 48.29% | 311/744 |
| plan_selected_paths | 48.29% | 311/744 |
| plan_selected_reference_segments | 48.29% | 311/744 |
| geometry_result | 48.29% | 311/744 |
| assembled_scene_nets | 42.40% | 271/744 |

First loss transitions for edges with Wire mask coverage >=80%:

**v3_frame_guard:** graph_paths=7, assembled_scene_nets=1
**v8_baseline_anchored_recall:** assembled_scene_nets=32, graph_paths=7
**v9_branch_salvage_recall:** assembled_scene_nets=40, graph_paths=7

## Evidence boundary

This checkout’s `WireGraph` has `paths` and `reference_segments` but no `reference_segment_edge_ranges`; exact source-edge provenance is unavailable. The saved result records graph path IDs near endpoints only as spatial hints. They are not asserted to be exact source mappings. `reference_segments` are generated from paths by `cv2.approxPolyDP` with epsilon 1 px. Stage coverage can locate geometric loss but not identify a particular source edge with certainty.

Connectivity results are limited to exactly matched, unambiguous pin IDs. They do not validate unmatched pins, labels without parseable pin identities, or ambiguous GT memberships. Geometric overlap does not establish net identity. No official score is claimed.

Default Wire remains `v3_frame_guard`; no other-owner code or config was modified. No full Component/Pin inference was run.

Output files: `results.json` (per-sample stage coverage, selected path IDs, connectivity TP/FN/FP and unmatched/ambiguous pin lists); this report.
