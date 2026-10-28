# Wire v9 + NetLine integration regression

Integration HEAD: `dd48dbe48c672ae95c6cbab2581c7b321012717f`

This is a local development-set diagnostic, not an official competition score. Component/Pin predictions were reused from saved artifacts; no Component/Pin inference was run.

## Aggregate geometry (5 px sampled edge coverage)

| Scenario | Wire mask | Graph paths | GeometryResult | Scene.nets | >=80% edges at Wire / Graph / Geometry / Scene | Net count (sum) | Geometry segments (sum) | Scene segments (sum) |
|---|---:|---:|---:|---:|---:|---:|---:|
| default_v3_legacy | 11.46% | 8.41% | 8.41% | 7.59% | 16 / 9 / 9 / 8 | 517 | 2330 | 1946 |
| v9_legacy | 50.86% | 48.29% | 48.29% | 42.40% | 318 / 311 / 311 / 271 | 517 | 2622 | 2216 |
| v9_collinear | 50.86% | 48.29% | 48.29% | 42.41% | 318 / 311 / 311 / 271 | 517 | 5849 | 4936 |

## Connectivity exact-ID diagnostic

| Scenario | TP | FP | FN | Precision | Recall | matched Pins (sum) | unmatched GT Pins (sum) | unmatched predicted Pins (sum) | ambiguous GT Pins (sum) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_v3_legacy | 2 | 54 | 324 | 3.57% | 0.61% | 370 | 299 | 231 | 0 |
| v9_legacy | 2 | 54 | 324 | 3.57% | 0.61% | 370 | 299 | 231 | 0 |
| v9_collinear | 2 | 54 | 324 | 3.57% | 0.61% | 370 | 299 | 231 | 0 |

Historical exact-ID pair baseline: TP=2, FP=54, FN=324, precision=3.57%, recall=0.61%, matched unambiguous pins=370. It is comparable only if current counts and protocol agree.

## Per-sample status and stage coverage

| Sample | Status | Scenario | GT edges | Wire mask | Graph paths | GeometryResult | Scene.nets | no-Pin groups | filtered geometry segments | complete/unknown provenance refs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0001 | passed | default_v3_legacy | 116 | 23.78% | 18.19% | 18.14% | 16.07% | 35 | 126 | 480/0 |
| 0001 | passed | v9_legacy | 116 | 49.33% | 44.47% | 44.41% | 38.65% | 35 | 131 | 502/0 |
| 0001 | passed | v9_collinear | 116 | 49.33% | 44.47% | 44.47% | 38.70% | 35 | 266 | 1100/0 |
| 0002 | passed | default_v3_legacy | 40 | 11.25% | 9.33% | 9.33% | 8.83% | 10 | 21 | 169/0 |
| 0002 | passed | v9_legacy | 40 | 41.19% | 39.52% | 39.52% | 39.19% | 9 | 18 | 179/0 |
| 0002 | passed | v9_collinear | 40 | 41.19% | 39.52% | 39.52% | 39.19% | 9 | 32 | 347/0 |
| 0003 | passed | default_v3_legacy | 46 | 10.35% | 8.67% | 8.67% | 8.30% | 7 | 13 | 120/0 |
| 0003 | passed | v9_legacy | 46 | 43.09% | 41.77% | 41.77% | 34.62% | 9 | 19 | 145/0 |
| 0003 | passed | v9_collinear | 46 | 43.09% | 41.77% | 41.77% | 34.62% | 9 | 30 | 311/0 |
| 0004 | passed | default_v3_legacy | 59 | 8.85% | 7.31% | 7.31% | 6.14% | 14 | 23 | 117/0 |
| 0004 | passed | v9_legacy | 59 | 52.25% | 49.96% | 49.96% | 43.57% | 15 | 24 | 136/0 |
| 0004 | passed | v9_collinear | 59 | 52.25% | 49.96% | 49.96% | 43.57% | 15 | 43 | 291/0 |
| 0005 | passed | default_v3_legacy | 57 | 5.51% | 4.34% | 4.34% | 3.54% | 17 | 22 | 150/0 |
| 0005 | passed | v9_legacy | 57 | 55.74% | 54.60% | 54.60% | 37.17% | 17 | 26 | 170/0 |
| 0005 | passed | v9_collinear | 57 | 55.74% | 54.60% | 54.60% | 37.17% | 17 | 48 | 327/0 |
| 0006 | passed | default_v3_legacy | 86 | 5.95% | 5.11% | 5.11% | 5.01% | 10 | 11 | 160/0 |
| 0006 | passed | v9_legacy | 86 | 46.45% | 45.81% | 45.81% | 44.68% | 10 | 11 | 176/0 |
| 0006 | passed | v9_collinear | 86 | 46.45% | 45.81% | 45.81% | 44.68% | 10 | 19 | 390/0 |
| 0007 | passed | default_v3_legacy | 68 | 4.65% | 3.79% | 3.79% | 3.79% | 6 | 7 | 153/0 |
| 0007 | passed | v9_legacy | 68 | 54.80% | 53.28% | 53.28% | 53.28% | 6 | 6 | 179/0 |
| 0007 | passed | v9_collinear | 68 | 54.80% | 53.28% | 53.28% | 53.28% | 6 | 7 | 410/0 |
| 0008 | passed | default_v3_legacy | 50 | 12.27% | 10.01% | 10.01% | 9.66% | 3 | 10 | 174/0 |
| 0008 | passed | v9_legacy | 50 | 56.22% | 54.67% | 54.67% | 52.85% | 3 | 10 | 189/0 |
| 0008 | passed | v9_collinear | 50 | 56.22% | 54.67% | 54.67% | 52.85% | 3 | 16 | 380/0 |
| 0009 | passed | default_v3_legacy | 39 | 7.71% | 6.78% | 6.78% | 6.71% | 7 | 12 | 88/0 |
| 0009 | passed | v9_legacy | 39 | 58.85% | 57.82% | 57.82% | 52.82% | 7 | 11 | 98/0 |
| 0009 | passed | v9_collinear | 39 | 58.85% | 57.82% | 57.82% | 52.82% | 7 | 15 | 209/0 |
| 0010 | passed | default_v3_legacy | 82 | 7.37% | 6.33% | 6.31% | 6.23% | 8 | 12 | 184/0 |
| 0010 | passed | v9_legacy | 82 | 52.39% | 51.09% | 51.09% | 51.00% | 6 | 9 | 210/0 |
| 0010 | passed | v9_collinear | 82 | 52.39% | 51.09% | 51.09% | 51.00% | 6 | 10 | 510/0 |
| 0011 | passed | default_v3_legacy | 24 | 10.32% | 7.64% | 7.83% | 7.12% | 11 | 11 | 150/0 |
| 0011 | passed | v9_legacy | 24 | 39.64% | 37.76% | 37.96% | 35.02% | 11 | 11 | 160/0 |
| 0011 | passed | v9_collinear | 24 | 39.64% | 37.76% | 37.76% | 34.82% | 11 | 13 | 335/0 |
| 0012 | passed | default_v3_legacy | 28 | 12.69% | 6.55% | 6.52% | 5.11% | 14 | 57 | 191/0 |
| 0012 | passed | v9_legacy | 28 | 69.94% | 68.28% | 68.22% | 33.56% | 16 | 64 | 243/0 |
| 0012 | passed | v9_collinear | 28 | 69.94% | 68.28% | 68.28% | 33.56% | 16 | 249 | 665/0 |
| 0013 | passed | default_v3_legacy | 49 | 21.48% | 7.11% | 7.15% | 4.56% | 40 | 59 | 194/0 |
| 0013 | passed | v9_legacy | 49 | 44.70% | 33.69% | 33.76% | 22.16% | 44 | 66 | 235/0 |
| 0013 | passed | v9_collinear | 49 | 44.70% | 33.69% | 33.69% | 22.16% | 44 | 165 | 574/0 |


## Aggregate object and provenance counts

| Scenario | Net count sum | no-Pin groups | GeometryResult segments | segments filtered by no-Pin rule | Scene.nets segments | provenance complete / unknown | source ranges valid / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| default_v3_legacy | 517 | 182 | 2330 | 384 | 1946 | 2330 / 0 | 2323 / 2323 |
| v9_legacy | 517 | 188 | 2622 | 406 | 2216 | 2622 / 0 | 2615 / 2615 |
| v9_collinear | 517 | 188 | 5849 | 913 | 4936 | 5849 / 0 | 2615 / 2615 |

All 39 sample-scenario executions completed. Pair-set comparison against the default is 0 added and 0 lost TP, FP, and FN pairs for both experimental scenarios. The five-pixel geometry values are local development diagnostics, not official scores.

## Known sample 0001 cases

Detailed stage coverage and Plan group records are in `sample_0001_cases.json`. For v9 + legacy, `net_4/edge_4` is 100% in Wire, Graph, and GeometryResult but 0% in Scene.nets: spatially overlapping group 58 has no Pins, path [58], and 2 geometry segments, so the current assembler filters it. For v9 + path_collinear_v1 the same group has 3 segments and remains filtered. For v3 the same edge is only 4% in Wire/Graph/GeometryResult and 0% in Scene.nets (group 126 has no Pins, path [126]). `net_12/edge_2` stays 100% in Wire and 60% from Graph through Scene.nets in both v3 and v9. In v9, spatially overlapping Plan groups are 184 (Pin U3.UNK1; paths [179,184,186,187,188]) and 196 (Pin U3.UNK2; paths [195,196,197]). These are exact Plan/provenance records, but their spatial overlap with the GT edge does not prove electrical membership or exact source-edge-to-GT correspondence. No Pin-less group export policy was changed.

## Limitations and owner handoff

- Wire/Graph geometry proximity is not proof of electrical connectivity.
- The exact-ID connectivity metric excludes unmatched and ambiguous Pins; it does not assess them as correct.
- `reference_segment_edge_ranges` maps simplified reference segments to raw path-edge intervals; it does not establish Pin-to-edge attachment.
- Existing `assemble()` omits GeometryResult groups with no Pins. Whether those geometries should be exported requires Integration/Submission and Topology owner decision.
- GT geometry present in Wire but absent in Graph paths should be investigated by Graph owner; Pin identity and plan membership issues belong to Pin/Topology/NetHypergraph owners.
- Experimental Wire versions remain non-default; this report is not algorithm acceptance or an official score.

Failures: 0. The first execution reached all stage runs but exposed a report aggregation KeyError (`per_edge` not retained); the aggregator was corrected, and the complete cache replay was rerun successfully. See `results.json` for per-sample records.
