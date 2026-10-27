# Wire v9 branch-salvage recall replay

Cached Component/Pin outputs were reused for authorized development samples 0001–0013. Component/Pin inference was not rerun. The same Wire and Topology inputs were used for v3, v7, v8, and v9; GT was used only after inference for offline diagnostics.

## Aggregate comparison

| Version | GT edge coverage @5px | GT edges >=80% | Mask pixels | Mask precision proxy | New pin pairs vs v3 | New cross-GT pairs | Known correct v3 pairs split | NetLine geometry coverage | NetLine edges >=80% | NetLine segments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_frame_guard | 11.46% | 16/744 | 100564 | 13.14% | 0 | 0 | 0 | 7.59% | 8/744 | 1946 |
| v7_palette_corridor_recall | 30.99% | 167/744 | 138106 | 38.05% | 0 | 0 | 0 | 27.51% | 160/744 | 2127 |
| v8_baseline_anchored_recall | 44.06% | 273/744 | 160684 | 46.89% | 0 | 0 | 0 | 36.53% | 234/744 | 2168 |
| v9_branch_salvage_recall | 50.86% | 318/744 | 184262 | 52.80% | 0 | 0 | 0 | 42.40% | 271/744 | 2216 |

## Per-sample comparison

| Sample | GT edges | v7 coverage | v8 coverage | v9 coverage | v7/v8/v9 edges >=80% | v8/v9 mask px | v9 salvaged px | v8/v9 NetLine coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0001 | 116 | 30.69% | 40.15% | 49.33% | 19/31/40 | 22715/28278 | 5563 | 29.41%/38.65% |
| 0002 | 40 | 32.36% | 32.36% | 41.19% | 11/11/11 | 6761/7431 | 670 | 30.36%/39.19% |
| 0003 | 46 | 29.11% | 41.20% | 43.09% | 10/16/17 | 9021/9228 | 207 | 32.66%/34.62% |
| 0004 | 59 | 30.77% | 47.46% | 52.25% | 14/25/27 | 9423/10087 | 664 | 39.03%/43.57% |
| 0005 | 57 | 30.81% | 55.74% | 55.74% | 14/29/29 | 11583/11583 | 0 | 37.17%/37.17% |
| 0006 | 86 | 29.01% | 40.97% | 46.45% | 21/32/37 | 15036/16746 | 1710 | 39.21%/44.68% |
| 0007 | 68 | 42.84% | 53.55% | 54.80% | 27/35/36 | 13512/13730 | 218 | 52.03%/53.28% |
| 0008 | 50 | 30.63% | 50.91% | 56.22% | 9/20/23 | 8697/9592 | 895 | 47.40%/52.85% |
| 0009 | 39 | 38.92% | 58.85% | 58.85% | 14/22/22 | 8059/8059 | 0 | 52.82%/52.82% |
| 0010 | 82 | 31.60% | 47.27% | 52.39% | 20/35/39 | 15366/16284 | 918 | 45.86%/51.00% |
| 0011 | 24 | 22.95% | 39.64% | 39.64% | 4/8/8 | 8059/8059 | 0 | 35.02%/35.02% |
| 0012 | 28 | 16.93% | 28.18% | 69.94% | 1/5/16 | 14350/21094 | 6744 | 9.33%/33.56% |
| 0013 | 49 | 24.82% | 27.21% | 44.70% | 3/4/13 | 18102/24091 | 5989 | 8.73%/22.16% |

## Rejected component evidence and v9 policy

On the cached V8 replay, 155 candidate components (76,464 pixels) were rejected for touching multiple V7 mask components. The structural descriptors are approximate raster skeleton categories, not electrical classifications: 114 straight chains, 16 bent chains, 7 T/Y-like, 5 cross/multiarm-like, and 13 multi-branch/blob components. 144 rejected components touched exactly two V7 regions; 11 touched three or more. 141 blocks individually lie within 5 px of at least one GT edge for >=80% of that edge. Spatial coincidence is not proof of net identity.

Samples 0012 and 0013 include the most ambiguous regions: sample 0012 component label 3 (bbox [364,300,1398,567]) touched 10 baseline regions and was a multi-branch/blob; sample 0013 label 11 (bbox [535,237,906,573]) touched 8 regions and included several GT-net edge geometries. Representative overlays are in `../v8_rejected_component_analysis_20261027_202537/visuals/`.

V9 removes a small radius around skeleton branch clusters and considers only original palette/line-supported fragments that touch exactly one V7 region. Each fragment is tested incrementally against the current mask; a fragment that would merge separate mask components is rejected. V9 starts from the V8 result and only adds accepted pixels, so it does not delete V8 pixels. Straight and bent chain bridges without a branch point remain rejected because image geometry alone cannot establish whether their endpoints belong to the same electrical net. No GT-derived identifier or coordinate is used by the production rule.

## Results and limits

V9 improves over V8 on 10 samples and ties on the others; it has no per-sample coverage regression. Aggregate GT-edge coverage rises from 44.06% to 50.86%, edges >=80% from 273 to 318, mask precision proxy from 46.89% to 52.80%, and assembled NetLine coverage from 36.53% to 42.40%. The V9 incremental guard rejected 0 salvage pieces in this replay; newly measured pin pairs, cross-GT-net pairs, and splits of known-correct V3 pairs are all zero. These checks only cover pins matched by exact identifiers; unmatched-pin risk remains.

All geometry and precision values are local diagnostic proxies, not official competition metrics. The 13 samples are a development set with no independent holdout, and a zero measured connection delta does not prove electrical safety. The default remains `v3_frame_guard`; V9 is an experimental non-default candidate.

## Reproduction

Command: `D:\EDA\EDA\.venv\Scripts\python.exe -B docs/network/wire/experiments/v9_branch_salvage_safe_20261027_204200/compare_v9_branch_salvage.py`

Wire unit tests: `python -B -m unittest discover -s tests/wire -p "test_*.py" -v` (see task execution record).

Verification: python -B -m unittest discover -s tests/wire -p "test_*.py" -v passed 34/34 (exit 0); git diff --check passed. Default configuration remains 3_frame_guard.
