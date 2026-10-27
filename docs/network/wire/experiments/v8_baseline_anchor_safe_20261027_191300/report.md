# Wire v8 baseline-anchored recall replay

Cached Component/Pin outputs were reused for authorized development samples 0001–0013. No Component/Pin inference was rerun. GT was read only for offline geometry and pin-pair diagnostics; it was not passed to Wire or Topology.

## Aggregate comparison

| Version | GT edge coverage @5px | GT edges >=80% | Mask pixels | Mask precision proxy | New cross-GT pairs | Known correct v3 pairs split | NetLine geometry coverage | NetLine segments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_frame_guard | 11.46% | 16/744 | 100564 | 13.14% | 0 | 0 | 7.59% | 1946 |
| v7_palette_corridor_recall | 30.99% | 167/744 | 138106 | 38.05% | 0 | 0 | 27.51% | 2127 |
| v8_baseline_anchored_recall | 44.06% | 273/744 | 160684 | 46.89% | 0 | 0 | 36.53% | 2168 |

## Per-sample comparison

| Sample | Edges | v3 coverage | v7 coverage | v8 coverage | v7 >=80% | v8 >=80% | v7 mask px | v8 mask px | v8 cross-guard rejected px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0001 | 116 | 23.78% | 30.69% | 40.15% | 19 | 31 | 20785 | 22715 | 20534 |
| 0002 | 40 | 11.25% | 32.36% | 32.36% | 11 | 11 | 6707 | 6761 | 3033 |
| 0003 | 46 | 10.35% | 29.11% | 41.20% | 10 | 16 | 8387 | 9021 | 1668 |
| 0004 | 59 | 8.85% | 30.77% | 47.46% | 14 | 25 | 7736 | 9423 | 2991 |
| 0005 | 57 | 5.51% | 30.81% | 55.74% | 14 | 29 | 9092 | 11583 | 824 |
| 0006 | 86 | 5.95% | 29.01% | 40.97% | 21 | 32 | 12321 | 15036 | 4151 |
| 0007 | 68 | 4.65% | 42.84% | 53.55% | 27 | 35 | 11119 | 13512 | 2037 |
| 0008 | 50 | 12.27% | 30.63% | 50.91% | 9 | 20 | 7355 | 8697 | 2668 |
| 0009 | 39 | 7.71% | 38.92% | 58.85% | 14 | 22 | 4918 | 8059 | 284 |
| 0010 | 82 | 7.37% | 31.60% | 47.27% | 20 | 35 | 12543 | 15366 | 3818 |
| 0011 | 24 | 10.32% | 22.95% | 39.64% | 4 | 8 | 5984 | 8059 | 0 |
| 0012 | 28 | 12.69% | 16.93% | 28.18% | 1 | 5 | 13662 | 14350 | 12089 |
| 0013 | 49 | 21.48% | 24.82% | 27.21% | 3 | 4 | 17497 | 18102 | 22367 |

## Interpretation

Prior production-stage trace on the same 13 samples measured 744/744 GT edges with >=80% proximity coverage in the raw palette union, 721/744 after line support, 307/744 after v7 candidate anchoring but before its bridge guard, and 167/744 after the guard. This locates the major remaining loss in candidate anchoring and whole-component bridge rejection, not primarily in raw color generation.

These are geometry and connectivity diagnostics only. A higher coverage or fewer rejected pixels does not establish electrical correctness or official competition accuracy. The exact-pin-id limitation leaves unmatched-pin risks unmeasured.

Default configuration remains `v3_frame_guard`. No production default or shared module was changed.
