# Wire v7 corridor-anchored palette recall

Replayed the 13 authorized development images using saved Component/Pin outputs; no model inference was rerun. GT was used only for offline analysis.

| Version | GT-edge macro coverage @5px | GT edges >=80% | mask pixels | precision proxy @5px (sample macro) | new pin pairs | new cross-GT-net pairs | baseline pairs split |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3_frame_guard | 11.46% | 16 / 744 | 100564 | 13.14% | 0 | 0 | 0 |
| v5_palette_recall | 21.67% | 102 / 744 | 115557 | 21.04% | 28 | 8 | 0 |
| v6_palette_bridge_guard | 19.79% | 86 / 744 | 111955 | 20.15% | 14 | 0 | 0 |
| v7_palette_corridor_recall | 30.99% | 167 / 744 | 138106 | 38.05% | 0 | 0 | 0 |

## Assembled NetLine geometry

| Version | GT-edge macro coverage @5px | edges >=80% | segment count | geometry precision proxy @5px |
|---|---:|---:|---:|---:|
| v3_frame_guard | 7.59% | 8 / 744 | 1946 | 10.73% |
| v5_palette_recall | 16.25% | 80 / 744 | 1981 | 18.42% |
| v6_palette_bridge_guard | 14.46% | 66 / 744 | 1972 | 17.41% |
| v7_palette_corridor_recall | 27.51% | 160 / 744 | 2127 | 43.05% |

These are assembled NetLine output segments rasterized at 1 px and evaluated as a geometry-only 5 px proximity diagnostic. They are not electrical or official Wire scores.

## Per-sample coverage

| Sample | v3 mask | v5 mask | v6 mask | v7 mask | v3 NetLine | v7 NetLine | v7 edges >=80% | v7 mask px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0001 | 23.78% | 23.78% | 23.78% | 30.69% | 16.07% | 23.23% | 19 | 20785 |
| 0002 | 11.25% | 97.69% | 79.58% | 32.36% | 8.83% | 30.36% | 11 | 6707 |
| 0003 | 10.35% | 10.35% | 10.35% | 29.11% | 8.30% | 27.54% | 10 | 8387 |
| 0004 | 8.85% | 8.85% | 8.85% | 30.77% | 6.14% | 28.43% | 14 | 7736 |
| 0005 | 5.51% | 5.51% | 5.51% | 30.81% | 3.54% | 29.14% | 14 | 9092 |
| 0006 | 5.95% | 5.95% | 5.95% | 29.01% | 5.01% | 28.25% | 21 | 12321 |
| 0007 | 4.65% | 4.65% | 4.65% | 42.84% | 3.79% | 42.53% | 27 | 11119 |
| 0008 | 12.27% | 95.04% | 81.68% | 30.63% | 9.66% | 28.32% | 9 | 7355 |
| 0009 | 7.71% | 7.71% | 7.71% | 38.92% | 6.71% | 38.34% | 14 | 4918 |
| 0010 | 7.37% | 7.37% | 7.37% | 31.60% | 6.23% | 30.93% | 20 | 12543 |
| 0011 | 10.32% | 10.32% | 10.32% | 22.95% | 7.12% | 20.59% | 4 | 5984 |
| 0012 | 12.69% | 12.69% | 12.69% | 16.93% | 5.11% | 9.33% | 1 | 13662 |
| 0013 | 21.48% | 21.48% | 21.48% | 24.82% | 4.56% | 8.73% | 3 | 17497 |

## Interpretation

The stage audit showed candidate evidence exists before global palette selection: for the 11 samples other than 0002 and 0008, line-supported candidates covered 633/654 edges to at least 80%, but score gating reduced that to 12/654. v5/v6 did not change those samples because their additional palettes remained below the 71% relative threshold.

v7 is non-default. It considers line-supported palette components only when they meet the predicted terminal corridor, preserves the complete v3 baseline, applies the existing body/frame suppression to supplemental pixels, and rejects any supplemental connected component that joins multiple baseline mask components. It does not create pixels from pin proximity alone.

The conservative bridge rule rejects possible genuine wires that connect fragmented baseline regions. The unmarked crossing and T-junction behavior was checked with synthetic topology tests. Zero cross-GT-net pin pairs on this development set is encouraging, but does not establish general safety or accuracy.

The cached default output pixel counts reproduced for 12/13 samples; one sample differed by the known 3-pixel replay discrepancy. Coverage and precision are development diagnostics and do not establish competition accuracy.

Default configuration remains `v3_frame_guard`; this candidate is not recommended as default based on 13 samples alone.
