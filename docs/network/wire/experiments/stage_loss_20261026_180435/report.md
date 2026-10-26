# Wire stage-loss trace — cached 13-sample replay

No Component/Pin model inference was rerun. Stage masks were captured from the existing production Wire implementation.

## Coverage by actual production stage

| Stage | GT edge macro coverage @5px | Edges >=80% |
|---|---:|---:|
| raw_palette_union | 100.00% | 744 |
| after_suppression_and_frame_union | 99.33% | 737 |
| after_line_support_union | 97.09% | 721 |
| after_score_gate_union | 16.03% | 59 |
| after_candidate_limit_union | 10.51% | 13 |
| after_close_and_border | 9.79% | 13 |
| after_terminal_corridor_restore | 11.46% | 16 |
| after_component_body_barrier | 11.46% | 16 |
| after_frame_guard | 11.46% | 16 |

## Version comparison

| Version | GT edge coverage @5px | Edges >=80% | mask pixels | pixel precision proxy @5px |
|---|---:|---:|---:|
| default_v3_frame_guard | 11.46% | 16 | 100564 | 13.14% |
| v5_palette_recall | 21.67% | 102 | 115557 | 21.04% |
| v6_palette_bridge_guard | 19.79% | 86 | 111955 | 20.15% |

## First-loss classification

| First loss stage for final edges below 80% | Edge count |
|---|---:|
| after_candidate_limit_union | 46 |
| after_line_support_union | 15 |
| after_score_gate_union | 660 |
| after_suppression_and_frame_union | 7 |

Full per-sample stage statistics, candidate scores and low-coverage edge details are in stage_loss.json.
