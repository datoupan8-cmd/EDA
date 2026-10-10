"""One decoder change: explicit abstention only after proven infeasibility.

OCR, pair costs, resource exclusivity and side order remain legacy L3.
No image, annotation, model or submission is loaded by this module.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import pin_word_decoder_l3 as legacy
from pin_contextual_roles_e6 import assert_invariants
from pcb.schema import Scene

LEGACY_SOLVE = legacy.solve_options
PRIVATE_PREFIX = '__L7_ABSTAIN_'
POLICY = {
    'single_variable': 'explicit zero-score local abstention after solver_status=2',
    'eligible': 'legacy infeasible only; feasible/empty/timeout owners unchanged',
    'abstention_score': 0.0,
    'decoder': asdict(legacy.POLICY),
    'resources': 'same physical word and number uniqueness, same side order',
    'pool': 'original frozen L3 words; no L4 regrouping or L7.1 supplementation',
    'private_number': 'non-exported per-owner unique placeholder prevents duplicate-ref aliasing',
}


def abstention_option(index: int, term: dict[str, Any]) -> dict[str, Any]:
    """A missing semantic assignment; it consumes no word/number resource."""
    return {'terminal': index, 'side': term['side'], 'number': None, 'name': None,
            'number_word': None, 'name_word': None,
            'number_tangent': None, 'name_tangent': None,
            'quality': 0.0, 'score': 0.0, 'fallback': True,
            'abstain': True, 'reason': 'L7_local_conflict_abstention'}


def solve_options(options: list[dict], terminals: list[dict], policy=legacy.POLICY):
    """Reuse the original solver, preserving an already feasible selection."""
    original, before = LEGACY_SOLVE(options, terminals, policy)
    trace = {**before, 'legacy_solver': copy.deepcopy(before),
             'legacy_selected': copy.deepcopy(original), 'rescue_attempted': False,
             'abstained_terminals': [], 'added_semantic_options': 0}
    if before.get('solver_status') != 2:
        return original, trace
    augmented = list(options) + [abstention_option(i, t) for i, t in enumerate(terminals)]
    selected, after = LEGACY_SOLVE(augmented, terminals, policy)
    trace.update(rescue_attempted=True, rescue_solver=after,
                 added_semantic_options=len(terminals))
    if after['status'] != 'optimal':
        return original, trace
    trace.update({k: v for k, v in after.items()})
    trace['abstained_terminals'] = [p['terminal'] for p in selected if p.get('abstain')]
    return selected, trace


def private_number(index: int, occupied: set[str]) -> str:
    """Internal identity only, never a new inferred/observable pin number."""
    number = PRIVATE_PREFIX + str(index)
    while number in occupied:
        number += '_'
    occupied.add(number)
    return number


def stage(baseline: Scene, raw: list[dict], pool: dict,
          policy=legacy.POLICY) -> tuple[Scene, list[dict]]:
    """Apply legacy stage_c, then suppress the explicitly abstained Pins."""
    original = legacy.solve_options
    if original is not LEGACY_SOLVE:
        raise RuntimeError('Legacy solve_options already wrapped by another experiment')
    try:
        legacy.solve_options = solve_options
        scene, trace = legacy.stage_c(baseline, raw, pool, policy)
    finally:
        legacy.solve_options = original
    owners = {c.key: c for c in scene.components}
    events: dict[str, list[dict]] = {}
    for event in scene.diagnostics.get('pin_events', []):
        events.setdefault(event['component'], []).append(event)
    for row in trace:
        c = owners[row['component']]
        occupied = {str(p.number) for p in c.pins}
        row['abstention_export_trace'] = []
        for i in row['solver']['abstained_terminals']:
            pin, event = c.pins[i], events[c.key][i]
            previous = {'number': pin.number, 'name': pin.name, 'exportable': pin.exportable}
            pin.number = private_number(i, occupied)
            pin.name = None
            pin.exportable = False
            pin.observable_number = None
            pin.number_source = 'L7_abstained_private_identity'
            event.update(number=pin.number, pinname=None, exportable=False,
                         number_token=None, name_token=None,
                         number_word_id=None, name_word_id=None,
                         semantic_method='L7_local_conflict_abstention',
                         abstained=True, previous_semantics=previous)
            row['abstention_export_trace'].append({'terminal': i, 'previous': previous,
                                                 'private_number': pin.number})
    scene.diagnostics['L7_2_step'] = 'local abstention with legacy word/number/order constraints'
    assert_invariants(baseline, scene, raw)
    return scene, trace
