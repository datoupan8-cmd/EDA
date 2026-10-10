"""Expand local abstention eligibility, with frozen words/costs/constraints.

Original feasible choices are retained on equal objective. No image, OCR,
annotation, model, or submission access belongs to this decoder adapter.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict

import pin_joint_abstention_l7_2 as prior
import pin_word_decoder_l3 as legacy

LEGACY_SOLVE = prior.LEGACY_SOLVE
PRIOR_WRAPPER = prior.solve_options
PRIVATE_PREFIX = prior.PRIVATE_PREFIX
POLICY = {
    'single_variable': 'explicit local abstention eligibility includes original optimal owners',
    'eligible': 'nonempty box, original optimal or proven infeasible; not timeout/other failure',
    'abstention_score': 0.0,
    'feasible_keep_rule': 'strictly higher sum of unchanged option scores; original on tie/failure',
    'decoder': asdict(legacy.POLICY),
    'pool': 'frozen L7.3 physical resources and readings',
    'constraints': 'legacy physical word, number uniqueness and side order unchanged',
    'identity_and_export': 'same L7.2 private identity, never exported; geometry retained',
}


def solve_options(options: list[dict], terminals: list[dict], policy=legacy.POLICY):
    """Allow abstention within feasible owners without changing pair scores."""
    original, before = LEGACY_SOLVE(options, terminals, policy)
    trace = {**before, 'legacy_solver': copy.deepcopy(before),
             'legacy_selected': copy.deepcopy(original), 'rescue_attempted': False,
             'abstained_terminals': [], 'added_semantic_options': 0,
             'original_selection_retained': True, 'selection_reason': 'ineligible_original'}
    feasible = before.get('status') == 'optimal'
    infeasible = before.get('solver_status') == 2
    if not terminals or not (feasible or infeasible):
        return original, trace
    augmented = list(options) + [prior.abstention_option(i, t) for i, t in enumerate(terminals)]
    selected, after = LEGACY_SOLVE(augmented, terminals, policy)
    trace.update(rescue_attempted=True, rescue_solver=copy.deepcopy(after),
                 added_semantic_options=len(terminals))
    if after.get('status') != 'optimal':
        trace['selection_reason'] = 'augmented_failure_keep_original'
        return original, trace
    original_score = math.fsum(p['score'] for p in original)
    augmented_score = math.fsum(p['score'] for p in selected)
    trace.update(original_selected_score=original_score, augmented_selected_score=augmented_score)
    if feasible and augmented_score <= original_score:
        trace['selection_reason'] = 'no_strict_objective_improvement_keep_original'
        return original, trace
    # Actual selected-solver diagnostics supersede inherited fallback fields.
    trace.pop('solver_status', None)
    trace.pop('message', None)
    trace.update(after)
    trace.update(original_selection_retained=False,
                 selection_reason='higher_objective' if feasible else 'original_infeasible_rescued',
                 abstained_terminals=[p['terminal'] for p in selected if p.get('abstain')])
    return selected, trace


def stage(baseline, raw: list[dict], pool: dict, policy=legacy.POLICY):
    """Reuse the entire L7.2 adapter/export handling; scope the solver swap."""
    original = prior.solve_options
    if original is not PRIOR_WRAPPER:
        raise RuntimeError('L7.2 solve_options already wrapped by another experiment')
    try:
        prior.solve_options = solve_options
        scene, trace = prior.stage(baseline, raw, pool, policy)
    finally:
        prior.solve_options = original
    scene.diagnostics['L7_4_step'] = 'local abstention also in original feasible box owners'
    return scene, trace
