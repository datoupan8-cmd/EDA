"""Image-only continuous row cost on an otherwise identical legal option set.

No image/annotation IO or model access. This module does not create words,
roles, terminals, or pairs. Admission quality and constraints stay untouched.
"""
from __future__ import annotations

import copy
import math
from statistics import median
from typing import Any

import pin_word_decoder_l3 as legacy
from pin_terminal_reader_l2 import tangent

POLICY = {'single_variable': 'continuous signed same-side row position cost',
    'minimum_calibration_pairs': 3, 'residual_cap': 1.0, 'weight': 1.0,
    'position': 'unchanged frozen physical-word bbox center',
    'anchors': 'existing corroborated roles; number outside, name inside body',
    'matching': 'unique mutual nearest by tangent; median signed offsets',
    'missing_evidence': 'side scores unchanged', 'admission_and_quality': 'unchanged',
    'GT_used': False}


def word_tangent(box: list, side: str) -> float:
    a, b, c, d = box
    return float((b+d)/2 if side in ('left', 'right') else (a+c)/2)


def mutual_pairs(left: list[tuple[Any, float]], right: list[tuple[Any, float]]) -> list:
    """Accept only an unambiguous nearest position in both directions."""
    def closest(point, candidates):
        ranked = sorted((abs(point-v), str(k), k) for k, v in candidates)
        if not ranked or (len(ranked)>1 and math.isclose(ranked[0][0], ranked[1][0], abs_tol=1e-7)):
            return None
        return ranked[0][2]
    lr = {key: closest(value, right) for key, value in left}
    rl = {key: closest(value, left) for key, value in right}
    return [(key, other) for key, other in lr.items() if other is not None and rl.get(other)==key]


def calibrate(words: list[dict], component, terms: list[dict]) -> dict:
    """Learn layout offsets from frozen image-derived resources, never labels."""
    output = {}
    for side in ('left', 'right', 'top', 'bottom'):
        indices = [i for i, t in enumerate(terms) if t['side']==side]
        roles = {'number': {}, 'name': {}}
        for i in indices:
            for kind in roles:
                for r in legacy.role_options(words, component, terms[i], terms, kind):
                    if not r['corroborated']:
                        continue
                    normal = r['geometry']['normal']
                    if (kind=='number' and normal>0) or (kind=='name' and normal<0):
                        continue
                    old = roles[kind].get(r['word'])
                    # Position is identical across textual hypotheses of a
                    # resource. This tie break is deterministic and GT-free.
                    if old is None or (r['confidence'], r['score'], r['text']) > (
                            old['confidence'], old['score'], old['text']):
                        roles[kind][r['word']] = r
        numbers = [(k, word_tangent(r['bbox'], side)) for k, r in roles['number'].items()]
        names = [(k, word_tangent(r['bbox'], side)) for k, r in roles['name'].items()]
        points = [(i, tangent(terms[i]['base'], side)) for i in indices]
        number_name = mutual_pairs(numbers, names)
        terminal_name = mutual_pairs(points, names)
        np, sp, tp = dict(numbers), dict(names), dict(points)
        positions = sorted(set(sp.values()))
        gaps = [b-a for a, b in zip(positions, positions[1:]) if b-a>1e-7]
        supported = (len(number_name)>=POLICY['minimum_calibration_pairs']
                     and len(terminal_name)>=POLICY['minimum_calibration_pairs'] and bool(gaps))
        row = {'supported': supported, 'number_name_pair_count': len(number_name),
            'terminal_name_pair_count': len(terminal_name),
            'number_name_anchors': [{'number_word':n, 'name_word':s, 'signed_delta':np[n]-sp[s]}
                                   for n, s in number_name],
            'terminal_name_anchors': [{'terminal':i, 'name_word':s, 'signed_delta':sp[s]-tp[i]}
                                     for i, s in terminal_name]}
        if supported:
            delta_name = median(sp[s]-tp[i] for i, s in terminal_name)
            delta_pair = median(np[n]-sp[s] for n, s in number_name)
            row.update(pitch=median(gaps), delta_name=delta_name,
                       delta_number=delta_name+delta_pair, delta_number_minus_name=delta_pair)
        output[side] = row
    return output


def rescore(options: list[dict], words: list[dict], terms: list[dict], layout: dict) -> tuple[list, list]:
    """Change score only; no admission, quality, resources, or strings change."""
    result, trace = copy.deepcopy(options), []
    lookup = {w['id']: w for w in words}
    for option, old in zip(result, options):
        side = option['side']; fit = layout[side]
        if not fit['supported']:
            continue
        base = tangent(terms[option['terminal']]['base'], side)
        residuals = {}
        for kind in ('number', 'name'):
            wid = option.get(kind+'_word')
            residuals[kind] = (min(POLICY['residual_cap'], abs(
                word_tangent(lookup[wid]['bbox'], side) - base - fit['delta_'+kind]) / fit['pitch'])
                if wid is not None else 0.)
        penalty = POLICY['weight'] * (residuals['number']+residuals['name']) / 2
        option['score'] = old['score'] - penalty
        trace.append({'terminal':option['terminal'], 'side':side, 'number':option['number'],
            'name':option['name'], 'number_word':option.get('number_word'), 'name_word':option.get('name_word'),
            'fallback':option['fallback'], 'old_score':old['score'], 'new_score':option['score'],
            'penalty':penalty, 'residuals':residuals})
    if [{k:v for k,v in p.items() if k!='score'} for p in result] != [
            {k:v for k,v in p.items() if k!='score'} for p in options]:
        raise AssertionError('L11 changed more than objective scores')
    return result, trace


def stage(baseline, raw: list[dict], pool: dict):
    """Reuse the existing L7.4 adapter with a scoped score-only wrapper."""
    import pin_joint_abstention_l7_4 as prior
    original = legacy.pair_options
    score_trace = {}
    def scored_pairs(words, component, terms, old_pins, policy=legacy.POLICY):
        options, rejected = original(words, component, terms, old_pins, policy)
        layout = calibrate(words, component, terms)
        changed, cost = rescore(options, words, terms, layout)
        score_trace[component.key] = {'layout':layout, 'costs':cost,
            'legal_option_count':len(options), 'rejected_unchanged':rejected}
        return changed, rejected
    try:
        legacy.pair_options = scored_pairs
        scene, trace = prior.stage(baseline, raw, pool)
    finally:
        legacy.pair_options = original
    scene.diagnostics['L11_step'] = 'frozen legal options + continuous same-side row score'
    return scene, trace, score_trace
