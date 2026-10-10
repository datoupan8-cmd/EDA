"""L36: decode three ordered sequences together; no image/file/GT access.

The serial, scoped adapter reuses the complete historical V6 output path. It
changes only the two-pass number/name-row alignment to one three-way DP.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from pcb import pin_semantics_v6 as ordered
from pin_word_reader_l3 import stage_b


@dataclass(frozen=True)
class DecodePolicy:
    max_states: int = 5_000_000


POLICY = DecodePolicy()


def align_three(number_scores: np.ndarray, name_scores: np.ndarray,
                legal_pairs: np.ndarray, policy: DecodePolicy = POLICY
                ) -> tuple[dict[int, tuple[int | None, int | None, float]], dict]:
    """Exact maximum-score monotone alignment with independent free gaps.

Each terminal and each physical role instance is consumed at most once.
Triple score is the unchanged V6 number+name+0.5 score. Singles use their
unchanged individual score; triples require both individual scores > 0.
The three axes are terminals, number words, and name words, not pin numbers.
"""
    ns = np.asarray(number_scores, dtype=float)
    ss = np.asarray(name_scores, dtype=float)
    legal = np.asarray(legal_pairs, dtype=bool)
    if ns.ndim != 2 or ss.ndim != 2 or ns.shape[0] != ss.shape[0]:
        raise ValueError('Expected two score matrices sharing the terminal axis')
    n, m = ns.shape
    k = ss.shape[1]
    if legal.shape != (m, k) or not np.isfinite(ns).all() or not np.isfinite(ss).all():
        raise ValueError('Invalid pair eligibility or nonfinite scores')
    states = (n + 1) * (m + 1) * (k + 1)
    detail = {'terminal_count': n, 'number_token_count': m, 'name_token_count': k,
              'states': states, 'status': 'optimal', 'objective': 0.0}
    if states > policy.max_states:
        return {}, {**detail, 'status': 'capacity_fallback'}
    if not n or not (m or k):
        return {}, detail
    dp = np.zeros((n + 1, m + 1, k + 1), dtype=np.float64)
    trace = np.zeros(dp.shape, dtype=np.uint8)
    for i in range(n + 1):
        for j in range(m + 1):
            for h in range(k + 1):
                if not (i or j or h):
                    continue
                # Deterministic ties: triple > number > name > name gap >
                # number gap > terminal gap. No fitted score or gap penalty.
                best, action = -math.inf, 0
                if i:
                    best, action = float(dp[i - 1, j, h]), 1
                if j and dp[i, j - 1, h] >= best:
                    best, action = float(dp[i, j - 1, h]), 2
                if h and dp[i, j, h - 1] >= best:
                    best, action = float(dp[i, j, h - 1]), 3
                if i and h and ss[i - 1, h - 1] > 0:
                    value = dp[i - 1, j, h - 1] + ss[i - 1, h - 1]
                    if value >= best:
                        best, action = float(value), 4
                if i and j and ns[i - 1, j - 1] > 0:
                    value = dp[i - 1, j - 1, h] + ns[i - 1, j - 1]
                    if value >= best:
                        best, action = float(value), 5
                if (i and j and h and legal[j - 1, h - 1]
                        and ns[i - 1, j - 1] > 0 and ss[i - 1, h - 1] > 0):
                    value = (dp[i - 1, j - 1, h - 1] + ns[i - 1, j - 1]
                             + ss[i - 1, h - 1] + 0.5)
                    if value >= best:
                        best, action = float(value), 6
                dp[i, j, h], trace[i, j, h] = best, action
    assigned = {}
    i, j, h = n, m, k
    while i or j or h:
        action = int(trace[i, j, h])
        if action == 6:
            assigned[i - 1] = (j - 1, h - 1, float(ns[i - 1, j - 1] + ss[i - 1, h - 1] + .5))
            i -= 1; j -= 1; h -= 1
        elif action == 5:
            assigned[i - 1] = (j - 1, None, float(ns[i - 1, j - 1]))
            i -= 1; j -= 1
        elif action == 4:
            assigned[i - 1] = (None, h - 1, float(ss[i - 1, h - 1]))
            i -= 1; h -= 1
        elif action == 3:
            h -= 1
        elif action == 2:
            j -= 1
        elif action == 1:
            i -= 1
        else:
            raise AssertionError('Incomplete three-sequence traceback')
    detail['objective'] = float(dp[n, m, k])
    return assigned, detail


@contextmanager
def joint_adapter(traces: list[dict], policy: DecodePolicy = POLICY):
    """Serial-only adapter; original functions always restored, including errors."""
    previous_pair, previous_assign = ordered._pair_number_names, ordered._assign_side

    def defer_pairs(numbers, names, side, box):
        return [{'_L36_sequences': True, 'side': side, 'box': box,
                 'numbers': sorted(numbers, key=lambda r: ordered._role_tangent(r, side)),
                 'names': sorted(names, key=lambda r: ordered._role_tangent(r, side))}]

    def assign_side(terminals, bundle, component):
        if len(bundle) != 1 or not bundle[0].get('_L36_sequences'):
            raise AssertionError('Expected deferred role sequences')
        entry = bundle[0]
        side, box = entry['side'], entry['box']
        numbers, names = entry['numbers'], entry['names']
        items = sorted(terminals, key=lambda t: ordered._terminal_tangent(t[1]))
        # Existing role classification is disjoint. Do not let one physical
        # token satisfy both number and name when adapting future role types.
        if {r.index for r in numbers} & {r.index for r in names}:
            raise AssertionError('Same physical role offered as both number and name')
        ns = np.array([[ordered._number_score(r, t, component) for r in numbers]
                       for _, t in items], dtype=float).reshape(len(items), len(numbers))
        ss = np.array([[ordered._name_score(r, t, component) for r in names]
                       for _, t in items], dtype=float).reshape(len(items), len(names))
        legal = np.array([[abs(ordered._role_tangent(a, side) - ordered._role_tangent(b, side))
                           <= ordered._row_pair_limit(a, b, side, box)
                           for b in names] for a in numbers], dtype=bool).reshape(len(numbers), len(names))
        assigned, detail = align_three(ns, ss, legal, policy)
        detail.update(component=component.key, side=side)
        if detail['status'] == 'capacity_fallback':
            output = previous_assign(items, previous_pair(numbers, names, side, box), component)
            detail['selected'] = []; traces.append(detail)
            return output
        output, selected = {}, []
        for rank, (ni, si, score) in assigned.items():
            index, terminal = items[rank]
            num = numbers[ni] if ni is not None else None
            name = names[si] if si is not None else None
            tangent = [ordered._role_tangent(r, side) for r in (num, name) if r is not None]
            row = {'number': num, 'name': name, 'tangent': sum(tangent) / len(tangent)}
            output[index] = (row, score, rank, ordered._terminal_tangent(terminal))
            selected.append({'terminal_index': index, 'score': score,
                             'number_role_index': num.index if num else None,
                             'name_role_index': name.index if name else None,
                             'number_word_id': num.token.source_id if num else None,
                             'name_word_id': name.token.source_id if name else None,
                             'number': num.normalized if num else None,
                             'name': name.token.text.strip() if name else None})
        detail['selected'] = sorted(selected, key=lambda r: r['terminal_index'])
        traces.append(detail)
        return output

    ordered._pair_number_names, ordered._assign_side = defer_pairs, assign_side
    try:
        yield
    finally:
        ordered._pair_number_names, ordered._assign_side = previous_pair, previous_assign


def stage(scene, raw, pool, roles, policy: DecodePolicy = POLICY):
    """Frozen L3 B role/word preparation, with one substituted decoder."""
    traces: list[dict[str, Any]] = []
    with joint_adapter(traces, policy):
        candidate, preparation = stage_b(scene, raw, pool, roles)
    box_keys = {c.key for c in candidate.components if c.type == 'box'}
    side_lookup = {(r['component'], r['side']): r for r in traces}
    for event in candidate.diagnostics['pin_events']:
        if event['component'] in box_keys:
            detail = side_lookup[(event['component'], event['side'])]
            if detail['status'] == 'capacity_fallback':
                event['L36_capacity_fallback'] = True
            else:
                event['semantic_method'] = 'L36_simultaneous_three_sequence'
                event['side_alignment'].update(text_row_count=None,
                    text_rows_precomputed=False, alignment_state_count=detail['states'])
    candidate.diagnostics['L36_decode'] = traces
    candidate.diagnostics['L3_step'] = 'L36 frozen complete words + simultaneous side decoding'
    return candidate, {'sides': traces, 'role_preparation': preparation}
