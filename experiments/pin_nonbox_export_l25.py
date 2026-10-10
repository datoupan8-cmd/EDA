"""Single-variable shadow admission: export existing unnumbered nonbox Pins.

No detection, OCR, target, network or geometry change belongs in this module.
Local UNK references remain opaque identities, never claimed as physical numbers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import math
import re

from pcb.schema import Scene

POLICY = {
    'single_variable': 'nonbox existing Pin exportable False -> True for proven missing number',
    'type_source': 'predicted component type; not GT',
    'unresolved_owner': 'retain original UNRESOLVED_ abstention',
    'local_reference': 'keep original UNK suffix without claiming visible number',
    'geometry': 'all existing candidates and positions unchanged',
}


def admit(scene: Scene, baseline_events: list[dict]) -> tuple[Scene, list[dict]]:
    """Change only exportable, retaining source input and every other Pin field."""
    result = copy.deepcopy(scene)
    events = defaultdict(list)
    for event in baseline_events:
        events[event['component']].append(event)
    trace = []
    for c in result.components:
        if c.type == 'box':
            continue
        rows = events[c.key]
        if len(rows) != len(c.pins):
            raise AssertionError('Nonbox archived event / Pin count mismatch')
        occupied = {p.number for p in c.pins if p.exportable}
        occurrences = Counter(p.number for p in c.pins)
        for i, (p, event) in enumerate(zip(c.pins, rows)):
            base_same = ((p.base is None and event.get('base') is None) or
                         (p.base is not None and event.get('base') is not None and
                          math.dist(p.base, event['base']) <= 1e-9))
            if (p.number != str(event['number']) or p.side != event['side'] or
                math.dist(p.tip, event['tip']) > 1e-9 or not base_same):
                raise AssertionError('Archived nonbox Pin / event identity or geometry mismatch')
            if p.exportable:
                continue
            reason = 'eligible_missing_number'
            if c.key.startswith('UNRESOLVED_'):
                reason = 'unresolved_owner_still_rejected'
            elif not (re.fullmatch(r'UNK\d+', p.number) and p.observable_number is None and
                      p.number_confidence == 0 and event.get('number_token') is None and
                      event.get('number_confidence', 0) == 0 and event.get('exportable') is False and
                      event.get('semantic_method') in {'internal_id_not_guessed','terminal_local_text'}):
                reason = 'not_proven_missing_number'
            elif p.number in occupied or occurrences[p.number] != 1:
                reason = 'local_ref_conflict_still_rejected'
            elif not (all(math.isfinite(float(v)) for v in p.tip) and
                      0 <= p.tip[0] <= scene.width and 0 <= p.tip[1] <= scene.height):
                reason = 'invalid_schema_geometry_still_rejected'
            before = asdict(p)
            if reason == 'eligible_missing_number':
                p.exportable = True
                occupied.add(p.number)
            trace.append({'owner': c.key, 'type': c.type, 'terminal': i,
                          'reason': reason, 'admitted': p.exportable,
                          'event_semantic_method': event.get('semantic_method'),
                          'previous_pin': before, 'new_pin': asdict(p),
                          'not_a_visible_number': True})
    result.diagnostics['pin_nonbox_export_l25'] = {
        'policy': copy.deepcopy(POLICY), 'changes': trace,
        'old_pin_events_are_frozen_baseline': True,
        'exported_after': sum(p.exportable for c in result.components for p in c.pins),
    }
    assert_invariants(scene, result, trace)
    return result, trace


def assert_invariants(before: Scene, after: Scene, trace: list[dict]) -> None:
    permitted = {(r['owner'], r['terminal']) for r in trace if r['admitted']}
    if (before.width, before.height, before.texts, before.nets) != (after.width, after.height, after.texts, after.nets):
        raise AssertionError('Public scene inputs changed')
    if len(before.components) != len(after.components):
        raise AssertionError('Component count changed')
    for a, b in zip(before.components, after.components):
        x,y=asdict(a),asdict(b);ap,bp=x.pop('pins'),y.pop('pins')
        if x != y or len(ap) != len(bp):
            raise AssertionError('Component or candidate count/order changed')
        for i,(p,q) in enumerate(zip(ap,bp)):
            if (a.key,i) in permitted:
                expected={**p,'exportable':True}
                if a.type=='box' or p['exportable'] or q != expected:
                    raise AssertionError('Admission changed a field other than exportable')
            elif p != q:
                raise AssertionError('Unselected Pin changed')


def old_network_projection(pred: dict, refs: set[str]) -> list[dict]:
    """Old-ref subgraph, independent of target and network numbering shifts."""
    from evaluate_v2 import net_members
    # Parse through the existing evaluator; do not alter exported identifiers.
    rows=[]
    for n in pred['nets'].values():
        members=sorted(refs.intersection(net_members(n)))
        if members:
            rows.append({'members':members,'edges':n['edges']})
    import json
    return sorted(rows,key=lambda r:json.dumps(r,sort_keys=True))
