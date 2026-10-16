"""Only the conflict-adoption variable differs from L40."""
from __future__ import annotations

from .structural_l40 import L40Semantics,apply as structure_apply,POLICY


def safe_transactions(pins,proposals):
    active=dict(proposals);blocked=[]
    while True:
        reserved={p.number for i,p in enumerate(pins) if i not in active and p.exportable}
        reject=[i for i,row in active.items() if row[0] in reserved]
        if not reject:break
        for i in reject:
            blocked.append({'terminal_index':i,'number':active[i][0],'reason':'reserved_by_retained_pin'})
            del active[i]
    return active,blocked


def apply(scene,templates,policy=POLICY):
    return structure_apply(scene,templates,policy,conflict_resolver=safe_transactions)


class L40R1Semantics(L40Semantics):
    conflict_resolver=staticmethod(safe_transactions)
