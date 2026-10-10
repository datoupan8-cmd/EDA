"""Unchanged runtime helpers extracted from the L39 audit module."""
from __future__ import annotations
from collections import defaultdict
import math
def event_lookup(raw,diagnostics):
    groups=defaultdict(list)
    for event in diagnostics.get('pin_events',[]):
        groups[event['component']].append(event)
    for block in raw:
        if len(groups[block['component']]) != len(block['terminals']):
            raise AssertionError(f"Raw terminal/event count mismatch: {block['component']}")
        for terminal,event in zip(block['terminals'],groups[block['component']]):
            if math.dist(terminal['tip'],event['tip'])>1e-7 or terminal['side']!=event['side']:
                raise AssertionError('Event/terminal geometry mismatch')
    return groups

class BoxOrderedSemantics:
    """Reuse existing V6 only on box; every other type stays on formal V3."""
    def run(self,component,localization,context):
        from pcb.pin.semantics.v3 import PinSemanticsStageV3
        from .ordered_v6 import assign_pin_semantics_v6
        output=PinSemanticsStageV3().run(component,localization,context)
        old=event_lookup([{'component':c.key,'terminals':terms} for c,terms in localization.terminals],output.diagnostics)
        rows=[]
        for item,terminals in localization.terminals:
            if item.type=='box':
                pins,events=assign_pin_semantics_v6(item,terminals,component.roles)
                item.pins=pins
                rows.extend({'component':item.key,**e} for e in events)
            else:
                rows.extend(old[item.key])
        output.diagnostics['pin_events']=rows
        output.diagnostics['pin_semantics']='box_only_existing_v6_other_types_v3'
        output.diagnostics['exportable_pins']=sum(p.exportable for c in component.components for p in c.pins)
        output.diagnostics['pin_v3_frozen']=False
        return output

def interval_distance(value,low,high):
    return max(float(low)-float(value),0.,float(value)-float(high))

def number_bbox_score(role,terminal,component):
    """Experimental number distance uses the text extent, with frozen V5 gates.

    Numbers printed above a wire are not centered on the wire. Keep the
    outside/role/regex tests and all limits; change only how distance is measured.
    The number center must still be outside the correct component side.
    """
    from .semantics_v3 import PIN_NUMBER
    from .scores_v5 import _number_normal_distance,_tangent_limit
    if not PIN_NUMBER.fullmatch(role.normalized): return 0.
    if role.role=='PIN_NUMBER': role_bonus=.25
    elif component.type in {'box','block'} and role.role=='VALUE' and role.normalized.isdigit(): role_bonus=0.
    else: return 0.
    box=component.body_bbox or component.bbox
    if _number_normal_distance(role,terminal,box) is None: return 0.
    x1,y1,x2,y2=box;a,b,c,d=role.token.bbox
    side=terminal['side'];vertical=side in ('left','right')
    normal={'left':max(0.,x1-c),'right':max(0.,a-x2),
            'top':max(0.,y1-d),'bottom':max(0.,b-y2)}[side]
    tangent=interval_distance(terminal['base'][1] if vertical else terminal['base'][0],
                              b if vertical else a,d if vertical else c)
    span=x2-x1 if vertical else y2-y1
    normal_limit=max(30.,min(120.,.25*span))
    tangent_limit=_tangent_limit(role,terminal,box)
    if normal>normal_limit or tangent>tangent_limit: return 0.
    return (1.+2.*(1.-tangent/tangent_limit)+.5*(1.-normal/normal_limit)
            +float(role.confidence)+role_bonus)

class BoxBBoxSemantics:
    """Process-local scorer adapter, for serial experiments only (not production).

    Reuses V6 ordering/pairing/dedup with number_bbox_score. The imported scoring
    reference is restored in finally; no source or registry files are modified.
    """
    def run(self,component,localization,context):
        from . import ordered_v6 as ordered
        old_score=ordered._number_score
        try:
            ordered._number_score=number_bbox_score
            output=BoxOrderedSemantics().run(component,localization,context)
        finally:
            ordered._number_score=old_score
        output.diagnostics['pin_semantics']='box_only_ordered_bbox_number_distance_other_types_v3'
        return output
