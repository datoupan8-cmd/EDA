"""One-factor candidate experiment: tolerate pixel-scale boundary contact jitter.

Only the contact slice is wider than V3. Ink, OCR masking, total support,
scan ranges, grouping, dedup and initial tip extension are the original rules.
The supplied baseline sequence is immutable; new contacts are append-only.
No GT, case/source identifiers, OCR/model execution, or filesystem inputs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
import math

import cv2
import numpy as np
from pcb.pin_detection import _groups
from pcb.pin.localization.box_skeleton_experiment import raw_ink, text_mask, dimensions, tip_from_base


@dataclass(frozen=True)
class ContactPolicy:
    outward_contact_tolerance_scale: float = 1.0


POLICY=ContactPolicy()


def contact_rows(ink,box,reach,minimum,margin,tolerance):
    """V3 row/column tests, changing only the near-boundary contact window."""
    h,w=ink.shape; x1,y1,x2,y2=box
    values={side:[] for side in ('left','right','top','bottom')}
    for y in range(max(0,y1+margin),min(h,y2-margin+1)):
        left=ink[y,max(0,x1-reach):min(w,x1+2)]
        right=ink[y,max(0,x2-1):min(w,x2+reach+1)]
        if left.sum()>=minimum and left[-min(2+tolerance,len(left)):].any():values['left'].append(y)
        if right.sum()>=minimum and right[:min(2+tolerance,len(right))].any():values['right'].append(y)
    for x in range(max(0,x1+margin),min(w,x2-margin+1)):
        top=ink[max(0,y1-reach):min(h,y1+2),x]
        bottom=ink[max(0,y2-1):min(h,y2+reach+1),x]
        if top.sum()>=minimum and top[-min(2+tolerance,len(top)):].any():values['top'].append(x)
        if bottom.sum()>=minimum and bottom[:min(2+tolerance,len(bottom))].any():values['bottom'].append(x)
    return values


def append_contact_candidates(image,components,texts,baseline,policy=POLICY):
    """Keep all G terminals exactly; append independently scanned box contacts.

    Dedup uses tangential distance, so G's shorter normal tip cannot create
    duplicates with a candidate from the original V3 extension formula.
    """
    ink=raw_ink(image); masked=text_mask(ink.shape,texts); ink=ink & ~masked
    h,w=ink.shape; result=copy.deepcopy(baseline); decisions=[]
    if len(components)!=len(result):raise AssertionError('Owner count mismatch')
    for component,original,block in zip(components,baseline,result):
        if component.key!=block['component'] or original['component']!=component.key:
            raise AssertionError('Owner ordering mismatch')
        if component.type!='box':continue
        box,scale,reach,margin=dimensions(ink.shape,component)
        minimum=max(3,int(round(4*scale)))
        tolerance=max(1,int(math.ceil(policy.outward_contact_tolerance_scale*scale)))
        widened=contact_rows(ink,box,reach,minimum,margin,tolerance)
        strict=contact_rows(ink,box,reach,minimum,margin,0)
        x1,y1,x2,y2=box; proposals=[]
        for side,values in widened.items():
            for group in _groups(values,max(2,int(round(3*scale)))):
                coordinate=float(np.median(group))
                base={'left':(float(x1),coordinate),'right':(float(x2),coordinate),
                      'top':(coordinate,float(y1)),'bottom':(coordinate,float(y2))}[side]
                proposals.append({'base':base,'tip':tip_from_base(base,side,max(6.,13.*scale),ink.shape),
                    'side':side,'method':'contact_band_e4','wire_support_score':len(group),
                    'outward_contact_tolerance':tolerance})
        # Mirror original post-group same-side dedup; do not change its distance.
        unique=[]
        for proposal in proposals:
            old=next((t for t in unique if t['side']==proposal['side'] and math.dist(t['tip'],proposal['tip'])<7*scale),None)
            if old is not None:
                if proposal['wire_support_score']>old['wire_support_score']:old.update(proposal)
            else:unique.append(proposal)
        added,rejected=[],[]
        for proposal in unique:
            axis=1 if proposal['side'] in ('left','right') else 0
            if any(t['side']==proposal['side'] and abs(t['base'][axis]-proposal['base'][axis])<7*scale for t in block['terminals']):
                rejected.append({'reason':'same_side_tangent_duplicate','proposal':proposal}); continue
            # A change in the contact window must have supplied new accepted rows.
            tangent=proposal['base'][axis]
            if any(abs(v-tangent)<=3*scale for v in strict[proposal['side']]):
                rejected.append({'reason':'already_has_original_contact_support','proposal':proposal}); continue
            block['terminals'].append(copy.deepcopy(proposal)); added.append(proposal)
        if block['terminals'][:len(original['terminals'])]!=original['terminals']:
            raise AssertionError('Frozen G candidate prefix changed')
        decisions.append({'component':component.key,'tolerance':tolerance,'baseline_count':len(original['terminals']),
            'added_count':len(added),'added':added,'rejected':rejected,
            'accepted_rows':{s:{'original':len(strict[s]),'widened':len(widened[s])} for s in widened}})
    return result,{'policy':asdict(policy),'only_changed_factor':'near-boundary contact window',
        'unchanged':['ink threshold','OCR masks','minimum support','margin','reach','group gap','dedup','initial tip rule'],
        'added_count':sum(r['added_count'] for r in decisions),'decisions':decisions}
