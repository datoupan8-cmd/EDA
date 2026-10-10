"""Isolated, GT-free tip policies. No candidate generation or file reads.

These policies are experiments only. They do not change any public Stage,
schema, global registry, model, OCR or current configuration.
"""
from __future__ import annotations
import copy
import math
import numpy as np

from pcb.pin.localization.directional_v5 import build_raw_ink,_normal_support
from pcb.pin.localization.continuity_v6 import continuity_features,verify_outward_continuity
from pcb.pin.localization.tip_placement_v4 import place_boundary_tip,image_scale,NORMALS
from pcb.pin_semantics_v5 import _name_score
from pin_skeleton_followup import number_bbox_score

# Reuse the earlier P6.4 OCR confidence criterion. Not tuned on Check.
CONFIDENCE_MIN=.8


def trusted_joint_pin(pin,event,terminal,component,roles):
    """Require selected number/name evidence, not just a non-empty key."""
    if (not pin.exportable or not pin.name or pin.number_confidence<CONFIDENCE_MIN
            or pin.name_confidence<CONFIDENCE_MIN or event.get('number_duplicate')):
        return False
    numbers=[r for r in roles if r.token.text==event.get('number_token')]
    names=[r for r in roles if r.token.text==event.get('name_token')]
    return (any(number_bbox_score(r,terminal,component)>0 for r in numbers)
            and any(_name_score(r,terminal,component)>0 for r in names))


def visible_stub_endpoint(ink,terminal,texts,image_shape):
    """Return only the first supported local run endpoint, never a remote net end.

    Connected/unbounded wires, image-edge truncation and OCR-overlapped ends
    are explicitly ambiguous and retain the original tip. An image endpoint
    is still a hypothesis for target tip, not assumed equivalent to GT.
    """
    scale=image_scale(image_shape);reach=max(8,int(round(20*scale)))
    evidence=continuity_features(ink,terminal,reach)
    if (evidence['first_offset']>4 or evidence['span']<max(5,int(round(8*scale)))
            or evidence['occupancy']<.75):
        return tuple(terminal['tip']),{'applied':False,'reason':'weak_or_discontinuous_stub',**evidence}
    extension=evidence['first_offset']+evidence['span']-1
    # Need two actual empty samples inside the measured image, not padding.
    side=terminal['side'];bx,by=terminal['base']
    boundary=round(bx if side in ('left','right') else by)
    tangent=round(by if side in ('left','right') else bx)
    signal=_normal_support(ink,side,boundary,tangent,reach)
    end_index=int(extension-2)
    if end_index+2>=len(signal) or signal[end_index+1:end_index+3].any():
        return tuple(terminal['tip']),{'applied':False,'reason':'connected_or_truncated_no_visible_end',**evidence}
    nx,ny=NORMALS[side];point=(float(bx+nx*extension),float(by+ny*extension))
    h,w=image_shape[:2]
    if not (0<=point[0]<w and 0<=point[1]<h):
        return tuple(terminal['tip']),{'applied':False,'reason':'image_edge',**evidence}
    for token in texts:
        a,b,c,d=token.bbox
        if a-2<=point[0]<=c+2 and b-2<=point[1]<=d+2:
            return tuple(terminal['tip']),{'applied':False,'reason':'endpoint_overlaps_ocr',**evidence}
    return point,{'applied':True,'reason':'supported_visible_local_endpoint','extension':extension,**evidence}


def refine_scene_tips(scene,localization,roles,image,policy):
    """Only tip normal coordinates change; rejected candidates remain present."""
    if policy not in {'joint','endpoint'}:raise ValueError('Unknown experimental tip policy')
    result=copy.deepcopy(scene);loc=copy.deepcopy(localization)
    ink=build_raw_ink(image);events={}
    for event in scene.diagnostics['pin_events']:
        events.setdefault(event['component'],[]).append(event)
    decisions=[]
    for c,(owner,terms) in zip(result.components,loc.terminals):
        if c.key!=owner.key or len(c.pins)!=len(terms):raise AssertionError('Pin/candidate mapping changed')
        if c.type!='box':continue
        component_events=events.get(c.key,[])
        if len(component_events)!=len(terms):raise AssertionError('Event/candidate mapping changed')
        for index,(pin,term,event) in enumerate(zip(c.pins,terms,component_events)):
            before=tuple(pin.tip)
            if policy=='joint':
                if trusted_joint_pin(pin,event,term,c,roles):
                    valid,detail=verify_outward_continuity(ink,term,image.shape)
                    if valid:
                        point,detail_tip=place_boundary_tip(term,image.shape)
                        detail={**detail,**detail_tip,'reason':'trusted_joint_and_continuous_stub'}
                    else:
                        point=before;detail={**detail,'applied':False,'reason':'weak_continuity'}
                else:
                    point=before;detail={'applied':False,'reason':'untrusted_number_or_name'}
            else:
                point,detail=visible_stub_endpoint(ink,term,result.texts,image.shape)
            pin.tip=tuple(point);term['tip']=tuple(point)
            normal_axis=0 if term['side'] in ('left','right') else 1
            tangent_axis=1-normal_axis
            if before[tangent_axis]!=pin.tip[tangent_axis] or tuple(pin.base)!=tuple(term['base']):
                raise AssertionError('Base/tangent invariant failed')
            # Shared P3 uses infinity for "no first ink". Keep the decision
            # unchanged, but JSON diagnostics must represent missing as null.
            detail={k:(None if isinstance(v,(int,float,np.floating)) and not math.isfinite(v) else v)
                    for k,v in detail.items()}
            decisions.append({'component':c.key,'candidate_index':index,'side':term['side'],
                              'number':pin.number,'name':pin.name,'old_tip':before,'new_tip':pin.tip,
                              'moved':math.dist(before,pin.tip)>1e-9,**detail})
    for original,new in zip(localization.terminals,loc.terminals):
        if len(original[1])!=len(new[1]):raise AssertionError('Candidate count changed')
        for a,b in zip(original[1],new[1]):
            if any(a[k]!=b[k] for k in ('base','side','method')):raise AssertionError('Candidate identity changed')
    result.diagnostics['tip_regression_experiment']={'policy':policy,'decisions':decisions}
    # Retain original event fields and provenance, updating only tip geometry.
    result.diagnostics['pin_events']=[{**e,'tip':tuple(result.components[i].pins[j].tip)}
        for i,c in enumerate(result.components) for j,e in enumerate(events.get(c.key,[]))]
    return result,loc,decisions
