"""GT-free local endpoint evidence for already moved box tips; no file access.

Only isolated straight stub ends are accepted. A corner/junction/remote wire
end is not silently equated with the target Pin point. All constants are
frozen before the new Design/Check evaluation; no case/source-name rules.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from pcb.pin.localization.directional_v5 import build_raw_ink,_normal_support
from pcb.pin.localization.continuity_v6 import continuity_features
from pcb.pin.localization.tip_placement_v4 import image_scale,NORMALS

PARAMETERS={'normal_body_fraction':.5,'old_tip_reach_factor':2.,'pitch_reach_factor':1.5,
            'near_offset_max':4.,'minimum_occupancy':.75,'terminal_count_change':False,
            'geometry_changes_only_previously_moved_tips':True}


def local_reach(terminal,component,peers):
    """Use symbol size and same-side pitch rather than page size alone."""
    side=terminal['side'];normal=0 if side in ('left','right') else 1;tangent=1-normal
    bbox=component.body_bbox or component.bbox
    normal_span=bbox[2+normal]-bbox[normal]
    positions=sorted(set(float(t['base'][tangent]) for t in peers if t['side']==side))
    gaps=[b-a for a,b in zip(positions,positions[1:]) if b-a>0]
    pitch=float(np.median(gaps)) if gaps else 0.
    old_extension=math.dist(terminal['base'],terminal['tip'])
    return max(3,int(math.floor(min(PARAMETERS['normal_body_fraction']*normal_span,
                      max(PARAMETERS['old_tip_reach_factor']*old_extension,
                          PARAMETERS['pitch_reach_factor']*pitch)))))


def perpendicular_support(ink,point,side,scale):
    """An ink continuation sideways indicates a turn/bus, not a free stub end."""
    h,w=ink.shape;px,py=map(round,point)
    extent=max(5,round(4*scale));core=max(2,round(1.5*scale));counts=[]
    for sign in (-1,1):
        support=0
        for offset in range(core+1,extent+1):
            x=px if side in ('left','right') else px+sign*offset
            y=py+sign*offset if side in ('left','right') else py
            if not (0<=x<w and 0<=y<h):continue
            normal_pixels=(ink[y,max(0,x-1):min(w,x+2)] if side in ('left','right')
                           else ink[max(0,y-1):min(h,y+2),x])
            support+=int(normal_pixels.any())
        counts.append(support)
    return max(counts,default=0)


def isolated_stub_endpoint(ink,terminal,component,peers,texts,image_shape):
    """Evidence only: outward ink run must end locally without a sideways branch."""
    scale=image_scale(image_shape);reach=local_reach(terminal,component,peers)
    features=continuity_features(ink,terminal,reach)
    detail={'reach':reach,**features}
    if (features['first_offset']>PARAMETERS['near_offset_max']
            or features['span']<max(5,round(8*scale))
            or features['occupancy']<PARAMETERS['minimum_occupancy']):
        return None,{**detail,'reason':'weak_local_run'}
    extension=features['first_offset']+features['span']-1
    side=terminal['side'];bx,by=terminal['base'];nx,ny=NORMALS[side]
    point=(float(bx+nx*extension),float(by+ny*extension))
    h,w=image_shape[:2]
    if not (0<=point[0]+nx*2<w and 0<=point[1]+ny*2<h):
        return None,{**detail,'reason':'image_edge'}
    boundary=round(bx if side in ('left','right') else by)
    tangent=round(by if side in ('left','right') else bx)
    signal=_normal_support(ink,side,boundary,tangent,reach)
    end=int(extension-2)
    if end+2>=len(signal) or signal[end+1:end+3].any():
        return None,{**detail,'reason':'no_measured_local_end'}
    lateral=perpendicular_support(ink,point,side,scale)
    if lateral>=max(3,round(2*scale)):
        return None,{**detail,'reason':'turn_or_junction','lateral_support':lateral}
    if any(a-2<=point[0]<=c+2 and b-2<=point[1]<=d+2 for a,b,c,d in (t.bbox for t in texts)):
        return None,{**detail,'reason':'ocr_overlap'}
    return point,{**detail,'reason':'isolated_local_stub_end','extension':extension,'lateral_support':lateral}


def refine_existing_changes(scene,localization,original,image):
    """Start from G; only its existing moved tips can receive image-length evidence."""
    result=copy.deepcopy(scene);output=copy.deepcopy(localization);ink=build_raw_ink(image)
    decisions=[]
    for c,(owner,terms),(old_owner,old_terms) in zip(result.components,output.terminals,original.terminals):
        if c.key!=owner.key or c.key!=old_owner.key or len(terms)!=len(old_terms):
            raise AssertionError('Candidate mapping changed')
        if c.type!='box':continue
        for index,(pin,term,old) in enumerate(zip(c.pins,terms,old_terms)):
            if math.dist(term['tip'],old['tip'])<1e-9:continue
            point,detail=isolated_stub_endpoint(ink,old,c,old_terms,result.texts,image.shape)
            before=tuple(term['tip'])
            if point is not None and math.dist(point,term['base'])>math.dist(before,term['base']):
                pin.tip=point;term['tip']=point
            decisions.append({'component':c.key,'candidate_index':index,'side':term['side'],
                              'before':before,'after':tuple(term['tip']),
                              'changed':math.dist(before,term['tip'])>1e-9,**detail})
    for (_,a),(_,b) in zip(localization.terminals,output.terminals):
        if len(a)!=len(b):raise AssertionError('Candidate count changed')
        for x,y in zip(a,b):
            if {k:v for k,v in x.items() if k!='tip'}!={k:v for k,v in y.items() if k!='tip'}:
                raise AssertionError('Candidate attribute changed')
            tangent=1 if x['side'] in ('left','right') else 0
            if x['tip'][tangent]!=y['tip'][tangent]:raise AssertionError('Tangent changed')
    by_owner={c.key:c for c in result.components};indices={}
    for event in result.diagnostics['pin_events']:
        owner=event['component'];index=indices.get(owner,0);indices[owner]=index+1
        event['tip']=by_owner[owner].pins[index].tip
    for detail in decisions:
        for field,value in list(detail.items()):
            if isinstance(value,(int,float,np.floating)) and not math.isfinite(value):detail[field]=None
    result.diagnostics['pin_tip_local_geometry']={'parameters':PARAMETERS,'decisions':decisions}
    return result,output,decisions
