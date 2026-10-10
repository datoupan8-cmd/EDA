"""L3.2: frozen words -> joint pin pair MILP, without OCR or target access.

Physical-word, number uniqueness and side order constraints are shared by
number/name hypotheses. Original L1 output is an explicit fallback strategy.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
import copy
import math
import re
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from pcb.schema import Pin
from pcb.text_detection import PIN_NUMBER_RE, SIGNAL_RE
from pin_word_reader_l3 import normalize, reading_options, POLICY as WORD_POLICY
from pin_terminal_reader_l2 import tangent, normal_inside
from pin_contextual_roles_e6 import assert_invariants
from pin_skeleton_followup import event_lookup


@dataclass(frozen=True)
class DecoderPolicy:
    top_role_candidates: int = 4
    minimum_pair_score: float = 3.4
    replacement_margin: float = .35
    old_fallback_score: float = 3.5
    number_inside_heights: float = 2.0
    name_edge_fraction: float = .45
    tangent_font_fraction: float = .80
    tangent_pitch_fraction: float = .45
    solver_seconds: float = 20.0


POLICY=DecoderPolicy()


def interval_distance(value,lo,hi):return max(lo-value,value-hi,0.)


def geometry(word,term,box,terms,kind,policy=POLICY):
    a,b,c,d=word['bbox'];side=term['side'];vertical=side in ('left','right')
    center=((a+c)/2,(b+d)/2);v=tangent(term['base'],side)
    tlo,thi=(b,d) if vertical else (a,c);height=max(1.,thi-tlo)
    span=box[2]-box[0] if vertical else box[3]-box[1]
    distances=[abs(tangent(t['base'],side)-v) for t in terms if t['side']==side
               and abs(tangent(t['base'],side)-v)>1e-6]
    pitch=min(distances,default=40.)
    limit=max(7.,policy.tangent_font_fraction*height,policy.tangent_pitch_fraction*pitch)
    delta=interval_distance(v,tlo,thi)
    if delta>limit:return None
    normal=normal_inside(center,side,box)
    if kind=='number':
        outer=max(30.,min(120.,.25*span))
        if not -outer<=normal<=policy.number_inside_heights*height:return None
        normal_score=1.-abs(min(normal,0.))/outer
    else:
        # Word extent nearest to the body edge matters, not only its center.
        near={'left':a-box[0],'right':box[2]-c,'top':b-box[1],'bottom':box[3]-d}[side]
        if normal < -2 or near>policy.name_edge_fraction*span or normal>span*.65:return None
        normal_score=1.-max(0.,near)/max(1.,policy.name_edge_fraction*span)
    return {'tangent':(tlo+thi)/2,'height':height,'normal':normal,
            'alignment':1.-delta/limit,'normal_score':normal_score,'limit':limit}


def role_options(words,component,term,terms,kind,policy=POLICY):
    own={normalize(x) for x in (component.key,component.name,component.value,component.model) if x}
    box=component.body_bbox or component.bbox;output=[]
    for word in words:
        g=geometry(word,term,box,terms,kind,policy)
        if not g:continue
        for read in reading_options(word):
            text=read['text']
            if text in own:continue
            if kind=='number':
                if not PIN_NUMBER_RE.fullmatch(text):continue
                # Alphanumeric identifiers inside a box are signal-name
                # hypotheses; outside BGA identifiers remain legal numbers.
                if not text.isdigit() and g['normal']>2:continue
            else:
                if not SIGNAL_RE.fullmatch(text):continue
                # Central model labels with no edge anchor are not pin names.
                span=box[2]-box[0] if term['side'] in ('left','right') else box[3]-box[1]
                if g['normal']>.40*span and re.search(r'[A-Z]{2,}\d{3,}',text):continue
            score=1.+2*g['alignment']+read['score']+.25*g['normal_score']
            output.append({'word':word['id'],'text':text,'confidence':read['score'],
                           'families':read['families'],'corroborated':read['corroborated'],
                           'original_global':read['original_global'],'bbox':word['bbox'],
                           'geometry':g,'score':score})
    output.sort(key=lambda r:(-r['score'],-len(r['families']),r['word'],r['text']))
    return output[:policy.top_role_candidates]


def replacement_allowed(old: Pin,number: dict,name: dict,score: float,old_score: float,
                        policy=POLICY):
    """No fragments or unsupported changed fields may overwrite an old export."""
    if not old.exportable:return True,'new_complete_pair'
    changed_number=number['text']!=str(old.number)
    changed_name=name['text']!=str(old.name or '')
    if not changed_number and not changed_name:return True,'same_fields'
    if changed_number and not ({'strip','direct'}<=set(number['families'])):
        return False,'new_number_lacks_strip_direct_agreement'
    if changed_name:
        before=normalize(old.name or '');after=name['text']
        if before and after!=before and after in before:
            return False,'proper_substring_cannot_replace_full_name'
        if not name['corroborated']:return False,'new_name_lacks_complete_word_agreement'
    if score<=old_score+policy.replacement_margin:return False,'insufficient_joint_improvement'
    return True,'corroborated_joint_improvement'


def pair_options(words,component,terms,old_pins,policy=POLICY):
    all_options=[];rejected=[]
    for index,(term,old) in enumerate(zip(terms,old_pins)):
        nums=role_options(words,component,term,terms,'number',policy)
        names=role_options(words,component,term,terms,'name',policy)
        pairs=[]
        for num in nums:
            for name in names:
                if num['word']==name['word']:continue
                ng,sg=num['geometry'],name['geometry']
                if ng['normal']>sg['normal']-1.:continue
                if abs(ng['tangent']-sg['tangent'])>max(8.,.75*max(ng['height'],sg['height'])):continue
                quality=(num['score']+name['score'])/2
                if quality<policy.minimum_pair_score:continue
                pairs.append({'terminal':index,'side':term['side'],'number':num['text'],'name':name['text'],
                    'number_word':num['word'],'name_word':name['word'],'number_token':num,'name_token':name,
                    'number_tangent':ng['tangent'],'name_tangent':sg['tangent'],
                    'quality':quality,'score':quality,'fallback':False})
        unchanged=[p['quality'] for p in pairs if old.exportable and p['number']==str(old.number)
                   and p['name']==str(old.name or '')]
        old_score=max(unchanged,default=policy.old_fallback_score) if old.exportable else 0.
        for pair in pairs:
            accepted,reason=replacement_allowed(old,pair['number_token'],pair['name_token'],pair['quality'],old_score,policy)
            if accepted:
                pair['reason']=reason
                if reason=='same_fields':pair['score']+=.05
                all_options.append(pair)
            else:rejected.append({'terminal':index,'number':pair['number'],'name':pair['name'],'reason':reason})
        # An explicit keep-old/abstain option for every terminal prevents a
        # failed solver or missing OCR from silently deleting baseline pins.
        old_number=next((r for r in nums if old.exportable and r['text']==str(old.number)),None)
        old_name=next((r for r in names if old.exportable and r['text']==str(old.name or '')),None)
        # Reserve observable fallback words too: keeping an old pin does not
        # authorize another terminal to consume that same physical label.
        all_options.append({'terminal':index,'side':term['side'],'number':str(old.number) if old.exportable else None,
            'name':old.name,'number_word':old_number['word'] if old_number else None,
            'name_word':old_name['word'] if old_name else None,
            'number_tangent':old_number['geometry']['tangent'] if old_number else None,
            'name_tangent':old_name['geometry']['tangent'] if old_name else None,
            'quality':old_score,'score':old_score+.025 if old.exportable else 0.,'fallback':True,'reason':'L1_fallback'})
    return all_options,rejected


def solve_options(options,terminals,policy=POLICY):
    """Joint binary set packing with a per-terminal fallback and side order.

    No guessed monotone pin-number sequence is imposed: geometric word order
    is constrained, while the actual number strings are read from the image.
    """
    if not options:return [],{'status':'empty'}
    groups=defaultdict(list)
    for j,p in enumerate(options):
        groups[('terminal',p['terminal'])].append(j)
        if p['number'] is not None:groups[('number',p['number'])].append(j)
        for wid in (p['number_word'],p['name_word']):
            if wid:groups[('word',wid)].append(j)
    rr=[];cc=[];vv=[];lower=[];upper=[]
    def constraint(columns,equal=False):
        i=len(lower)
        for j in columns:rr.append(i);cc.append(j);vv.append(1.)
        lower.append(1. if equal else -np.inf);upper.append(1.)
    for key,columns in groups.items():constraint(columns,key[0]=='terminal')
    order_constraints=0
    for side in ('left','right','top','bottom'):
        idx=[j for j,p in enumerate(options) if p['side']==side]
        for offset,j in enumerate(idx):
            p=options[j];pv=tangent(terminals[p['terminal']]['base'],side)
            for k in idx[offset+1:]:
                q=options[k];qv=tangent(terminals[q['terminal']]['base'],side)
                if p['terminal']==q['terminal'] or abs(pv-qv)<1e-6:continue
                crosses=any(p[field] is not None and q[field] is not None and
                            (pv-qv)*(p[field]-q[field])<0 for field in ('number_tangent','name_tangent'))
                if crosses:
                    constraint([j,k]);order_constraints+=1
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(lower),len(options))).tocsc()
    result=milp(-np.asarray([p['score'] for p in options],float),integrality=np.ones(len(options)),
                bounds=Bounds(0.,1.),constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper)),
                options={'time_limit':policy.solver_seconds,'mip_rel_gap':0.})
    if result.status!=0 or result.x is None:
        return [],{'status':'fallback','solver_status':int(result.status),'message':result.message,
                   'variables':len(options),'constraints':len(lower),'order_constraints':order_constraints}
    selected=[p for p,x in zip(options,result.x) if x>.5]
    if len(selected)!=len(terminals) or len({p['terminal'] for p in selected})!=len(terminals):
        raise AssertionError('Solver terminal exclusivity broken')
    words=[w for p in selected for w in (p['number_word'],p['name_word']) if w]
    if len(words)!=len(set(words)):raise AssertionError('Physical-word reuse')
    numbers=[p['number'] for p in selected if p['number'] is not None]
    if len(numbers)!=len(set(numbers)):raise AssertionError('Number collision')
    return selected,{'status':'optimal','variables':len(options),'constraints':len(lower),
                     'order_constraints':order_constraints,'objective':float(-result.fun)}


def stage_c(baseline,raw,pool,policy=POLICY):
    """C consumes the exact B word pool and geometry, without any OCR invocation."""
    scene=copy.deepcopy(baseline);lookup={r['component']:r['terminals'] for r in raw}
    events=event_lookup(raw,baseline.diagnostics);output=[];trace=[]
    for component in scene.components:
        terms=lookup[component.key]
        if component.type!='box':output.extend(events[component.key]);continue
        options,rejected=pair_options(pool[component.key]['words'],component,terms,component.pins,policy)
        selected,solver=solve_options(options,terms,policy)
        chosen={p['terminal']:p for p in selected}
        for i,term in enumerate(terms):
            decision=chosen.get(i);event=copy.deepcopy(events[component.key][i])
            if decision and not decision['fallback']:
                num,name=decision['number_token'],decision['name_token']
                previous_pin=component.pins[i]
                component.pins[i]=Pin(decision['number'],decision['name'],previous_pin.tip,base=previous_pin.base,
                    side=term['side'],exportable=not component.key.startswith('UNRESOLVED_'),
                    number_confidence=num['confidence'],name_confidence=name['confidence'],
                    observable_number=decision['number'],number_source='L3_complete_word_joint')
                event.update(number=decision['number'],pinname=decision['name'],exportable=component.pins[i].exportable,
                             number_token=num['text'],name_token=name['text'],semantic_method='L3_word_pair_MILP',
                             number_word_id=num['word'],name_word_id=name['word'],joint_assignment_score=decision['quality'])
            output.append(event)
        trace.append({'component':component.key,'solver':solver,'selected':selected,'rejected_replacements':rejected})
    scene.diagnostics['pin_events']=output;scene.diagnostics['L3_step']='3.2 frozen words + joint decoder'
    assert_invariants(baseline,scene,raw)
    return scene,trace
