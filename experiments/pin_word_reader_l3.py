"""L3.1 image-only complete-word OCR. No annotation or terminal generation.

Side strips are detected before individual word crops are recognized. The
saved physical-word pool is also the entire input of L3.2 (no second OCR run).
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from pcb.io import write_json
from pcb.schema import Text
from pcb.pin.local_ocr import rotate_points
from pcb.text_detection import classify_tokens
from pin_terminal_reader_l2 import TerminalRowOCR, SIDES, scale_of, tangent
from pin_contextual_roles_e6 import local_roles, assert_invariants
from pin_skeleton_followup import number_bbox_score, event_lookup


@dataclass(frozen=True)
class WordPolicy:
    magnification: float = 3.0
    strip_span: int = 256
    strip_overlap: int = 48
    padding: int = 12
    confidence: float = .80
    normal_max: float = 120.0
    normal_min: float = 30.0
    artificial_edge_margin: float = 1.0
    direct_padding: float = 2.0


POLICY = WordPolicy()


def normalize(text: str) -> str:
    return re.sub(r'\s+', '', str(text)).upper()


def intersection(a, b) -> float:
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def area(box) -> float:
    return max(0., box[2]-box[0])*max(0., box[3]-box[1])


def same_instance(a, b) -> bool:
    overlap = intersection(a,b); aa,bb = area(a),area(b)
    if min(aa,bb) <= 0: return False
    return overlap/(aa+bb-overlap) >= .45 or (
        overlap/min(aa,bb) >= .85 and max(aa,bb)/min(aa,bb) <= 2.5)


def strip_regions(component, terminals, texts, shape, policy=POLICY):
    """Whole-side strips with overlap; extend across any intersecting old word.

    Existing word extents are included before OCR, never truncated at a
    terminal midpoint or exactly at the body half. Original image coordinates.
    """
    a,b,c,d = component.body_bbox or component.bbox
    h,w = shape[:2]; scale = scale_of(shape); regions=[]
    for side in SIDES:
        terms=[t for t in terminals if t['side']==side]
        if not terms: continue
        vertical=side in ('left','right'); span=(c-a if vertical else d-b)
        outer=max(policy.normal_min*scale,min(policy.normal_max*scale,.25*span))
        inner=min(policy.normal_max*scale,.50*span)
        lo,hi=(b,d) if vertical else (a,c)
        margin=max(8.,8*scale)
        lo-=margin;hi+=margin
        step=policy.strip_span-policy.strip_overlap
        for start in np.arange(lo,hi,step):
            end=min(hi,start+policy.strip_span)
            if side=='left': box=[a-outer,start,a+inner,end]
            elif side=='right':box=[c-inner,start,c+outer,end]
            elif side=='top':box=[start,b-outer,end,b+inner]
            else:box=[start,d-inner,end,d+outer]
            # One pass avoids an expanding chain of unrelated nearby words.
            touching=[t.bbox for t in texts if intersection(box,t.bbox)>0]
            for t in touching:
                box=[min(box[0],t[0]-2),min(box[1],t[1]-2),max(box[2],t[2]+2),max(box[3],t[3]+2)]
            box=[max(0,math.floor(box[0])),max(0,math.floor(box[1])),
                 min(w,math.ceil(box[2])),min(h,math.ceil(box[3]))]
            if box[2]>box[0] and box[3]>box[1]:regions.append({'side':side,'roi':box})
            if end>=hi:break
    return regions


def group_words(readings: list[dict]) -> list[dict]:
    """Collapse physical duplicates/fragments, retaining every reading variant."""
    parent=list(range(len(readings)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(readings):
        for j in range(i):
            if same_instance(r['bbox'],readings[j]['bbox']):parent[find(i)]=find(j)
    groups={}
    for i,r in enumerate(readings):groups.setdefault(find(i),[]).append(r)
    output=[]
    for rows in groups.values():
        intact=[r for r in rows if r['complete']]
        usable=intact or rows
        bbox=[min(r['bbox'][0] for r in usable),min(r['bbox'][1] for r in usable),
              max(r['bbox'][2] for r in usable),max(r['bbox'][3] for r in usable)]
        wid=hashlib.sha256(json.dumps([round(v,2) for v in bbox]).encode()).hexdigest()[:16]
        output.append({'id':wid,'bbox':bbox,'readings':rows})
    return sorted(output,key=lambda r:(r['bbox'][1],r['bbox'][0],r['id']))


def reading_options(word: dict, confidence: float=POLICY.confidence) -> list[dict]:
    groups={}
    for r in word['readings']:
        if not r['complete'] or r['score']<confidence:continue
        n=normalize(r['text'])
        if n:groups.setdefault(n,[]).append(r)
    output=[]
    for n,rows in groups.items():
        families=sorted({r['family'] for r in rows})
        # Same weights, different crops: corroboration, NOT independent models.
        output.append({'text':n,'score':max(r['score'] for r in rows),
                       'families':families,'corroborated':len(families)>=2,
                       'complete':True,'original_global':any(r['family']=='global' for r in rows)})
    return sorted(output,key=lambda r:(-len(r['families']),-r['score'],r['text']))


def canonical_tokens(words: list[dict]) -> list[Text]:
    output=[]
    for word in words:
        readings=reading_options(word)
        if not readings:continue
        # A single new reading cannot displace an intact old word.
        agreed=[r for r in readings if r['corroborated']]
        globals_=[r for r in readings if r['original_global']]
        best=(agreed or globals_ or readings)[0]
        output.append(Text(best['text'],tuple(word['bbox']),best['score'],'L3_word:'+word['id']))
    return output


class WholeWordOCR:
    """Existing RapidOCR detector/recognizer and weights, private bounded runtime."""
    def __init__(self,cache: Path,policy=POLICY):
        self.cache=Path(cache);self.policy=policy
        self.backend=TerminalRowOCR(self.cache/'backend')
        self.calls=self.direct_calls=self.cache_hits=0

    def read_owner(self,image,component,terminals,texts):
        engine=self.backend.ensure_engine();h,w=image.shape[:2]
        regions=strip_regions(component,terminals,texts,image.shape,self.policy)
        selected=[t for t in texts if any(intersection(t.bbox,r['roi'])>0 for r in regions)]
        signature={'policy':asdict(self.policy),'models':self.backend.model_signature,
                   'regions':regions,'global':[(t.text,t.bbox,t.score) for t in selected]}
        key=hashlib.sha256(image.tobytes()+json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
        path=self.cache/(key+'.json')
        if path.exists():self.cache_hits+=1;return json.loads(path.read_text(encoding='utf-8'))
        readings=[{'text':t.text,'bbox':list(t.bbox),'score':float(t.score),'family':'global',
                   'complete':True,'side':None,'angle':None} for t in selected]
        for region in regions:
            x1,y1,x2,y2=region['roi'];crop=image[y1:y2,x1:x2]
            factor=self.policy.magnification;pad=self.policy.padding
            large=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
            sy,sx=large.shape[0]/crop.shape[0],large.shape[1]/crop.shape[1]
            large=cv2.copyMakeBorder(large,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255))
            ph,pw=large.shape[:2]
            angles=(0,180) if region['side'] in ('left','right') else (90,270)
            for angle in angles:
                output,_=engine(np.ascontiguousarray(np.rot90(large,angle//90)))
                self.calls+=1
                for polygon,text,score in output or []:
                    points=rotate_points(polygon,pw,ph,angle,inverse=True)
                    box=[(points[:,0].min()-pad)/sx+x1,(points[:,1].min()-pad)/sy+y1,
                         (points[:,0].max()-pad)/sx+x1,(points[:,1].max()-pad)/sy+y1]
                    margin=self.policy.artificial_edge_margin
                    complete=not ((x1>0 and box[0]<=x1+margin) or (y1>0 and box[1]<=y1+margin)
                       or (x2<w and box[2]>=x2-margin) or (y2<h and box[3]>=y2-margin))
                    box=[max(0.,box[0]),max(0.,box[1]),min(float(w),box[2]),min(float(h),box[3])]
                    if area(box)<=0:continue
                    readings.append({'text':text,'bbox':box,'score':float(score),'family':'strip',
                                     'complete':complete,'side':region['side'],'angle':angle})
        words=group_words(readings)
        crops=[];metadata=[];scale=scale_of(image.shape)
        for index,word in enumerate(words):
            if not any(r['complete'] for r in word['readings']):continue
            a,b,c,d=word['bbox'];p=self.policy.direct_padding*scale
            box=(max(0,math.floor(a-p)),max(0,math.floor(b-p)),min(w,math.ceil(c+p)),min(h,math.ceil(d+p)))
            crop=image[box[1]:box[3],box[0]:box[2]]
            if min(crop.shape[:2])<2:continue
            crop=cv2.resize(crop,None,fx=self.policy.magnification,fy=self.policy.magnification,interpolation=cv2.INTER_CUBIC)
            sides={r['side'] for r in word['readings'] if r['side']}
            angles=(0,180) if sides & {'left','right'} or not sides else (90,270)
            for angle in angles:
                crops.append(np.ascontiguousarray(np.rot90(crop,angle//90)));metadata.append((index,angle))
        # Recognition uses existing model; batch calls do not change ordering.
        for start in range(0,len(crops),64):
            output,_=self.backend.recognizer(crops[start:start+64]);self.direct_calls+=len(output)
            if len(output)!=len(metadata[start:start+64]):raise AssertionError('OCR batch length changed')
            for (index,angle),(text,score) in zip(metadata[start:start+64],output):
                words[index]['readings'].append({'text':text,'score':float(score),'bbox':words[index]['bbox'],
                    'family':'direct','complete':True,'side':None,'angle':angle})
        payload={'regions':regions,'words':words,'model_sha256':self.backend.model_signature,
                 'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'component':component.key}
        write_json(path,payload);return payload


def old_association(component,terms,words,original_roles):
    """B: feed complete words through exactly the existing V6/E6 algorithms."""
    import pcb.pin_semantics_v6 as ordered
    tokens=canonical_tokens(words)
    retained=[r.token for r in original_roles if not any(same_instance(r.token.bbox,t.bbox) for t in tokens)]
    roles=classify_tokens([*retained,*tokens],[component.body_bbox or component.bbox])
    roles,decisions=local_roles(component,terms,roles)
    previous=ordered._number_score
    try:
        ordered._number_score=number_bbox_score
        pins,events=ordered.assign_pin_semantics_v6(component,terms,roles)
    finally:ordered._number_score=previous
    return pins,events,decisions


def stage_b(baseline,raw,pool,roles):
    candidate=copy.deepcopy(baseline);lookup={r['component']:r['terminals'] for r in raw}
    old=event_lookup(raw,baseline.diagnostics);events=[];trace=[]
    for component in candidate.components:
        if component.type=='box':
            component.pins,rows,decisions=old_association(component,lookup[component.key],pool[component.key]['words'],roles)
            for old_pin,pin in zip(next(c for c in baseline.components if c.key==component.key).pins,component.pins):
                pin.tip,pin.base,pin.side=old_pin.tip,old_pin.base,old_pin.side
            events.extend({**previous,**r,'component':component.key}
                          for previous,r in zip(old[component.key],rows))
            trace.append({'component':component.key,'role_decisions':decisions,
                          'canonical_tokens':[asdict(t) for t in canonical_tokens(pool[component.key]['words'])]})
        else:events.extend(old[component.key])
    candidate.diagnostics['pin_events']=events
    candidate.diagnostics['L3_step']='3.1 complete words + unchanged V6/E6'
    assert_invariants(baseline,candidate,raw)
    return candidate,trace
