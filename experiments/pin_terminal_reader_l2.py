"""Image-only terminal-conditioned OCR and joint box-pin text ownership.

The module has no dataset, annotation, case-id, model-training or export code.
All terminals remain unchanged. Local glyph recognition bypasses text detection
only for actual connected-ink crops; it never supplies a guessed reading order.
"""
from __future__ import annotations

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
from pcb.ocr_backends import _bbox_iou, deduplicate_texts
from pcb.pin.local_ocr import rotate_points
from pcb.schema import Pin, Scene, Text
from pcb.text_detection import PIN_NUMBER_RE, SIGNAL_RE


@dataclass(frozen=True)
class ReaderPolicy:
    magnification: float = 3.0
    padding: int = 12
    confidence: float = .80
    max_normal_extent: float = 120.0
    min_normal_extent: float = 30.0
    min_row_half_height: float = 6.0
    max_row_half_height: float = 24.0
    number_inside_fraction: float = .20
    name_inside_fraction: float = .49
    number_inside_cap: float = 40.0
    direct_number_min_confidence: float = .90
    det_max_side: int = 768
    conflict_margin: float = .12
    cpu_threads: int = 4


POLICY = ReaderPolicy()
SIDES = ('left', 'right', 'top', 'bottom')


def scale_of(image_shape: tuple) -> float:
    return max(.55, min(2.2, max(image_shape[:2]) / 1200.0))


def tangent(point, side: str) -> float:
    return float(point[1] if side in ('left', 'right') else point[0])


def normal_inside(point, side: str, box) -> float:
    """Positive toward the body interior; original top-left image pixels."""
    x, y = point
    a, b, c, d = box
    return {'left': x-a, 'right': c-x, 'top': y-b, 'bottom': d-y}[side]


def row_windows(component, terminals: list[dict], image_shape: tuple,
                policy: ReaderPolicy = POLICY) -> list[dict[str, Any]]:
    """Same-side midpoint cells prevent a token from owning two terminal rows."""
    box = component.body_bbox or component.bbox
    scale = scale_of(image_shape)
    h, w = image_shape[:2]
    rows = []
    for side in SIDES:
        ordered = sorted([(i, t) for i, t in enumerate(terminals) if t['side'] == side],
                         key=lambda r: tangent(r[1]['base'], side))
        values = [tangent(t['base'], side) for _, t in ordered]
        span = box[2]-box[0] if side in ('left', 'right') else box[3]-box[1]
        outside = max(policy.min_normal_extent*scale,
                      min(policy.max_normal_extent*scale, .25*span))
        inside = min(policy.max_normal_extent*scale, policy.name_inside_fraction*span)
        number_inside = min(policy.number_inside_cap*scale, policy.number_inside_fraction*span)
        for rank, (index, term) in enumerate(ordered):
            v = values[rank]
            gaps = [abs(v-other) for other in values if abs(v-other)>1e-7]
            half = max(policy.min_row_half_height*scale,
                       min(policy.max_row_half_height*scale, .48*min(gaps, default=40*scale)))
            lo = max(v-half, (v+values[rank-1])/2 if rank else v-half)
            hi = min(v+half, (v+values[rank+1])/2 if rank+1<len(values) else v+half)
            # At image borders a crop may shrink; no candidate is removed.
            def make_region(outer, inner):
                a,b,c,d = box
                if side == 'left': region=(a-outer,lo,a+inner,hi)
                elif side == 'right': region=(c-inner,lo,c+outer,hi)
                elif side == 'top': region=(lo,b-outer,hi,b+inner)
                else: region=(lo,d-inner,hi,d+outer)
                x1,y1,x2,y2=region
                return (max(0,math.floor(x1)),max(0,math.floor(y1)),
                        min(w,math.ceil(x2)),min(h,math.ceil(y2)))
            rows.append({'index':index,'side':side,'rank':rank,'tangent':v,
                         'lo':lo,'hi':hi,'outside':outside,'inside':inside,
                         'number_inside':number_inside,'roi':make_region(outside,inside),
                         'number_roi':make_region(outside,number_inside)})
    return sorted(rows, key=lambda r:r['index'])


def isolated_glyph_boxes(crop: np.ndarray, scale: float) -> list[tuple]:
    """Bound short text ink, excluding long horizontal wires before direct OCR.

    No output from this helper is a terminal or a reading. A glyph may be a
    non-text artifact; downstream recognition/conflict/pair gates must pass.
    """
    if crop.size == 0 or min(crop.shape[:2])<3:
        return []
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    long=cv2.morphologyEx(ink,cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT,(max(16,round(18*scale)),1)))
    cleaned=cv2.bitwise_and(ink,cv2.bitwise_not(long))
    # Join character strokes and adjacent digits without joining complete rows.
    grouped=cv2.dilate(cleaned,cv2.getStructuringElement(cv2.MORPH_RECT,(max(2,round(3*scale)),1)))
    count,_,stats,_=cv2.connectedComponentsWithStats(grouped,8)
    boxes=[]
    for x,y,width,height,area in stats[1:count]:
        if height<max(3,round(3*scale)) or area<5 or width>max(48,60*scale):
            continue
        if height>max(32*scale,crop.shape[0]*.90) or width/height>6:
            continue
        pad=max(1,round(scale))
        boxes.append((max(0,int(x)-pad),max(0,int(y)-pad),
                      min(crop.shape[1],int(x+width)+pad),min(crop.shape[0],int(y+height)+pad)))
    return sorted(boxes,key=lambda b:(b[1],b[0]))


class TerminalRowOCR:
    """Use installed RapidOCR models; save all local readings including rejects."""
    def __init__(self, cache_dir: Path, policy: ReaderPolicy = POLICY, engine=None):
        self.cache=Path(cache_dir)
        self.policy=policy
        self.engine=engine
        self.recognizer=None
        self.calls=self.direct_calls=self.cache_hits=0
        self.model_signature=None

    def ensure_engine(self):
        if self.engine is not None:
            if self.recognizer is None:
                self.recognizer=getattr(self.engine,'text_rec',None) or getattr(self.engine,'text_recognizer',None)
            return self.engine
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR
        import onnxruntime as ort
        import yaml
        package=Path(rapidocr_onnxruntime.__file__).parent
        config=yaml.safe_load((package/'config.yaml').read_text(encoding='utf-8'))
        paths={k:package/config[k.title()]['model_path'] for k in ('det','rec')}
        self.model_signature={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
        self.engine=RapidOCR(use_cls=False,use_angle_cls=False,use_det=True,use_text_det=True,min_height=0,
            width_height_ratio=-1,det_model_path=str(paths['det']),
            det_limit_type='max',det_limit_side_len=self.policy.det_max_side,
            rec_model_path=str(paths['rec']),intra_op_num_threads=self.policy.cpu_threads,
            inter_op_num_threads=1)
        # Only this private experiment instance: identical ONNX weights/backend,
        # bounded CPU threads rather than all host cores. No installed file edits.
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=self.policy.cpu_threads
        opts.inter_op_num_threads=1
        opts.log_severity_level=4
        if hasattr(self.engine,'text_rec'):
            self.recognizer=self.engine.text_rec
        else:
            self.recognizer=self.engine.text_recognizer
            for adapter,key in ((self.engine.text_detector.infer,'det'),
                                (self.recognizer.session,'rec')):
                adapter.session=ort.InferenceSession(str(paths[key]),sess_options=opts,
                                                     providers=['CPUExecutionProvider'])
        return self.engine

    def read(self, image: np.ndarray, row: dict) -> list[Text]:
        engine=self.ensure_engine()
        roi=row['roi']; x1,y1,x2,y2=roi
        crop=image[y1:y2,x1:x2]
        if crop.size==0: return []
        signature={'policy':asdict(self.policy),'model':self.model_signature,
                   'side':row['side'],'number_roi_relative':[row['number_roi'][i]-roi[i%2] for i in range(4)]}
        key=hashlib.sha256(crop.tobytes()+json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
        path=self.cache/(key+'.json')
        if path.exists():
            local=json.loads(path.read_text(encoding='utf-8')); self.cache_hits+=1
        else:
            local=[]
            factor=self.policy.magnification; pad=self.policy.padding
            large=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
            sy,sx=large.shape[0]/crop.shape[0],large.shape[1]/crop.shape[1]
            large=cv2.copyMakeBorder(large,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255))
            ph,pw=large.shape[:2]
            angles=(0,180) if row['side'] in ('left','right') else (90,270)
            for angle in angles:
                output,_=engine(np.ascontiguousarray(np.rot90(large,angle//90)))
                self.calls+=1
                for polygon,reading,confidence in output or []:
                    points=rotate_points(polygon,pw,ph,angle,inverse=True)
                    points[:,0]=(points[:,0]-pad)/sx
                    points[:,1]=(points[:,1]-pad)/sy
                    a,b=np.min(points,axis=0); c,d=np.max(points,axis=0)
                    a,b=max(0,float(a)),max(0,float(b))
                    c,d=min(float(crop.shape[1]),float(c)),min(float(crop.shape[0]),float(d))
                    if c>a and d>b:
                        local.append({'text':str(reading),'score':float(confidence),
                            'bbox':[a,b,c,d],'source':f'row_det:{angle}'})
            nx1,ny1,nx2,ny2=row['number_roi']
            small=image[ny1:ny2,nx1:nx2]
            # Direct recognizer sees real isolated ink. Never use an allowlist
            # to force an arbitrary stroke/model name to become a number.
            if row['side'] in ('top','bottom'):
                small=np.ascontiguousarray(np.rot90(small))
            boxes=isolated_glyph_boxes(small,scale_of(image.shape))
            crops=[]; spec=[]
            for box in boxes:
                a,b,c,d=box; glyph=small[b:d,a:c]
                for angle in (0,180):
                    rotated=np.ascontiguousarray(np.rot90(glyph,angle//90))
                    enlarged=cv2.resize(rotated,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
                    crops.append(cv2.copyMakeBorder(enlarged,6,6,6,6,cv2.BORDER_CONSTANT,value=(255,255,255)))
                    spec.append((box,angle))
            if crops:
                readings,_=self.recognizer(crops); self.direct_calls+=len(crops)
                for (box,angle),(reading,confidence) in zip(spec,readings):
                    normalized=re.sub(r'\s+','',str(reading))
                    if not PIN_NUMBER_RE.fullmatch(normalized): continue
                    a,b,c,d=box
                    points=np.array([[a,b],[c,d]],dtype=float)
                    if row['side'] in ('top','bottom'):
                        points=rotate_points(points,nx2-nx1,ny2-ny1,90,inverse=True)
                    a,b=np.min(points,axis=0); c,d=np.max(points,axis=0)
                    local.append({'text':normalized,'score':float(confidence),
                        'bbox':[float(a+nx1-x1),float(b+ny1-y1),float(c+nx1-x1),float(d+ny1-y1)],
                        'source':f'row_direct:{angle}'})
            write_json(path,local)
        return [Text(r['text'],tuple(r['bbox'][i]+(x1 if i%2==0 else y1) for i in range(4)),
                     r['score'],r['source']) for r in local]


def token_candidates(tokens: list[Text], row: dict, component, policy=POLICY) -> tuple[list,list]:
    """Interpret raw text by local ownership, not global VALUE/PIN_NUMBER role."""
    box=component.body_bbox or component.bbox
    identity={re.sub(r'\s+','',str(t)).upper() for t in
              (component.key,component.name,component.model,component.value,component.observable_designator) if t}
    numbers=[]; names=[]
    a,b,c,d=box
    for token in deduplicate_texts(tokens):
        raw=token.text.strip(); text=re.sub(r'\s+','',raw)
        if not text or token.score<policy.confidence or text.upper() in identity: continue
        u=tangent(token.center,row['side']); inside=normal_inside(token.center,row['side'],box)
        if not (row['lo']<=u<row['hi']): continue
        if inside < -row['outside'] or inside > row['inside']: continue
        # Reject rotated alternatives that disagree at comparable confidence.
        conflict=any(_bbox_iou(token.bbox,other.bbox)>=.5 and
                     re.sub(r'\s+','',other.text).upper()!=text.upper() and
                     other.score>=token.score-policy.conflict_margin
                     for other in tokens if other is not token)
        if conflict: continue
        width=max(1,token.bbox[2]-token.bbox[0]); height=max(1,token.bbox[3]-token.bbox[1])
        cross=height if row['side'] in ('left','right') else width
        alignment=abs(u-row['tangent'])/max(cross, (row['hi']-row['lo'])/2,1)
        base_score=2+token.score-max(0,alignment)
        direct=token.source_id.startswith('row_direct:')
        if PIN_NUMBER_RE.fullmatch(text) and inside<=row['number_inside']:
            if not direct or token.score>=policy.direct_number_min_confidence:
                numbers.append((base_score+(0.25 if inside<=2 else 0),token,text))
        name_pattern=SIGNAL_RE.fullmatch(text) or re.fullmatch(r'[A-Za-z]',text)
        center_inside=(a-2<=token.center[0]<=c+2 and b-2<=token.center[1]<=d+2)
        if name_pattern and center_inside and inside>=-2 and not direct:
            names.append((base_score,token,raw))
    return numbers,names


def joint_row_choice(tokens, row, component, policy=POLICY):
    """Choose a complete number/name pair in the same terminal-owned row."""
    numbers,names=token_candidates(tokens,row,component,policy)
    options=[]
    for ns,number,ntext in numbers:
        for ss,name,stext in names:
            if number is name or _bbox_iou(number.bbox,name.bbox)>.5: continue
            distance=abs(tangent(number.center,row['side'])-tangent(name.center,row['side']))
            cross=max(1,row['hi']-row['lo'])
            score=ns+ss+1-distance/cross
            options.append({'number':ntext,'name':stext,'number_token':number,
                            'name_token':name,'score':float(score)})
    if not options: return None, {'numbers':len(numbers),'names':len(names),'complete_pairs':0}
    result=max(options,key=lambda r:(r['score'],r['number_token'].score+r['name_token'].score))
    return result, {'numbers':len(numbers),'names':len(names),'complete_pairs':len(options)}


def refine_scene(image: np.ndarray, scene: Scene, raw: list[dict], reader: TerminalRowOCR):
    """Box semantics only; complete joint evidence, duplicate-key-safe fallback."""
    from pin_skeleton_followup import event_lookup
    old=event_lookup(raw,scene.diagnostics)
    result=copy.deepcopy(scene)
    events=[]; trace=[]; statistics={'owners':0,'rows':0,'complete_pairs':0,'accepted':0}
    for component,block in zip(result.components,raw):
        before=copy.deepcopy(component.pins); previous=old[component.key]
        if component.type!='box' or component.key.startswith('UNRESOLVED_'):
            events.extend(copy.deepcopy(previous)); continue
        statistics['owners']+=1
        choices={}; owner_trace=[]
        windows=row_windows(component,block['terminals'],image.shape,reader.policy)
        for row in windows:
            local=reader.read(image,row)
            all_tokens=[*scene.texts,*local]
            choice,details=joint_row_choice(all_tokens,row,component,reader.policy)
            statistics['rows']+=1; statistics['complete_pairs']+=bool(choice)
            entry={'component':component.key,'candidate':row['index'],'window':row,
                   'local_tokens':[asdict(t) for t in local], 'candidate_counts':details,
                   'selected':None,'accepted':False}
            if choice:
                choices[row['index']]=choice
                entry['selected']={**choice,'number_token':asdict(choice['number_token']),
                                    'name_token':asdict(choice['name_token'])}
            owner_trace.append(entry)
        # Do not overwrite another terminal's existing exported reference.
        # Reserve existing numbers first; a new pair may correct a name, or
        # fill a missing number, but never silently collide with another pin.
        used={p.number:i for i,p in enumerate(before) if p.exportable}
        best={}
        for index,choice in choices.items():
            number=choice['number']
            if number in used and used[number]!=index: continue
            if number not in best or choice['score']>choices[best[number]]['score']: best[number]=index
        for index,pin in enumerate(component.pins):
            choice=choices.get(index); event=copy.deepcopy(previous[index])
            accepted=bool(choice and best.get(choice['number'])==index)
            if accepted:
                number=choice['number']; name=choice['name']
                pin.number=number; pin.name=name; pin.exportable=True; pin.inferred_number=False
                pin.observable_number=number; pin.number_source='terminal_row_joint_l2'
                pin.number_confidence=choice['number_token'].score
                pin.name_confidence=choice['name_token'].score
                event.update(number=number,pinname=name,exportable=True,semantic_method='terminal_row_joint_l2',
                    number_token=choice['number_token'].text,name_token=choice['name_token'].text,
                    number_confidence=pin.number_confidence,name_confidence=pin.name_confidence,
                    joint_assignment_score=choice['score'],number_source=choice['number_token'].source_id,
                    name_source=choice['name_token'].source_id)
                statistics['accepted']+=1
            event['L2_attempted']=True; event['L2_accepted']=accepted
            events.append(event)
            owner_trace[index]['accepted']=accepted
        for p,q,t in zip(before,component.pins,block['terminals']):
            if any(getattr(p,f)!=getattr(q,f) or tuple(getattr(q,f))!=tuple(t[f]) for f in ('tip','base','side')):
                raise AssertionError('L2 modified frozen terminal geometry')
        if len({p.number for p in component.pins if p.exportable})!=sum(p.exportable for p in component.pins):
            raise AssertionError('L2 generated duplicate exported keys')
        trace.extend(owner_trace)
    for a,b in zip(scene.components,result.components):
        before,after=asdict(a),asdict(b)
        bp,ap=before.pop('pins'),after.pop('pins')
        if before!=after or len(bp)!=len(ap): raise AssertionError('L2 changed component/candidate count')
        if a.type!='box' and bp!=ap: raise AssertionError('L2 changed nonbox pins')
    result.diagnostics['pin_events']=events
    result.diagnostics['terminal_reader_l2']=statistics
    result.diagnostics['exportable_pins']=sum(p.exportable for c in result.components for p in c.pins)
    event_lookup(raw,result.diagnostics)
    return result,trace,statistics
