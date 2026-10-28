"""Image-only OCR recovery views, independently switchable for ablation.

No annotation, expected text, case identifier or scoring result is accepted by
candidate generation. Region/crop selection uses pixels and frontend outputs.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import cv2
import numpy as np

from .candidate_pool import merge_literal_candidates
from .tokens import OcrToken

METHODS=("multiscale","rotation","contrast","binarize","split","targeted")
DESCRIPTIONS={
    "multiscale":"512px 重叠分块、3倍放大，补小字与切块边界",
    "rotation":"90/270度旋转识别，再将坐标映回原图",
    "contrast":"CLAHE 对比度增强＋轻度锐化后重识别",
    "binarize":"自适应二值化后重识别低对比文字",
    "split":"对疑似粘连文字按像素间隙切分，识别真实子图",
    "targeted":"按前端缺失字段选择器件邻域，局部放大重识别",
}


def inverse_rotated_box(box,width,height,turns):
    """Map continuous xyxy boundaries after np.rot90 back to source pixels."""
    x1,y1,x2,y2=box
    turns%=4
    if turns==1:return (width-y2,x1,width-y1,x2)
    if turns==2:return (width-x2,height-y2,width-x1,height-y1)
    if turns==3:return (y1,height-x2,y2,height-x1)
    return tuple(box)


def tile_boxes(shape,side,overlap):
    def origins(length):
        if length<=side:return [0]
        values=list(range(0,length-side+1,side-overlap))
        if values[-1]!=length-side:values.append(length-side)
        return values
    height,width=shape[:2]
    return [(x,y,min(x+side,width),min(y+side,height)) for y in origins(height) for x in origins(width)]


def enhance(image,kind):
    if kind=="none":return image
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if kind=="contrast":
        gray=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
        gray=cv2.addWeighted(gray,1.5,cv2.GaussianBlur(gray,(0,0),.8),-.5,0)
    elif kind=="binarize":
        gray=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
    else:raise ValueError(kind)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)


def split_word_crops(image,tokens,max_crops=96):
    """Propose up to three pixel-gap cuts per suspicious OCR region.

    No candidate strings are fabricated from a target or digit split. Every
    returned half will be reread from its own source pixels by the recognizer.
    """
    height,width=image.shape[:2];rows=[];seen=set()
    for token in tokens:
        text=token.text.strip();box=token.bbox_xyxy
        h=max(1.,box[3]-box[1]);ratio=(box[2]-box[0])/h
        compact=bool(re.fullmatch(r"[RCLrcl]\d{2,}(?:[pnuµμmkKMΩFfHh]|ohm)+",text))
        if not (compact or (len(text.split())>=2 and ratio>=1.2) or (len(text)>=12 and ratio>=4)):
            continue
        x1,y1,x2,y2=(max(0,math.floor(box[0])),max(0,math.floor(box[1])),min(width,math.ceil(box[2])),min(height,math.ceil(box[3])))
        crop=image[y1:y2,x1:x2]
        if crop.size==0 or y2-y1<5:continue
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        blank=np.count_nonzero(ink,axis=0)<=max(0,int(.05*(y2-y1)))
        runs=[];start=None
        for x,is_blank in enumerate([*blank,False]):
            if is_blank and start is None:start=x
            elif not is_blank and start is not None:
                cut=(start+x)//2
                if .6*h<=cut<=(x2-x1)-.6*h:runs.append((x-start,cut))
                start=None
        for gap,cut in sorted(runs,reverse=True)[:3]:
            for left,right in ((x1,x1+cut),(x1+cut,x2)):
                candidate=(left,max(0,y1-1),right,min(height,y2+1))
                if candidate in seen:continue
                seen.add(candidate)
                rows.append((int(compact),gap,token.confidence,{"bbox":candidate,"source_text":text,"cut":x1+cut}))
    rows.sort(key=lambda row:row[:3],reverse=True)
    return [row[3] for row in rows[:max_crops]]


def method_views(method,image,tokens,before_filter,diagnostics):
    height,width=image.shape[:2]
    if method=="multiscale":
        return [{"bbox":b,"turns":0,"scale":3.,"enhance":"none","recognition_only":False} for b in tile_boxes(image.shape,512,128)]
    if method in {"rotation","contrast","binarize"}:
        turns=(1,3) if method=="rotation" else (0,)
        kind=method if method in {"contrast","binarize"} else "none"
        regions=[((0,0,width,height),min(1.,1536/max(width,height))),
                 *[(b,2.) for b in tile_boxes(image.shape,768,128)]]
        return [{"bbox":b,"turns":k,"scale":scale,"enhance":kind,"recognition_only":False} for b,scale in regions for k in turns]
    if method=="split":
        return [{**row,"turns":0,"scale":3.,"enhance":"none","recognition_only":True} for row in split_word_crops(image,tokens)]
    if method=="targeted":
        from pcb.component.local_joint import select_regions
        from pcb.schema import Component,Text
        original=SimpleNamespace(components=[Component(**c) for c in before_filter],
            texts=[Text(t.text,tuple(t.bbox_xyxy),t.confidence,t.source) for t in tokens],diagnostics=diagnostics)
        regions=select_regions(image.shape,original)
        return [{**row,"turns":0,"scale":min(3.,max(1.,40/row["font_height"]),1280/max(row["bbox"][2]-row["bbox"][0],row["bbox"][3]-row["bbox"][1])),
                 "enhance":"none","recognition_only":False} for row in regions]
    raise ValueError(method)


def generate_method_tokens(method,backend,image,tokens,before_filter,diagnostics,cache_root,progress=None):
    """Read deterministic image views, checkpointing each view independently."""
    if method not in METHODS:raise ValueError(method)
    height,width=image.shape[:2]
    views=method_views(method,image,tokens,before_filter,diagnostics)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    image_key=hashlib.sha256(image.tobytes()+backend.signature+source_hash.encode()).hexdigest()[:24]
    folder=Path(cache_root)/method/image_key;folder.mkdir(parents=True,exist_ok=True)
    raw=[];cache_hits=0
    for index,view in enumerate(views):
        encoded=json.dumps(view,sort_keys=True).encode()
        key=hashlib.sha256(encoded).hexdigest()[:20];path=folder/(key+".json")
        if path.is_file():
            payload=json.loads(path.read_text(encoding="utf8"));rows=[OcrToken(**t) for t in payload["tokens"]];cache_hits+=1
        else:
            left,top,right,bottom=view["bbox"]
            crop=image[top:bottom,left:right]
            native_h,native_w=crop.shape[:2]
            patch=enhance(crop,view["enhance"])
            patch=np.ascontiguousarray(np.rot90(patch,view["turns"]))
            scale=view["scale"]
            if scale!=1.:patch=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            rows=[]
            if view["recognition_only"]:
                patch=cv2.copyMakeBorder(patch,4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
                output=backend._ensure_engine()(patch,use_det=False,use_cls=False,use_rec=True)
                if output.txts is not None:
                    rows=[OcrToken(str(text),tuple(view["bbox"]),float(score),"server_"+method,method)
                          for text,score in zip(output.txts,output.scores) if str(text).strip() and float(score)>=.70]
            else:
                greedy,beam,_=backend._read_pair(patch,scale=scale)
                for token in [*greedy,*beam]:
                    x1,y1,x2,y2=inverse_rotated_box(token.bbox,native_w,native_h,view["turns"])
                    box=(max(0.,left+x1),max(0.,top+y1),min(float(width),left+x2),min(float(height),top+y2))
                    if box[0]<box[2] and box[1]<box[3]:rows.append(OcrToken(token.text,box,token.score,"server_"+method,method))
            payload={"view":view,"tokens":[asdict(t) for t in rows],"GT_used_in_inference":False}
            temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf8");temporary.replace(path)
        raw.extend(rows)
        if progress and (index==0 or (index+1)%20==0 or index==len(views)-1):progress(f"{method} views {index+1}/{len(views)}")
    return merge_literal_candidates(raw),{"method":method,"views":len(views),"view_cache_hits":cache_hits,"raw_readings":len(raw),"source_sha256":source_hash,"GT_used_in_inference":False}
