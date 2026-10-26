"""Color-adaptive wire extraction with explicit suppression diagnostics."""
import cv2
import numpy as np
from skimage.morphology import skeletonize

def _drop_page_frame(mask):
    """Remove page-border connected components without deleting interior wires."""
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8);h,w=mask.shape;out=mask.copy();removed=0
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i])
        touches=x<=2 or y<=2 or x+bw>=w-2 or y+bh>=h-2
        if touches and (bw>.55*w or bh>.55*h):out[lab==i]=0;removed+=area
    return out,removed

def _outward_hits(mask,scene,scale):
    """Count pins for which this color is visible just outside the symbol."""
    hits=0;h,w=mask.shape;reach=max(5,int(round(10*scale)))
    for c in scene.components:
        cx=(c.bbox[0]+c.bbox[2])/2;cy=(c.bbox[1]+c.bbox[3])/2
        for p in c.pins:
            v=np.array(p.tip)-np.array(p.base if p.base is not None else (cx,cy));norm=float(np.linalg.norm(v))
            if norm<.5:continue
            v=v/norm;found=False
            for d in range(1,reach+1):
                x=int(round(p.tip[0]+v[0]*d));y=int(round(p.tip[1]+v[1]*d))
                if 0<=x<w and 0<=y<h and mask[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)].any():found=True;break
            hits+=int(found)
    return hits

def _line_supported(mask,scale):
    k=max(5,int(round(11*scale)));k+=1-k%2
    h=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,k),np.uint8))
    v=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((k,1),np.uint8));seeds=cv2.bitwise_or(h,v)
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8);keep=np.zeros_like(mask)
    # Vectorized equivalent of repeatedly evaluating ``seeds[lab == i]``.
    # Dense datasheet pages can contain thousands of components; the old loop
    # rescanned the complete image once per component and dominated runtime.
    keep_ids=set(map(int,np.unique(lab[seeds>0]))) if seeds.any() else set()
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA]);x,y,w,hg=map(int,stats[i,:4])
        if area<3:continue
        if max(w,hg)>=k*2 and min(w,hg)<=max(5,k//2):keep_ids.add(i)
    keep_ids.discard(0)
    if keep_ids:keep[np.isin(lab,np.fromiter(keep_ids,dtype=np.int32))]=255
    supported=sum(int(stats[i,cv2.CC_STAT_AREA]) for i in keep_ids)
    return keep,seeds,supported

def _stroke(mask):
    if not mask.any():return 1.0
    sk=skeletonize(mask>0);dist=cv2.distanceTransform(mask,cv2.DIST_L2,3);vals=dist[sk]
    return float(max(1,np.median(vals)*2)) if vals.size else 1.0

def _select_palette_candidates(scored, *, minimum, score_ratio=.72, max_candidates=2):
    """Select supported palettes, optionally without a candidate-count cap.

    ``max_candidates=2`` preserves the historical V2 behavior. ``None`` keeps
    every palette that passes the existing minimum-density and relative-score
    gates. This only combines image-derived, line-supported masks; it does not
    infer electrical connectivity.
    """
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive or None")
    if not 0.0 <= score_ratio <= 1.0:
        raise ValueError("score_ratio must be between 0 and 1")
    selected=[]
    for item in scored:
        if item[3] < minimum:
            continue
        if selected and item[0] < selected[0][0] * score_ratio:
            continue
        selected.append(item)
        if max_candidates is not None and len(selected) >= max_candidates:
            break
    return selected


def extract_wire_v2(image,scene,*,max_palette_candidates=2,palette_score_ratio=.72,_trace=None,
                    _candidate_callback=None):
    h,w=image.shape[:2];scale=max(.6,min(2.,max(h,w)/1500));hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sat=hsv[:,:,1];val=hsv[:,:,2];candidates=[]
    # Dark/monochrome candidate plus overlapping hue sectors. Overlap retains
    # antialiased strokes that sit on a hue boundary.
    candidates.append(('dark',((gray<205)&(sat<90)).astype(np.uint8)*255))
    colorful=(sat>=45)&(val<248)
    for i in range(12):
        lo=i*15;hi=(i+2)*15
        sel=((hsv[:,:,0]>=lo)&(hsv[:,:,0]<hi)) if hi<=180 else ((hsv[:,:,0]>=lo)|(hsv[:,:,0]<(hi-180)))
        candidates.append((f'hue_{lo}_{hi%180}',(colorful&sel).astype(np.uint8)*255))
    suppression=np.zeros((h,w),np.uint8)
    for t in scene.texts:
        x1,y1,x2,y2=map(int,t.bbox);pad=max(1,int(round(scale)))
        cv2.rectangle(suppression,(max(0,x1-pad),max(0,y1-pad)),(min(w-1,x2+pad),min(h-1,y2+pad)),255,-1)
    for c in scene.components:
        x1,y1,x2,y2=map(int,c.body_bbox or c.bbox);margin=max(1,int(round(1.5*scale)))
        cv2.rectangle(suppression,(max(0,x1+margin),max(0,y1+margin)),(min(w-1,x2-margin),min(h-1,y2-margin)),255,-1)
    scored=[];debug=np.zeros_like(image);color_fraction=float((sat>=45).mean())
    raw_union=np.zeros((h,w),np.uint8);clean_union=np.zeros((h,w),np.uint8);supported_union=np.zeros((h,w),np.uint8)
    supported_candidates={}
    palette=[(0,150,0),(180,0,0),(0,0,190),(170,90,0),(120,0,170),(0,140,180)]
    for name,raw in candidates:
        if _trace is not None:raw_union=cv2.bitwise_or(raw_union,raw)
        clean=raw.copy();clean[suppression>0]=0;clean,frame_removed=_drop_page_frame(clean)
        if _trace is not None:clean_union=cv2.bitwise_or(clean_union,clean)
        keep,seeds,support=_line_supported(clean,scale)
        if _trace is not None:
            supported_union=cv2.bitwise_or(supported_union,keep)
            supported_candidates[name]=keep.copy()
        if _candidate_callback is not None:
            _candidate_callback(name, keep)
        density=int((keep>0).sum());pin_hits=_outward_hits(keep,scene,scale)
        score=pin_hits*12.0+support/max(1,density)*np.sqrt(max(0,density))
        if name=='dark' and color_fraction>.02:score*=.35
        scored.append((score,name,keep,density,int((seeds>0).sum()),pin_hits,frame_removed))
    scored.sort(reverse=True,key=lambda x:x[0])
    minimum=max(20,h*w*.000015)
    eligible=_select_palette_candidates(
        scored, minimum=minimum, score_ratio=palette_score_ratio,
        max_candidates=None) if _trace is not None else []
    selected=_select_palette_candidates(
        scored, minimum=minimum, score_ratio=palette_score_ratio,
        max_candidates=max_palette_candidates)
    eligible_union=np.zeros((h,w),np.uint8)
    if _trace is not None:
        for item in eligible:eligible_union=cv2.bitwise_or(eligible_union,item[2])
    mask=np.zeros((h,w),np.uint8)
    for j,(_,name,m,_,_,_,_) in enumerate(selected):
        mask=cv2.bitwise_or(mask,m);debug[m>0]=palette[j%len(palette)]
    # Close only sub-stroke gaps; long bridge decisions belong in topology.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    mask[:2,:]=0;mask[-2:,:]=0;mask[:,:2]=0;mask[:,-2:]=0
    if _trace is not None:
        _trace.update({
            "raw_palette_union":raw_union,
            "after_suppression_and_frame_union":clean_union,
            "after_line_support_union":supported_union,
            "line_supported_candidates":supported_candidates,
            "after_score_gate_union":eligible_union,
            "after_candidate_limit_union":np.bitwise_or.reduce(
                np.stack([item[2] for item in selected]),axis=0
            ) if selected else np.zeros((h,w),np.uint8),
            "after_close_and_border":mask.copy(),
            "candidate_scores":[{"name":item[1],"score":float(item[0]),"pixels":int(item[3]),
                                 "selected_by_gate":item[1] in {entry[1] for entry in eligible},
                                 "selected_by_limit":item[1] in {entry[1] for entry in selected}}
                                for item in scored],
            "score_ratio":float(palette_score_ratio),
            "max_palette_candidates":max_palette_candidates,
            "minimum_density":float(minimum),
        })
    stroke=_stroke(mask)
    scene.diagnostics.update({'wire_extractor':'adaptive_v2','wire_palette_candidates':[{'name':n,'score':round(float(s),2),'pixels':d,'seed_pixels':sp,'outward_pin_hits':ph,'frame_pixels_removed':fr} for s,n,_,d,sp,ph,fr in scored],
        'wire_palette_selected':[x[1] for x in selected],'wire_stroke_width':round(stroke,3),'suppressed_pixels':int((suppression>0).sum())})
    return mask,debug,suppression
