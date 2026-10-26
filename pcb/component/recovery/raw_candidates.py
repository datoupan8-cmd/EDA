from dataclasses import asdict
import hashlib,json
from .tokens import OcrToken
from .candidate_pool import merge_literal_candidates

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf8")

def raw_server_candidates(backend,image,cache_path):
    recipe=b"literal-all-views-v1" if max(image.shape[:2])<=1536 else b"literal-max1536-tiles768x2-v2"
    key=hashlib.sha256(image.tobytes()+backend.signature+recipe).hexdigest()
    if cache_path.is_file():
        saved=json.loads(cache_path.read_text(encoding="utf8"))
        if saved["key"]==key:return [OcrToken(**t) for t in saved["tokens"]]
    import cv2
    from pcb.component.detector_yolo import tile_origins
    raw=[]
    def read(patch,left=0,top=0,scale=1.):
        greedy,beam,_=backend._read_pair(patch,left=left,top=top,scale=scale)
        raw.extend(OcrToken(t.text,tuple(t.bbox),t.score,"ppocrv5_server","server_raw") for t in [*greedy,*beam])
    height,width=image.shape[:2]
    # Bound server detector memory; tiles still read every source pixel at 2x.
    scale=min(1.,1536/max(height,width))
    read(image if scale==1. else cv2.resize(image,None,fx=scale,fy=scale),scale=scale)
    for top in tile_origins(height,768,128):
        for left in tile_origins(width,768,128):
            patch=cv2.resize(image[top:top+768,left:left+768],None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            read(patch,left,top,2.)
    tokens=merge_literal_candidates(raw)
    write(cache_path,{"key":key,"tokens":[asdict(t) for t in tokens],"raw_count":len(raw),"GT_used_in_inference":False})
    return tokens
