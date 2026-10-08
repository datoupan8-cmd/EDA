from pathlib import Path
import json, re
import cv2
import numpy as np

def natural(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(s))]

def read_image(path):
    image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f'Cannot decode image: {path}')
    return image

def write_image(path,image):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode(p.suffix,image)
    if not ok:raise ValueError(str(p))
    data.tofile(str(p))

def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    def encode(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=encode),encoding='utf-8')

def discover_images(root):
    """Only original case images; do not accidentally infer on the user's overlays."""
    root=Path(root)
    candidates=sorted((p for p in root.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg')),key=lambda p:natural(p.as_posix()))
    filtered=[p for p in candidates if p.stem.lower() not in ('overlay','wire_mask','skeleton') and not any(t in p.stem.lower() for t in ('_result','_labels','_overlay','_mask','_debug'))]
    return filtered

def case_names(images):
    """Preserve Dataset300 folder IDs while avoiding flat-directory output collisions."""
    from collections import Counter
    counts=Counter(p.parent for p in images);used=set();result=[]
    for p in images:
        name=p.parent.name if counts[p.parent]==1 else p.stem
        base=name;i=2
        while name in used:name=f'{base}_{i}';i+=1
        used.add(name);result.append(name)
    return result

def annotation_for(image):
    p=Path(image)
    q=p.with_name(p.stem+'_CVJsonStd.json')
    if not q.exists():raise FileNotFoundError(f'Annotation-assisted mode requires explicit matching file: {q}')
    return q
