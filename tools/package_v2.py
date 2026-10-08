"""Create the reviewable V2 delivery ZIP without development caches or Dataset300."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'PCB_Competition_Solution_V2.zip'

def include(p):
    rel=p.relative_to(ROOT);parts=set(rel.parts)
    if {'runs','__pycache__','.git'} & parts:return False
    if any(x in parts for x in ('_baseline_debug','_v2_debug')):return False
    if rel.as_posix()=='reports/v2_delivery_manifest.json':return False
    return p.is_file() and p.suffix.lower() not in ('.pyc','.pyo')

files=sorted((p for p in ROOT.rglob('*') if include(p)),key=lambda p:p.relative_to(ROOT).as_posix())
manifest={'delivery':'PCB_Competition_Solution_V2','file_count_without_manifest':len(files),'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]}
manifest_path=ROOT/'reports/v2_delivery_manifest.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
files=files+[manifest_path]
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:z.write(p,(Path('PCB_Competition_Solution_V2')/p.relative_to(ROOT)).as_posix())
print(json.dumps({'zip':str(OUT),'files':len(files),'bytes':OUT.stat().st_size}))
