"""Record/verify old network outputs from frozen GT-free image-derived scenes."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from pcb.schema import Scene,Component,Pin,Text
from pcb.io import read_image
from pcb.core.context import PipelineContext
from pcb.core.team_config import load_team_config
from pcb.core.registry import build_default_registry
from pcb.submission import export


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda v:v.tolist()),encoding='utf8')
def plain(v):return json.loads(json.dumps(v,default=lambda x:x.tolist()))
def scene(data):
    rows=[]
    for c in data['components']:
        obj=Component(**{k:v for k,v in c.items() if k!='pins'});obj.pins=[Pin(**p) for p in c['pins']];rows.append(obj)
    return Scene(data['width'],data['height'],rows,[Text(**t) for t in data['texts']],diagnostics=copy.deepcopy(data['diagnostics']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('capture','verify'),required=True)
    p.add_argument('--scenes',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    done=read(args.scenes/'complete.json');inputs=read(args.scenes/'input_manifest.json')
    assert done['count']==35 and done['target_opens']==0
    wanted={14,17,18,87,103}|set(range(5,151,5))
    assert {int(r['case_id']) for r in inputs}==wanted
    source_rows={r['case_id']:r for r in done['records']}
    config=load_team_config(ROOT/'configs/team.json');registry=build_default_registry()
    wire=registry.create('wire',config.wire);topology=registry.create('topology',config.topology)
    rows=[];start=time.perf_counter()
    for item in inputs:
        cid=item['case_id'];record=source_rows[cid];source=args.scenes/cid/'scene.json'
        assert sha(source)==record['scene_sha256'] and sha(item['path'])==item['sha256']
        s=scene(read(source));immutable=plain([asdict(c) for c in s.components]);texts=plain([asdict(t) for t in s.texts])
        image=read_image(Path(item['path']));context=PipelineContext(Path(item['path']).name,s.width,s.height,config,None)
        w=wire.run(image,s,context);result=topology.run(s,w,context);pred=export(s)
        assert immutable==plain([asdict(c) for c in s.components]) and texts==plain([asdict(t) for t in s.texts])
        assert pred['components']==read(args.scenes/cid/'result.json')['components']
        assert pred['pins']==read(args.scenes/cid/'result.json')['pins']
        if args.phase=='capture':
            assert pred==read(args.scenes/cid/'result.json'), 'Old replay must reproduce full existing output'
            write(args.output/cid/'result.json',pred);write(args.output/cid/'diagnostics.json',s.diagnostics)
            np.save(args.output/cid/'skeleton.npy',result.skeleton)
        else:
            assert pred==read(args.baseline/cid/'result.json'), f'Prediction differs {cid}'
            assert plain(s.diagnostics)==read(args.baseline/cid/'diagnostics.json'), f'Diagnostics differ {cid}'
            assert np.array_equal(result.skeleton,np.load(args.baseline/cid/'skeleton.npy',allow_pickle=False)), f'Skeleton differs {cid}'
        row={'case_id':cid,'full_prediction_equal':True,'Component_Pin_unchanged':True,'image_sha256':item['sha256']}
        rows.append(row);write(args.output/'progress.json',{'records':rows})
        print(json.dumps({'event':'case_complete',**row}),flush=True)
    report={'phase':args.phase,'count':len(rows),'all_full_predictions_equal':True,'all_Component_Pin_unchanged':True,
            'all_diagnostics_and_skeleton_equal':args.phase=='verify','GT_reads':0,'new_frontend_or_Pin_inference':False,
            'seconds':time.perf_counter()-start,'config':config.as_dict(),'records':rows,'OFFICIAL_SCORE':False}
    write(args.output/'complete.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
