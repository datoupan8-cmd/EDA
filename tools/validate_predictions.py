"""Strictly validate a numbered 0001..0150 prediction batch without reading targets."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUNTIME=ROOT.parent/'work/runtime'
if RUNTIME.exists():sys.path.insert(0,str(RUNTIME))

from pcb.io import read_image,write_json
from pcb.submission import validate_strict

HOLDOUT_START=151

def validate_batch(prediction_dir:Path,image_root:Path,start:int,end:int):
    if not (1<=start<=end<HOLDOUT_START):
        raise ValueError('Holdout guard: validator accepts only 0001..0150')
    if image_root.name!='200_train_cases':
        raise ValueError('image_root must be exactly 200_train_cases')
    rows=[]
    for number in range(start,end+1):
        case=f'{number:04d}'
        try:
            images=sorted((image_root/case).glob('*.png'))
            if len(images)!=1:raise FileNotFoundError(f'expected one PNG, got {len(images)}')
            image=read_image(images[0])
            path=prediction_dir/case/'result.json'
            data=json.loads(path.read_text(encoding='utf-8'))
            validate_strict(data,(image.shape[1],image.shape[0]))
            rows.append({'case_id':case,'status':'valid'})
        except Exception as exc:
            rows.append({'case_id':case,'status':'invalid','error':repr(exc)})
    failures=[r for r in rows if r['status']!='valid']
    return {'case_start':start,'case_end':end,'holdout_used':False,
            'validated_count':len(rows)-len(failures),'failure_count':len(failures),'cases':rows}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prediction_dir',type=Path,required=True)
    ap.add_argument('--image_root',type=Path,required=True)
    ap.add_argument('--case_start',type=int,default=1)
    ap.add_argument('--case_end',type=int,default=150)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=validate_batch(args.prediction_dir,args.image_root,args.case_start,args.case_end)
    write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('validated_count','failure_count','holdout_used')},ensure_ascii=False))
    return int(result['failure_count']>0)

if __name__=='__main__':raise SystemExit(main())
