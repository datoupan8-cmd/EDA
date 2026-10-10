"""Supervised training only on permitted train cases; no validation GT reads."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'),str(ROOT)]

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pin_learned_locator_l1 import (POLICY, PinPointNet, strip_specs, sample_strip,
    global_to_strip, nearest_side, point_loss)
from pcb.coordinates import target_to_opencv, target_bbox_to_opencv
from pcb.data_policy import discover_allowed_cases, split_for_case, assert_allowed_case_id
from pcb.io import read_image

OUT = ROOT/'reports/pin_learned_locator_l1'
DATASET = Path('C:/Users/bluty/Desktop/PCB_Competition/赛题六公开数据集/200_train_cases')


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


class Guard:
    """Targets allowed only in an explicit training/evaluator context."""
    def __init__(self):
        self.target_case = None
        self.blocked = 0
        sys.addaudithook(self.audit)

    def audit(self,event,args):
        if event != 'open' or not args or not isinstance(args[0],(str,bytes,Path)): return
        parts = str(args[0]).replace('\\','/').lower().split('/')
        if any(p in {'10_gtcase','10gt','golden','eda_pin_crossing_quicktest'} for p in parts):
            self.blocked += 1; raise PermissionError('Sealed/QuickTest blocked')
        if '200_train_cases' in parts:
            i = parts.index('200_train_cases')
            if i+1 >= len(parts): return
            case = int(parts[i+1]); assert_allowed_case_id(case)
            if '_target' in parts[-1] and parts[-1].endswith('.json') and self.target_case != case:
                self.blocked += 1; raise PermissionError('GT read outside explicit supervision/evaluation')

    def target(self,case,training=True):
        assert_allowed_case_id(case.case_id)
        if training and split_for_case(case.case_id) != 'train':
            raise PermissionError('Validation target prohibited in training')
        self.target_case = case.case_id
        try:
            raw = json.loads(case.target_path.read_text(encoding='utf-8'))
            sha = digest(case.target_path)
            return raw, sha
        finally: self.target_case = None


def freeze():
    protected = ('pcb/core/registry.py','configs/current.json','pcb/pin_detection.py',
        'pcb/pin/localization/v3.py','pcb/schema.py','pcb/submission.py','evaluate_v2.py',
        'models/component_yolo11n_continue_v2_best.pt','experiments/pin_tip_evidence.py',
        'experiments/pin_contextual_roles_e6.py','experiments/pin_g_frame_e3.py',
        'experiments/pin_skeleton_followup.py','experiments/pin_tip_regression.py',
        'pcb/pin_semantics_v5.py','pcb/pin_semantics_v6.py',
        'pcb/wire_v3.py','pcb/topology_v2.py','pcb/coordinates.py')
    files = ('experiments/pin_learned_locator_l1.py','experiments/pin_learned_locator_l1_train.py',
        'experiments/pin_learned_locator_l1_eval.py','tests/pin/test_pin_learned_locator_l1.py',
        'reports/pin_learned_locator_l1/experiment_plan.md')
    value = {'policy':asdict(POLICY),'source_sha256':{f:digest(ROOT/f) for f in files},
        'protected_sha256':{f:digest(ROOT/f) for f in protected},
        'split':'existing case_id%5==0 validation; train elsewhere',
        'model_selection':'minimum training loss, no validation labels',
        'single_factor':'box point localization', 'OFFICIAL_SCORE':False,
        'sealed_holdout_used':False}
    path = OUT/'frozen_policy.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise AssertionError('Frozen L1 protocol/source changed; do not tune on validation')
    if not path.exists(): write(path,value)
    return value


def prepare():
    freeze(); guard = Guard()
    cases = discover_allowed_cases(DATASET)
    images = {c.case_id:digest(c.image_path) for c in cases}
    val = [c.case_id for c in cases if c.split == 'dev']
    val_hashes = {images[i] for i in val}
    excluded = [c.case_id for c in cases if c.split == 'train' and images[c.case_id] in val_hashes]
    train = [c for c in cases if c.split == 'train' and c.case_id not in excluded]
    manifest = {'dataset_root':str(DATASET),'train_ids':[c.case_id for c in train],
        'validation_ids':val,'excluded_train_duplicate_validation':excluded,
        'image_sha256':images,'records':[],'OFFICIAL_SCORE':False,'sealed_holdout_used':False}
    seen = {}; totals = Counter(); patch = OUT/'training_patches'; patch.mkdir(parents=True,exist_ok=True)
    for case in train:
        raw, sha = guard.target(case)
        # Same image with conflicting targets cannot silently become mixed supervision.
        if images[case.case_id] in seen and seen[images[case.case_id]] != sha:
            raise ValueError('Same train image has different targets; resolve version conflict first')
        if images[case.case_id] in seen:
            manifest.setdefault('excluded_train_duplicate_train',[]).append(case.case_id)
            continue
        seen[images[case.case_id]] = sha
        image = read_image(case.image_path); h,w = image.shape[:2]
        record = {'case_id':case.case_id,'target_sha256':sha,'box_count':0,'gt_box_pins':0,
                  'covered_box_pins':0,'patches':0,'invalid_points':0}
        for owner, component in raw['components'].items():
            if component['type'] != 'box': continue
            box = target_bbox_to_opencv(component['bbox'],h)
            if box[0] >= box[2] or box[1] >= box[3]:
                record.setdefault('invalid_boxes',0); record['invalid_boxes'] += 1; continue
            keys, points = [], []
            for key, pin in raw['pins'].get(owner,{}).items():
                x,y = target_to_opencv(pin['point']['x'],pin['point']['y'],h)
                if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h):
                    record['invalid_points'] += 1; continue
                keys.append(key); points.append((x,y))
            covered = set(); record['box_count'] += 1; record['gt_box_pins'] += len(points)
            for spec in strip_specs(box,image.shape):
                coords = []
                for index,p in enumerate(points):
                    if nearest_side(p,box) != spec['side']: continue
                    col,row = global_to_strip(p,spec)
                    if -3 <= col < POLICY.width+3 and -3 <= row < POLICY.height+3:
                        coords.append((col,row))
                    if 0 <= round(col) < POLICY.width and 0 <= round(row) < POLICY.height:
                        covered.add(index)
                value = sample_strip(image,spec)
                rgb = np.rint((value[:3]+1)*127.5).clip(0,255).astype(np.uint8)
                owner_mask = value[3:].astype(np.uint8)
                name = f'{case.case_id:04d}_{record["patches"]:05d}.npz'
                path = patch/name
                np.savez_compressed(path,rgb=rgb,owner=owner_mask,
                    points=np.array(coords,dtype=np.float32).reshape(-1,2),scale=spec['scale'])
                record['patches'] += 1
            record['covered_box_pins'] += len(covered)
        manifest['records'].append(record); totals.update({k:v for k,v in record.items()
            if isinstance(v,int) and k != 'case_id'})
        print(json.dumps({'prepare_case':f'{case.case_id:04d}','boxes':record['box_count'],
                          'patches':record['patches']},ensure_ascii=True),flush=True)
    manifest['totals'] = dict(totals)
    manifest['actual_unique_train_cases'] = len(manifest['records'])
    manifest['target_access_blocked'] = guard.blocked
    write(OUT/'training_manifest.json',manifest)
    print(json.dumps({'prepared':dict(totals),'validation_count':len(val),'excluded':excluded}),flush=True)
    return manifest


class Patches(Dataset):
    def __init__(self,manifest):
        self.rows=[]
        for record in manifest['records']:
            for i in range(record['patches']):
                path = OUT/'training_patches'/f'{record["case_id"]:04d}_{i:05d}.npz'
                with np.load(path) as v:
                    self.rows.append((v['rgb'].copy(),v['owner'].copy(),v['points'].copy(),float(v['scale'])))
        if not self.rows: raise ValueError('No training patches')
        self.grid=np.mgrid[:POLICY.height,:POLICY.width].astype(np.float32)

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        rgb,owner,points,scale = self.rows[index]
        # Image translation against a fixed owner hint simulates bbox error.
        limit=int(3/scale); dx=random.randint(-limit,limit); dy=random.randint(-limit,limit)
        rgb=cv2.warpAffine(rgb.transpose(1,2,0),np.float32([[1,0,dx],[0,1,dy]]),
            (POLICY.width,POLICY.height),borderValue=(255,255,255)).transpose(2,0,1).astype(np.float32)/255
        rgb=np.clip(rgb*random.uniform(.85,1.15)+random.uniform(-.08,.08),0,1)*2-1
        image=np.concatenate((rgb,owner.astype(np.float32)),axis=0)
        heat=np.zeros((1,POLICY.height,POLICY.width),np.float32)
        offsets=np.zeros((2,POLICY.height,POLICY.width),np.float32); mask=np.zeros_like(heat)
        rr,cc=self.grid
        for col,row in points+np.array([dx,dy]):
            x,y=int(round(float(col))),int(round(float(row)))
            if not (0<=x<POLICY.width and 0<=y<POLICY.height): continue
            heat[0]=np.maximum(heat[0],np.exp(-((rr-y)**2+(cc-x)**2)/(2*POLICY.sigma**2)))
            offsets[:,y,x]=(col-x,row-y); mask[0,y,x]=1
        return tuple(torch.from_numpy(x.copy()) for x in (image,heat,offsets,mask))


def train():
    frozen=freeze()
    if (OUT/'training_complete.json').exists():
        raise RuntimeError('L1 training already completed; do not rerun implicitly')
    manifest=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    if any(split_for_case(i) != 'train' for i in manifest['train_ids']): raise AssertionError('Split leak')
    random.seed(POLICY.seed); np.random.seed(POLICY.seed); torch.manual_seed(POLICY.seed)
    torch.cuda.manual_seed_all(POLICY.seed); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available(): raise RuntimeError('GPU missing; do not silently change device protocol')
    data=Patches(manifest)
    loader=DataLoader(data,batch_size=POLICY.batch,shuffle=True,num_workers=0,pin_memory=True)
    model=PinPointNet().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=POLICY.learning_rate)
    scaler=torch.amp.GradScaler('cuda'); best=float('inf'); curve=[]; start=time.perf_counter()
    for epoch in range(1,POLICY.epochs+1):
        model.train(); total=0.; n=0; tick=time.perf_counter()
        for image,heat,offset,mask in loader:
            image,heat,offset,mask=(x.cuda(non_blocking=True) for x in (image,heat,offset,mask))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): output=model(image)
            loss=point_loss(output,heat,offset,mask)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(opt); scaler.update(); total+=float(loss.detach())*len(image); n+=len(image)
        value=total/n
        row={'epoch':epoch,'loss':value,'seconds':time.perf_counter()-tick}
        curve.append(row); write(OUT/'training_curve.json',curve)
        if value < best:
            best=value
            torch.save({'state_dict':model.state_dict(),'policy':asdict(POLICY),'epoch':epoch,
                'training_loss':value,'train_ids':manifest['train_ids'],'validation_ids':manifest['validation_ids'],
                'frozen_policy':frozen,'manifest_sha256':digest(OUT/'training_manifest.json')},OUT/'best.pt')
        print(json.dumps(row),flush=True)
    payload={'epochs':POLICY.epochs,'patches':len(data),'best_training_loss':best,
        'elapsed_seconds':time.perf_counter()-start,'weight_sha256':digest(OUT/'best.pt'),
        'parameters':sum(p.numel() for p in model.parameters()),'device':torch.cuda.get_device_name(0),
        'validation_labels_read':False,'sealed_holdout_used':False,'frozen_policy':frozen}
    write(OUT/'training_complete.json',payload)
    return payload


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--task',choices=('prepare','train'),required=True)
    args=parser.parse_args()
    prepare() if args.task == 'prepare' else train()
