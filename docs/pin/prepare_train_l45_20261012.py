"""Training supervision only; deployment inputs are frozen real predicted boxes."""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import random
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pcb.pin.rc_pair_l45 import POLICY,PairResidualNet,sample,canonical
from pcb.schema import Component,Pin
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

TASK=Path('C:/Users/bluty/Documents/Codex/2026-10-12/810-c-users-bluty-documents-codex')
STUDY=Path('D:/Codex_Experiments/Pin_20261012/work/pin_l45_remaining_validation_20261012')
REPORT=TASK/'outputs/pin_l45_remaining_validation_20261012'
OLD35={14,17,18,87,103}|set(range(5,151,5))
FILES=['pcb/pin/rc_pair_l45.py','configs/pin/l45_remaining_validation.json','tests/pin/test_rc_pair_l45.py',
       'docs/pin/l45_remaining_validation_plan_20261012.md','docs/pin/prepare_train_l45_20261012.py',
       'docs/pin/run_l45_remaining_validation_20261012.py']

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8');tmp.replace(p)
def event(name,**kw):print(json.dumps({'event':name,**kw},ensure_ascii=True),flush=True)
def protect():
    previous=read(ROOT/'runs/pin_l42_full150_20261012/initial.json')['signature']
    for rel,digest in previous['runtime_hashes'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen L42 file changed: '+rel)
    for record in previous['binary_and_effective_resources']:
        if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Frozen model/resource changed')
    original=ROOT.parent/'EDA_Team_Modular_Integration'
    manifest=read(original/'docs/pin/baseline_20261011_l39/runtime_manifest.json')
    for record in manifest['files'].values():
        if sha(original/record['source_relative'])!=record['sha256']:raise ValueError('Original L39 baseline changed')
    return {rel:sha(ROOT/rel) for rel in FILES}
def status(phase,**kw):write(REPORT/'执行状态.json',{'phase':phase,'complete':False,'working_directory':str(STUDY),**kw})

def prepare():
    if STUDY.exists():raise ValueError('Fresh study root required')
    frozen=protect()
    config=read(ROOT/'configs/pin/l45_remaining_validation.json')
    if any(config[k]!=v for k,v in asdict(POLICY).items()):raise ValueError('Policy/config mismatch')
    allowed=[n for n in range(1,151) if n not in OLD35]
    val=set(sorted(allowed,key=lambda n:hashlib.sha256(f'L45|{n:04d}'.encode()).hexdigest())[:15])
    train=set(allowed)-val
    STUDY.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2(ROOT/'docs/pin/l45_remaining_validation_plan_20261012.md',REPORT/'实验计划与完成条件.md')
    for rel in FILES:
        target=STUDY/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    write(STUDY/'study_initial.json',{'source_files':frozen,'policy':asdict(POLICY),
          'train_images':sorted(train),'validation_images':sorted(val),'excluded_design_check':sorted(OLD35),
          'GT_supervision_only':True,'L43_weight_used':False,'opponent_constants_used':False})
    status('PREPARING_TRAINING_INPUTS')
    from tools.team_access_guard import Guard
    from evaluate_v2 import component_key_map,bbox_error
    from pcb.io import read_image
    guard=Guard()
    old=ROOT/'runs/pin_l42_full150_20261012'
    old_inputs={r['case_id']:r for r in read(old/'initial.json')['inputs']}
    groups={name:{'rgb':[],'prior':[],'features':[],'target':[],'scale':[],'records':[]} for name in ('train','val')}
    rejects=[];targets=[];alphas=[]
    for n in allowed:
        cid=f'{n:04d}';item=old_inputs[cid]
        if sha(item['path'])!=item['sha256']:raise ValueError('PNG changed')
        image=read_image(Path(item['path']))
        scene=read(old/cid/'B/scene.json');pred=read(old/cid/'B/result.json')
        paths=list(Path(item['path']).parent.glob('*_target.json'))
        if len(paths)!=1:raise ValueError('One allowed supervision target required')
        with guard.evaluating(paths[0],[old/cid/'B/result.json']):
            gt=read(paths[0]);target_hash=sha(paths[0])
        targets.append({'case':cid,'target_sha256':target_hash})
        mapping=component_key_map(pred['components'],gt['components'])
        group=groups['val' if n in val else 'train']
        for row in scene['components']:
            if row['type'] not in ('r','c'):continue
            mapped=mapping.get(row['key'])
            if mapped not in gt['components'] or gt['components'][mapped]['type']!=row['type']:continue
            if bbox_error(pred['components'][row['key']]['bbox'],gt['components'][mapped]['bbox'])>20:continue
            gp=list(gt['pins'].get(mapped,{}).values())
            if len(gp)!=2:continue
            owner=Component(**{**row,'pins':[Pin(**p) for p in row['pins']]})
            sampled=sample(image,owner,owner.pins)
            if sampled is None:continue
            rgb,prior,features,spec=sampled
            points=np.asarray([[float(p['point']['x']),image.shape[0]-float(p['point']['y'])] for p in gp],np.float32)
            normalized=canonical(points,spec);normalized=normalized[np.argsort(normalized[:,0])]
            if np.max(np.abs(normalized))>1:
                rejects.append({'case':cid,'owner':row['key'],'reason':'label_outside_fixed_input_window'});continue
            group['rgb'].append(rgb);group['prior'].append(prior);group['features'].append(features)
            group['target'].append(normalized.reshape(4));group['scale'].append([spec['long_half'],spec['short_half']]*2)
            group['records'].append({'case':cid,'owner':row['key'],'type':row['type'],'input_frame':spec})
            if n in train:
                axis=1 if spec['vertical'] else 0;ordered=points[np.argsort(points[:,axis])]
                a,b,c,d=spec['bbox'];lo,hi=(b,d) if spec['vertical'] else (a,c);short=(c-a) if spec['vertical'] else (d-b)
                alphas.extend([(lo-ordered[0,axis])/short,(ordered[1,axis]-hi)/short])
        event('prepare_case',case=cid,train=len(groups['train']['records']),val=len(groups['val']['records']))
    for name,group in groups.items():
        for key in ('rgb','prior','features','target','scale'):
            values=np.stack(group[key]);np.save(STUDY/f'{name}_{key}.npy',values)
        write(STUDY/f'{name}_objects.json',group['records'])
    write(STUDY/'calibration.json',{'alpha':float(np.median(alphas)),'fit':'one joint R/C normalized lead median',
          'fit_endpoint_count':len(alphas),'training_images':sorted(train),'validation_used_for_fit':False})
    complete={'train_objects':len(groups['train']['records']),'val_objects':len(groups['val']['records']),
              'rejected':rejects,'target_hashes':targets,'supervision_GT_open_events':guard.target_opens,
              'inputs_are_predicted_boxes_not_GT_boxes':True,'source_files':frozen}
    write(STUDY/'preparation_complete.json',complete);status('TRAINING_INPUTS_READY',**{k:v for k,v in complete.items() if k in ('train_objects','val_objects')})
    if protect()!=frozen:raise ValueError('Source changed during preparation')
    event('preparation_complete',train_objects=complete['train_objects'],val_objects=complete['val_objects'],calibration_alpha=float(np.median(alphas)))

class Samples(Dataset):
    def __init__(self,name):self.data={k:np.load(STUDY/f'{name}_{k}.npy',mmap_mode='r') for k in ('rgb','prior','features','target','scale')}
    def __len__(self):return len(self.data['prior'])
    def __getitem__(self,i):
        return tuple(torch.from_numpy(np.array(self.data[k][i],copy=True)) for k in ('rgb','prior','features','target','scale'))

def augment(rgb,prior,features,target):
    for axis in (0,1):
        chosen=torch.rand(len(rgb),device=rgb.device)<.5
        if not chosen.any():continue
        rgb[chosen]=rgb[chosen].flip(3 if axis==0 else 2)
        pp=prior[chosen].reshape(-1,2,2);tt=target[chosen].reshape(-1,2,2)
        pp[:,:,axis]*=-1;tt[:,:,axis]*=-1
        if axis==0:pp=pp.flip(1);tt=tt.flip(1)
        prior[chosen]=pp.flatten(1);target[chosen]=tt.flatten(1)
        for off in (0,4):
            low=features[chosen,off+axis].clone();high=features[chosen,off+axis+2].clone()
            features[chosen,off+axis]=-high;features[chosen,off+axis+2]=-low
    return rgb,prior,features,target

def train():
    initial=read(STUDY/'study_initial.json');frozen=protect()
    if frozen!=initial['source_files']:raise ValueError('Frozen source changed')
    if (STUDY/'training_complete.json').exists():raise ValueError('Training never overwritten')
    random.seed(POLICY.seed);np.random.seed(POLICY.seed);torch.manual_seed(POLICY.seed)
    torch.set_num_threads(4)
    device='cuda'
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    model=PairResidualNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY.learning_rate,weight_decay=POLICY.weight_decay)
    train_loader=DataLoader(Samples('train'),batch_size=POLICY.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(Samples('val'),batch_size=POLICY.batch,shuffle=False,num_workers=0)
    curve=[];best=float('inf');started=time.perf_counter()
    for epoch in range(1,POLICY.epochs+1):
        model.train();losses=[]
        for rgb,prior,features,target,scale in train_loader:
            rgb=rgb.to(device).permute(0,3,1,2).float()/127.5-1
            prior,features,target,scale=(t.to(device).float() for t in (prior,features,target,scale))
            rgb,prior,features,target=augment(rgb,prior,features,target)
            output=model(rgb,prior,features)
            loss=torch.nn.functional.smooth_l1_loss(output*scale,target*scale,beta=2.)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval();errors=[];baseline=[]
        with torch.inference_mode():
            for rgb,prior,features,target,scale in val_loader:
                rgb=rgb.to(device).permute(0,3,1,2).float()/127.5-1
                prior,features,target,scale=(t.to(device).float() for t in (prior,features,target,scale))
                output=model(rgb,prior,features)
                errors.extend(torch.linalg.vector_norm(((output-target)*scale).reshape(-1,2),dim=1).cpu().tolist())
                baseline.extend(torch.linalg.vector_norm(((prior-target)*scale).reshape(-1,2),dim=1).cpu().tolist())
        mae=float(np.mean(errors));row={'epoch':epoch,'train_loss':float(np.mean(losses)),
            'validation_mean_endpoint_pixels':mae,'validation_at5':float(np.mean(np.asarray(errors)<=5)),
            'prior_mean_endpoint_pixels':float(np.mean(baseline)),'prior_at5':float(np.mean(np.asarray(baseline)<=5))}
        curve.append(row)
        if mae<best:
            best=mae
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'policy':asdict(POLICY),'epoch':epoch,'validation_mean_endpoint_pixels':mae},STUDY/'best.pt')
        write(STUDY/'training_curve.json',curve);status('TRAINING_RUNNING',epoch=epoch,epochs=POLICY.epochs,best_validation_error=best)
        event('training_epoch',**row)
    if protect()!=frozen:raise ValueError('Source changed during training')
    complete={'best_epoch':torch.load(STUDY/'best.pt',map_location='cpu',weights_only=True)['epoch'],
              'best_validation_mean_endpoint_pixels':best,'checkpoint_sha256':sha(STUDY/'best.pt'),
              'calibration_sha256':sha(STUDY/'calibration.json'),'seconds':time.perf_counter()-started,
              'GT_labels_in_training_only':True,'prediction_boxes_used':True,'L43_used':False,
              'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'GPU_peak_reserved':torch.cuda.max_memory_reserved()}
    write(STUDY/'training_complete.json',complete);status('TRAINING_COMPLETE_VALIDATION_PENDING',**complete)
    event('training_complete',**complete)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','train'),required=True)
    args=parser.parse_args();{'prepare':prepare,'train':train}[args.phase]()
