"""Non-official evaluator V2 aligned to the audited raw official target schema."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import sys
local_runtime=Path(__file__).resolve().parent.parent/'work/runtime'
try:
    import scipy  # noqa: F401
except ImportError:
    if local_runtime.exists():sys.path.insert(0,str(local_runtime))
import argparse,json,math,re
import numpy as np
from pcb.assignment import maximum_weight_assignment
from pcb.io import read_image,write_json
from pcb.submission import validate_strict

OFFICIAL_SCORE=False
HOLDOUT_START=151
EMPTY_RE=re.compile(r'^∅\d+$')
P_INTERNAL_RE=re.compile(r'^P_?\d+$',re.I)
VALUE_LIKE_RE=re.compile(r'(?:ohm|hz|pf|nf|uf|µf|μf|\d\s*[kKmMuUnNpPΩ])',re.I)

def linear_sum_assignment(cost):
    """Dependency-free minimum-cost assignment used by the diagnostic evaluator."""
    array=np.asarray(cost,dtype=float)
    if array.size==0:return np.asarray([],dtype=int),np.asarray([],dtype=int)
    ceiling=float(np.max(array))+1.0
    pairs=maximum_weight_assignment((ceiling-array).tolist())
    return np.asarray([r for r,_ in pairs],dtype=int),np.asarray([c for _,c in pairs],dtype=int)

def div(a,b):return a/b if b else (1.0 if a==0 else 0.0)
def prf(tp,pred,gt):
    p=div(tp,pred);r=div(tp,gt)
    # If prediction and reference are non-empty but share no matches, F1 is 0.
    # Only two truly empty sets receive the conventional perfect score.
    f1=(2*p*r/(p+r)) if (p+r) else (1.0 if pred==0 and gt==0 else 0.0)
    return {'tp':int(tp),'pred':int(pred),'gt':int(gt),'precision':p,'recall':r,'f1':f1}
def point(p):return float(p['x']),float(p['y'])
def bbox_error(a,b):return max(abs(float(x)-float(y)) for x,y in zip(a,b))
def source_from_name(name):
    for s in ('KiCad','Altium Designer','Datasheet','jlc','other'):
        if s.lower() in name.lower():return s
    return 'other'

def parse_ref(ref,component_keys):
    for key in sorted(component_keys,key=len,reverse=True):
        prefix=key+'.'
        if ref.startswith(prefix):return key,ref[len(prefix):]
    return None,ref

def net_members(net):
    text=str(net.get('hyperGraph',''))
    return set(text[1:-1].split(',')) if text.startswith('(') and text.endswith(')') and text[1:-1] else set()

def hungarian_pairs(a_sets,b_sets):
    if not a_sets or not b_sets:return []
    score=np.zeros((len(a_sets),len(b_sets)),float)
    for i,a in enumerate(a_sets):
        for j,b in enumerate(b_sets):score[i,j]=div(2*len(a&b),len(a)+len(b))
    rr,cc=linear_sum_assignment(-score)
    return [(int(i),int(j),float(score[i,j])) for i,j in zip(rr,cc) if score[i,j]>0]

def component_key_map(pred,gt):
    # Ordinary components keep strict key identity.  GND identifiers are all
    # withheld from exact matching because the organizer confirmed their serial
    # numbers have no scoring meaning; match the complete GND sets spatially.
    mapping={k:k for k in pred if k in gt and pred[k].get('type')!='gnd' and gt[k].get('type')!='gnd'}
    pg=[k for k,c in pred.items() if c['type']=='gnd']
    gg=[k for k,c in gt.items() if c['type']=='gnd']
    if pg and gg:
        cost=np.full((len(pg),len(gg)),1e6,float)
        for i,p in enumerate(pg):
            for j,g in enumerate(gg):cost[i,j]=bbox_error(pred[p]['bbox'],gt[g]['bbox'])
        rr,cc=linear_sum_assignment(cost)
        for i,j in zip(rr,cc):
            if cost[i,j]<=20:mapping[pg[i]]=gg[j]
    return mapping

def component_counts(pred,gt,key_map,observable=False):
    def keep(k,c):return (not EMPTY_RE.fullmatch(k)) or c['type']=='gnd'
    pkeys=[k for k,c in pred.items() if not observable or keep(k,c)];gkeys=[k for k,c in gt.items() if not observable or keep(k,c)]
    tp=0
    for pk in pkeys:
        gk=key_map.get(pk)
        if gk not in gt or gk not in gkeys:continue
        p,g=pred[pk],gt[gk]
        if p['type']!=g['type'] or bbox_error(p['bbox'],g['bbox'])>20:continue
        if p['type']=='box' and p.get('Name')!=g.get('Name'):continue
        if p['type'] in ('r','c') and p.get('value')!=g.get('value'):continue
        tp+=1
    return tp,len(pkeys),len(gkeys)

def observable_pin(component_key,pin_key,pin,components):
    c=components.get(component_key,{})
    if EMPTY_RE.fullmatch(component_key) and c.get('type')!='gnd':return False
    suffix=pin_key[4:] if pin_key.startswith('pin_') else pin_key
    if suffix.isdigit() and int(suffix)>=100:return False
    if any(x.isspace() for x in suffix) or VALUE_LIKE_RE.search(suffix):return False
    name=str(pin.get('pinname',''))
    if not name or P_INTERNAL_RE.fullmatch(name) or VALUE_LIKE_RE.search(name):return False
    return True

def pin_counts(pred,gt,key_map,observable=False):
    pp=[];gg=[]
    for ck,pins in pred.items():
        for pk,p in pins.items():
            if not observable or observable_pin(ck,pk,p,pred_components_global):pp.append((ck,pk,p))
    for ck,pins in gt.items():
        for pk,p in pins.items():
            if not observable or observable_pin(ck,pk,p,gt_components_global):gg.append((ck,pk,p))
    gt_lookup={(ck,pk):p for ck,pk,p in gg};tp=0
    for ck,pk,p in pp:
        g=gt_lookup.get((key_map.get(ck,''),pk))
        if g and p.get('pinname','')==g.get('pinname','') and math.dist(point(p['point']),point(g['point']))<=5:tp+=1
    return tp,len(pp),len(gg)

def normalize_pred_ref(ref,pred_components,key_map):
    ck,suffix=parse_ref(ref,pred_components)
    return f'{key_map[ck]}.{suffix}' if ck in key_map else f'__UNMATCHED__{ref}'

def allowed_gt_refs(gt,observable):
    if not observable:return None
    out=set()
    for ck,pins in gt['pins'].items():
        for pk,p in pins.items():
            if observable_pin(ck,pk,p,gt['components']):out.add(f'{ck}.{pk[4:]}')
    return out

def pair_set(netsets):
    result=set()
    for values in netsets:
        items=sorted(values)
        for i in range(len(items)):
            for j in range(i+1,len(items)):result.add((items[i],items[j]))
    return result

def segment_match_count(pred_edges,gt_edges):
    if not pred_edges or not gt_edges:return 0
    mat=np.zeros((len(pred_edges),len(gt_edges)),dtype=np.int8)
    for i,(pa,pb) in enumerate(pred_edges):
        for j,(ga,gb) in enumerate(gt_edges):
            direct=max(math.dist(point(pa),point(ga)),math.dist(point(pb),point(gb)))
            reverse=max(math.dist(point(pa),point(gb)),math.dist(point(pb),point(ga)))
            mat[i,j]=min(direct,reverse)<=5
    rr,cc=linear_sum_assignment(-mat);return int(mat[rr,cc].sum())

def evaluate_case(pred,gt,diagnostics=None):
    global pred_components_global,gt_components_global
    pred_components_global,gt_components_global=pred['components'],gt['components']
    key_map=component_key_map(pred['components'],gt['components'])
    c=component_counts(pred['components'],gt['components'],key_map,False);co=component_counts(pred['components'],gt['components'],key_map,True)
    p=pin_counts(pred['pins'],gt['pins'],key_map,False);po=pin_counts(pred['pins'],gt['pins'],key_map,True)
    pnet=list(pred['nets'].values());gnet=list(gt['nets'].values())
    psets=[{normalize_pred_ref(r,pred['components'],key_map) for r in net_members(n)} for n in pnet]
    gsets=[net_members(n) for n in gnet];matches=hungarian_pairs(psets,gsets)
    ntp=sum(len(psets[i]&gsets[j]) for i,j,_ in matches)
    ltp=0
    for i,j,_ in matches:ltp+=segment_match_count(list(pnet[i]['edges'].values()),list(gnet[j]['edges'].values()))
    pairs_p,pairs_g=pair_set(psets),pair_set(gsets);pair_tp=len(pairs_p&pairs_g)
    allowed=allowed_gt_refs(gt,True)
    opsets=[s&allowed for s in psets if s&allowed];ogsets=[s&allowed for s in gsets if s&allowed];om=hungarian_pairs(opsets,ogsets)
    ontp=sum(len(opsets[i]&ogsets[j]) for i,j,_ in om);opp,opg=pair_set(opsets),pair_set(ogsets)
    metrics={'Component':prf(*c),'Pin':prf(*p),'NetHypergraph':prf(ntp,sum(map(len,psets)),sum(map(len,gsets))),
        'NetLine':prf(ltp,sum(len(n['edges']) for n in pnet),sum(len(n['edges']) for n in gnet)),
        'PinPair':prf(pair_tp,len(pairs_p),len(pairs_g))}
    observable={'Component':prf(*co),'Pin':prf(*po),'NetHypergraph':prf(ontp,sum(map(len,opsets)),sum(map(len,ogsets))),
        'PinPair':prf(len(opp&opg),len(opp),len(opg))}
    score=100*(.30*metrics['Component']['f1']+.25*metrics['Pin']['f1']+.35*metrics['NetHypergraph']['f1']+.10*metrics['NetLine']['f1'])
    counts={'pred_component_count':len(pred['components']),'gt_component_count':len(gt['components']),
        'pred_pin_count':sum(map(len,pred['pins'].values())),'gt_pin_count':sum(map(len,gt['pins'].values())),
        'pred_net_count':len(pnet),'gt_net_count':len(gnet),'pred_edge_count':sum(len(n['edges']) for n in pnet),'gt_edge_count':sum(len(n['edges']) for n in gnet),
        'singleton_net_count':sum(len(s)==1 for s in psets),'empty_edge_net_count':sum(not n['edges'] for n in pnet),
        'unattached_pin_count':len((diagnostics or {}).get('unattached_pins',[])),'runtime':float((diagnostics or {}).get('seconds',0))}
    return {'metrics':metrics,'observable_only':observable,'FinalScore':score,'counts':counts,'component_key_matches':len(key_map)}

def aggregate(records):
    ok=[r for r in records if r.get('status')=='ok'];groups={'overall':ok}
    for source in ('KiCad','Altium Designer','Datasheet','jlc','other'):groups[source]=[r for r in ok if r['source']==source]
    result={}
    for name,rows in groups.items():
        if not rows:continue
        block={'case_count':len(rows),'metrics':{},'observable_only':{},'counts':{}}
        for family in ('Component','Pin','NetHypergraph','NetLine','PinPair'):
            vals=[r['evaluation']['metrics'][family] for r in rows]
            micro=prf(sum(v['tp'] for v in vals),sum(v['pred'] for v in vals),sum(v['gt'] for v in vals))
            block['metrics'][family]={**micro,'macro_f1':sum(v['f1'] for v in vals)/len(vals)}
        for family in ('Component','Pin','NetHypergraph','PinPair'):
            vals=[r['evaluation']['observable_only'][family] for r in rows]
            micro=prf(sum(v['tp'] for v in vals),sum(v['pred'] for v in vals),sum(v['gt'] for v in vals))
            block['observable_only'][family]={**micro,'macro_f1':sum(v['f1'] for v in vals)/len(vals)}
        for key in rows[0]['evaluation']['counts']:block['counts'][key]=sum(r['evaluation']['counts'][key] for r in rows)
        block['FinalScore']=100*(.30*block['metrics']['Component']['macro_f1']+.25*block['metrics']['Pin']['macro_f1']+.35*block['metrics']['NetHypergraph']['macro_f1']+.10*block['metrics']['NetLine']['macro_f1'])
        result[name]=block
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prediction_dir',type=Path,required=True);ap.add_argument('--target_root',type=Path,required=True);ap.add_argument('--case_start',type=int,default=1);ap.add_argument('--case_end',type=int,default=150);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not(1<=a.case_start<=a.case_end<HOLDOUT_START):raise SystemExit('Holdout guard: only 0001..0150 may be evaluated')
    if a.target_root.name!='200_train_cases':raise SystemExit('target_root must be exactly 200_train_cases')
    records=[]
    for n in range(a.case_start,a.case_end+1):
        case=f'{n:04d}';folder=a.target_root/case
        try:
            target_path=next(folder.glob('*_target.json'));image_path=next(folder.glob('*.png'));pred_path=a.prediction_dir/case/'result.json';diag_path=a.prediction_dir/case/'diagnostics.json'
            pred=json.loads(pred_path.read_text(encoding='utf-8'));gt=json.loads(target_path.read_text(encoding='utf-8'));diag=json.loads(diag_path.read_text(encoding='utf-8')) if diag_path.exists() else {}
            image=read_image(image_path);validate_strict(pred,(image.shape[1],image.shape[0]));ev=evaluate_case(pred,gt,diag)
            records.append({'case_id':case,'source':source_from_name(image_path.name),'status':'ok','evaluation':ev})
        except Exception as exc:records.append({'case_id':case,'status':'failed','error':repr(exc)})
    failures=[r for r in records if r['status']!='ok'];out={'OFFICIAL_SCORE':OFFICIAL_SCORE,'case_start':a.case_start,'case_end':a.case_end,'holdout_used':False,
        'success_count':len(records)-len(failures),'failure_count':len(failures),'aggregate':aggregate(records),'cases':records,
        'confirmed_rules':['component key direct match','bbox threshold 20px','pin key/name direct match','pin point 5px','net Hungarian','line endpoint 5px','net key ignored','gnd serial ignored per organizer clarification'],
        'inferred_rules':['raw 43-type target taxonomy supersedes older PDF example names for this benchmark','macro average follows single-case scoring formula'],
        'unknown_rules':['official tie-breaking and all empty-set boundary cases','non-gnd internal-key scoring in hidden set','special internal-like pin IDs in hidden set']}
    write_json(a.output,out);print(json.dumps({'success_count':out['success_count'],'failure_count':out['failure_count'],'aggregate':out['aggregate'].get('overall')},ensure_ascii=False,indent=2))
    return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
