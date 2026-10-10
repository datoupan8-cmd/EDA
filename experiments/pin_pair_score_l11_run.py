"""One fixed Design A/B with cached image-only inputs and unchanged downstream.

Annotations are opened only after A and B predictions have been saved. They
never enter the scorer. Existing targets must match the previous Design hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT)]
import pin_pair_admission_audit_l10 as cached
import pin_joint_abstention_l7_4 as prior
import pin_pair_score_l11 as scorer
from pin_contextual_roles_e6 import assert_invariants
from pin_skeleton_followup import strict_ids
from pin_g_wire_e1 import pair_changes
from wire_frame_guard import suppress_box_frames
from pcb.schema import Scene, Text
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.registry import build_default_registry
from pcb.io import read_image, write_json
from pcb.submission import validate_strict
from evaluate_v2 import evaluate_case, aggregate, source_from_name

OUT = ROOT/'reports/pin_pair_score_l11'
DATASET = Path(r'C:\Users\bluty\Desktop\PCB_Competition\赛题六公开数据集\200_train_cases')
DESIGN = cached.DESIGN
REFERENCE = ROOT/'reports/pin_joint_abstention_l7_4/design'


def json_hash(value):
    def encode(item):
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,np.ndarray):return item.tolist()
        raise TypeError(type(item).__name__)
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':'),default=encode).encode('utf-8')).hexdigest()


class EvaluationGuard:
    """Default-deny targets, with a narrow post-prediction evaluator context."""
    def __init__(self, install=True):
        self.allowed_target = None
        self.blocked = 0
        self.target_opens = 0
        if install:sys.addaudithook(self.audit)

    def audit(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):
            return
        value = str(args[0]).replace('\\','/').lower()
        parts = value.split('/')
        forbidden = any(p in {'10gt','10_gtcase','golden','sealed','eda_pin_crossing_quicktest'} for p in parts)
        if '200_train_cases' in parts:
            i = parts.index('200_train_cases')
            forbidden |= i+1>=len(parts) or parts[i+1] not in DESIGN
        is_target = value.endswith('_target.json')
        if forbidden or (is_target and value!=self.allowed_target):
            self.blocked+=1
            raise PermissionError('L11 denies reserved/non-Design inputs and inference target reads')
        if is_target:self.target_opens+=1

    @contextmanager
    def evaluating(self,path,prediction_path):
        if self.allowed_target is not None or not prediction_path.is_file():
            raise AssertionError('Evaluation must follow saved image-only prediction')
        self.allowed_target = str(path).replace('\\','/').lower()
        try:yield
        finally:self.allowed_target=None


def paths_for(key):
    if key not in DESIGN:raise ValueError('Fixed Design cases only')
    folder = DATASET/key
    images, targets = list(folder.glob('*.png')),list(folder.glob('*_target.json'))
    if len(images)!=1 or len(targets)!=1:
        raise RuntimeError(f'Unique image/target paths not resolved: {key}')
    return images[0],targets[0]


def hydrate(key,inputs):
    components,terms,pool,trace,signature = cached.hydrate_case(key,inputs)
    front_path = ROOT.parent/f'tmp/pin_sina_frontend/{key}.json'
    baseline_path = ROOT/f'reports/pin_terminal_reader_l2/design/predictions/L1/{key}/diagnostics.json'
    raw_path = ROOT/f'reports/pin_terminal_reader_l2/design/raw_terminals/{key}.json'
    front = cached.read(front_path)
    image_path,target_path = paths_for(key)
    image_hash = cached.digest(image_path)
    inputs[str(image_path)] = image_hash
    if image_hash!=signature['image_sha256']:
        raise AssertionError('Frozen frontend image changed')
    image = read_image(image_path)
    height,width = image.shape[:2]
    texts = [Text(**{**t,'bbox':tuple(t['bbox'])}) for t in front['texts']]
    scene = Scene(width,height,list(components.values()),texts,
                  diagnostics=copy.deepcopy(cached.read(baseline_path)))
    context = PipelineContext(image_path.name,width,height,PipelineConfig.load(ROOT/'configs/current.json'),None)
    return scene,cached.read(raw_path),pool,trace,image,context,target_path


def predict(scene,image,context,registry):
    wire = registry.create('wire','v3').run(image,scene,context)
    wire = suppress_box_frames(image,scene,wire)
    topology = registry.create('topology','v3').run(scene,wire,context)
    pred = registry.create('submission','official').run(scene,context).data
    validate_strict(pred,(context.width,context.height))
    geometry_fields = ('wire_paths','wire_segments','wire_pixels','wire_groups',
        'junctions','nonconnecting_crossings','short_bridges','snap_radius',
        'snap_candidates_evaluated','snap_rejections','signal_label_hits')
    geometry = {k:scene.diagnostics[k] for k in geometry_fields}
    geometry['snaps'] = [{k:v for k,v in s.items() if k!='pin'} for s in scene.diagnostics['snaps']]
    geometry['unattached_count'] = len(scene.diagnostics['unattached_pins'])
    def array_hash(value):
        return None if value is None else hashlib.sha256(value.tobytes()).hexdigest()
    return pred,{'mask':array_hash(wire.mask),'corridor':array_hash(wire.corridor),
                 'skeleton':array_hash(topology.skeleton),'snap_paths':json_hash(geometry)}


def selection_signature(trace):
    return {r['component']:[cached.short_option(p) for p in r['selected']] for r in trace}


def baseline_parity(key,scene,raw,pool,saved_trace,image,context,registry,inputs):
    before = json_hash([asdict(scene),raw,pool])
    candidate,trace = prior.stage(scene,raw,pool)
    assert_invariants(scene,candidate,raw)
    if json_hash([asdict(scene),raw,pool])!=before:raise AssertionError('A mutated cached inputs')
    saved_signature = selection_signature(list(saved_trace.values()))
    if selection_signature(trace)!=saved_signature:
        raise AssertionError(f'A saved choice parity failed: {key}')
    pred,geometry = predict(candidate,image,context,registry)
    ref_path = REFERENCE/f'predictions/D/{key}/result.json'
    inputs[str(ref_path)] = cached.digest(ref_path)
    if pred!=cached.read(ref_path):raise AssertionError(f'A full prediction parity failed: {key}')
    return candidate,trace,pred,geometry


def smoke():
    guard = EvaluationGuard()
    inputs = {}
    protected = cached.protected_snapshot()
    row = hydrate(DESIGN[0],inputs)
    scene,raw,pool,trace,image,context,_ = row
    baseline_parity(DESIGN[0],scene,raw,pool,trace,image,context,build_default_registry(),inputs)
    if cached.protected_snapshot()!=protected:raise AssertionError('Protected files changed')
    cached.write(OUT/'smoke.json',{'case_id':DESIGN[0],'full_A_prediction_equal':True,
        'saved_A_choices_equal':True,'target_opens':guard.target_opens,'OCR_runs':0,
        'blocked_access_attempts':guard.blocked,'protected_unchanged':True})
    print(json.dumps({'A_smoke_parity':True,'target_reads':guard.target_opens}),flush=True)


def run():
    if (OUT/'design/complete.json').exists():
        raise RuntimeError('One Design run already completed; no repeated parameter search')
    started = time.perf_counter()
    guard,registry = EvaluationGuard(),build_default_registry()
    protected,git_before,inputs = cached.protected_snapshot(),cached.git_state(),{}
    source_paths = ('experiments/pin_pair_score_l11.py','experiments/pin_pair_score_l11_run.py',
                    'tests/pin/test_pin_pair_score_l11.py','reports/pin_pair_score_l11/experiment_plan.md')
    frozen = {'policy':copy.deepcopy(scorer.POLICY),'case_ids':list(DESIGN),
        'source_sha256':{p:cached.digest(ROOT/p) for p in source_paths},
        'protected_sha256':protected,'git_before':git_before,
        'OFFICIAL_SCORE':False,'sealed_holdout_used':False,'OCR_runs':0,'inference_GT_reads':False}
    frozen_path = OUT/'design/frozen_protocol.json'
    if frozen_path.exists() and cached.read(frozen_path)!=frozen:
        raise AssertionError('Frozen L11 protocol changed before completion')
    cached.write(frozen_path,frozen)
    source = REFERENCE/'complete.json'
    inputs[str(source)] = cached.digest(source)
    reference = cached.read(source)
    hydrated = {}; A = {}
    # All five A outputs must reproduce before any B experiment or target IO.
    for key in DESIGN:
        hydrated[key] = hydrate(key,inputs)
        scene,raw,pool,trace,image,context,_ = hydrated[key]
        A[key] = baseline_parity(key,scene,raw,pool,trace,image,context,registry,inputs)
        print(json.dumps({'case':key,'A_prediction_parity':True,'A_choice_parity':True}),flush=True)
    cached.write(OUT/'design/A_parity.json',{'cases':list(DESIGN),'prediction_equal':True,
        'choices_equal':True,'target_opens_before_A_parity':guard.target_opens})
    evaluations = {'A':[],'B':[]}; records=[]; gained=[];lost=[];pairs=Counter();calibration=Counter()
    for key in DESIGN:
        baseline,raw,pool,_,image,context,target_path = hydrated[key]
        a_scene,a_trace,a_pred,a_geometry = A[key]
        before = json_hash([asdict(baseline),raw,pool])
        b_scene,b_trace,score_trace = scorer.stage(baseline,raw,pool)
        assert_invariants(baseline,b_scene,raw)
        if json_hash([asdict(baseline),raw,pool])!=before:raise AssertionError('B changed cached inputs')
        b_pred,b_geometry = predict(b_scene,image,context,registry)
        if a_pred['components']!=b_pred['components']:raise AssertionError('Component output changed')
        if a_geometry!=b_geometry:raise AssertionError('Wire/topology geometry changed')
        dest = OUT/f'design/predictions/B/{key}'
        cached.write(dest/'result.json',b_pred)
        write_json(dest/'diagnostics.json',b_scene.diagnostics)
        cached.write(OUT/f'design/trace/{key}.json',b_trace)
        cached.write(OUT/f'design/score_trace/{key}.json',score_trace)
        changed=[]
        a_owners={r['component']:r for r in a_trace}
        b_owners={r['component']:r for r in b_trace}
        for owner,row in score_trace.items():
            for side,fit in row['layout'].items():
                calibration['supported' if fit['supported'] else 'unsupported']+=1
            pa={p['terminal']:p for p in a_owners[owner]['selected']}
            pb={p['terminal']:p for p in b_owners[owner]['selected']}
            for i,old in pa.items():
                new=pb[i]
                fields=('number','name','number_word','name_word','fallback','abstain')
                if any(old.get(f)!=new.get(f) for f in fields):
                    changed.append({'owner':owner,'terminal':i,'A':cached.short_option(old),
                        'B':cached.short_option(new),'raw':copy.deepcopy(next(r['terminals'][i]
                            for r in raw if r['component']==owner))})
        # Only an independent evaluator now opens the latest original target.
        with guard.evaluating(target_path,dest/'result.json'):
            target_sha = cached.digest(target_path)
            old = next(r for r in reference['cases'] if r['case_id']==key)
            if target_sha!=old['target_sha256']:raise AssertionError('Latest official target changed; stop version drift')
            target = cached.read(target_path)
        inputs[str(target_path)] = target_sha
        ids={v:strict_ids(pred,target) for v,pred in (('A',a_pred),('B',b_pred))}
        new=sorted(ids['B']-ids['A']);missing=sorted(ids['A']-ids['B'])
        gained.extend({'case_id':key,'pin':list(p)} for p in new)
        lost.extend({'case_id':key,'pin':list(p)} for p in missing)
        row={'case_id':key,'source':source_from_name(context.image_name),'target_sha256':target_sha,
            'metrics':{},'gained':len(new),'lost':len(missing),'changed_choices':changed,
            'geometry_A':a_geometry,'geometry_B':b_geometry,'pool_unchanged':True,'raw_unchanged':True}
        for variant,pred,scene in (('A',a_pred,a_scene),('B',b_pred,b_scene)):
            evaluation=evaluate_case(pred,target,scene.diagnostics)
            evaluations[variant].append({'case_id':key,'source':row['source'],
                'status':'ok','evaluation':evaluation})
            row['metrics'][variant]=evaluation['metrics']['Pin']
            if evaluation['metrics']['Pin']['tp']!=len(ids[variant]):raise AssertionError('Strict TP disagreement')
        if row['metrics']['A']!=old['metrics']['D']:raise AssertionError('A Pin metrics differ from frozen reference')
        changes=pair_changes(a_pred,b_pred,target)
        row['pair_changes']={k:len(v) for k,v in changes.items()}
        pairs.update(row['pair_changes']);records.append(row)
        cached.write(OUT/f'design/pair_changes/{key}.json',changes)
        cached.write(OUT/'design/progress.json',{'cases':records,'gained':gained,'lost':lost})
        print(json.dumps({'case':key,'A':row['metrics']['A'],'B':row['metrics']['B'],
                          'gained':len(new),'lost':len(missing),'pairs':row['pair_changes']}),flush=True)
    summaries={v:aggregate(evaluations[v]) for v in evaluations}
    a,b=(summaries[v]['overall']['metrics']['Pin'] for v in ('A','B'))
    improved=[r['case_id'] for r in records if r['metrics']['B']['f1']>r['metrics']['A']['f1']]
    regressed=[r['case_id'] for r in records if r['metrics']['B']['f1']<r['metrics']['A']['f1']]
    guards={'strict_TP_net_gain':b['tp']>a['tp'],'micro_F1_net_gain':b['f1']>a['f1'],
        'macro_F1_not_lower':b['macro_f1']>=a['macro_f1'],'no_old_TP_loss':not lost,
        'no_incorrect_Pair_added':not pairs['incorrect_added'],'no_correct_Pair_lost':not pairs['correct_lost']}
    big_gain={'TP_relative_gain_at_least_20pct':b['tp']>=1.2*a['tp'] and b['tp']>a['tp'],
        'micro_F1_relative_gain_at_least_20pct':b['f1']>=1.2*a['f1'] and b['f1']>a['f1'],
        'at_least_3_improved_cases':len(improved)>=3}
    check_allowed=all(guards.values()) and all(big_gain.values())
    # Annotation hashes are checked again in the same isolated evaluator path.
    for p,h in inputs.items():
        path=Path(p)
        if path.name.endswith('_target.json'):
            key=path.parent.name
            with guard.evaluating(path,OUT/f'design/predictions/B/{key}/result.json'):
                actual=cached.digest(path)
        else:actual=cached.digest(path)
        if actual!=h:raise AssertionError('Cached input changed during experiment')
    if cached.protected_snapshot()!=protected:raise AssertionError('Protected formal source changed')
    if {p:cached.digest(ROOT/p) for p in source_paths}!=frozen['source_sha256']:
        raise AssertionError('Single-variable source changed during experiment')
    payload={'metadata':{'case_ids':list(DESIGN),'phase':'design','seconds':time.perf_counter()-started,
        'OFFICIAL_SCORE':False,'sealed_holdout_used':False,'OCR_runs':0,'YOLO_runs':0,
        'localization_runs':0,'inference_GT_reads':False,'target_file_count':len(DESIGN),
        'target_open_count':guard.target_opens,'blocked_access_attempts':guard.blocked},
        'end_to_end':summaries,'cases':records,'gained':gained,'lost':lost,'pair_changes':dict(pairs),
        'calibration_side_counts':dict(calibration),'keep_guards':guards,'large_gain_guards':big_gain,
        'Check_allowed':check_allowed,'current_modified':False,'Check_run':False,
        'improved_cases':improved,'regressed_cases':regressed,
        'unchanged_cases':[key for key in DESIGN if key not in improved+regressed]}
    cached.write(OUT/'design/complete.json',payload)
    cached.write(OUT/'verification.json',{'protected_sha256':protected,'protected_unchanged':True,
        'input_sha256':inputs,'inputs_unchanged':True,'source_sha256':frozen['source_sha256'],
        'source_unchanged':True,'git_before':git_before,'git_after':cached.git_state(),
        'predictions_contract_passed':2*len(DESIGN),'baseline_parity_passed':True,
        'component_pin_geometry_wire_paths_unchanged':True,'blocked_access_attempts':guard.blocked})
    print(json.dumps({'A':a,'B':b,'keep_guards':guards,'Check_allowed':check_allowed,
                      'report':str(OUT/'design/complete.json')}),flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True,choices=('smoke','design'))
    args=parser.parse_args()
    smoke() if args.task=='smoke' else run()
