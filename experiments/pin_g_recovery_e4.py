"""Frozen G + one contact-window repair; original Wire/F are independent factors.

No production edits, model/OCR execution or GT input to inference. Original G
candidate prefix geometry stays identical. Existing box semantics and the G
tip policy process new candidates; downstream output changes are measured.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from pin_g_wire_e1 import (
    verify_formal_state,AccessGuard,load_cases,read_json,write_json,digest,
    PipelineConfig,PipelineContext,signed_frontend,reconstruct_scene,saved_paths,
    build_default_registry,pair_changes,git_state,
)
from pin_contact_band_e4 import append_contact_candidates,POLICY
from pin_tip_evidence import refine_scene_tips
from pin_skeleton_followup import SETS,strict_ids,followup_localization,event_lookup
from pin_g_frame_e3 import freeze_policy as frozen_frame, suppress_box_frames
from pcb.io import read_image,write_image
from pcb.submission import validate_strict
from evaluate_v2 import evaluate_case,aggregate,source_from_name
from tools.run_box_skeleton_experiment import metrics as localization_metrics
from tools.evaluate_box_text_mask_counterfactual import _strict_box_pairs,_gt_pins
from tools.run_box_skeleton_experiment import component_payload
from local_experiments.pin_skeleton_recovery.runner import constrained_matches
import cv2
import numpy as np

OUT=ROOT/'reports/pin_g_recovery_e4'
VARIANTS=('G','GF','R','RF')


def freeze():
    files=('experiments/pin_contact_band_e4.py','experiments/pin_g_recovery_e4.py',
           'experiments/pin_tip_evidence.py','experiments/pin_skeleton_followup.py',
           'tests/pin/test_pin_contact_band_e4.py')
    payload={'policy':asdict(POLICY),'source_sha256':{f:digest(ROOT/f) for f in files},
         'frame_policy':frozen_frame(),'changed_factor':'contact window only',
         'no_GT_or_Check_threshold_fitting':True,
         'G_prefix_geometry_fixed':True,'new_candidates_use_existing_G_joint_tip_policy':True,
         'semantics_algorithm_unchanged_inputs_may_change_assignments':True}
    path=OUT/'frozen_policy.json'
    if path.exists():
        if read_json(path)!=payload:raise AssertionError('Frozen E4 policy changed; do not retune after Check')
    else:write_json(path,payload)
    return payload


def apply_G_to_new(scene,localization,raw,roles,image):
    """Reuse G for new tips only; restore every saved G-prefix coordinate exactly."""
    updated,loc,decisions=refine_scene_tips(scene,localization,roles,image,'joint')
    for comp,(owner,terms),block in zip(updated.components,loc.terminals,raw):
        if comp.key!=owner.key or comp.key!=block['component']:
            raise AssertionError('Candidate owner mismatch')
        for i,old in enumerate(block['terminals']):
            terms[i]=copy.deepcopy(old)
            comp.pins[i].tip=tuple(old['tip'])
        if terms[:len(block['terminals'])]!=block['terminals']:
            raise AssertionError('G prefix changed')
    counts={b['component']:len(b['terminals']) for b in raw}
    seen=Counter()
    for event in updated.diagnostics['pin_events']:
        key=event['component'];i=seen[key];seen[key]+=1
        c=next(c for c in updated.components if c.key==key)
        event['tip']=c.pins[i].tip
        event['recovery_e4_new_candidate']=i>=counts[key]
    for d in decisions:
        if d['candidate_index']<counts[d['component']]:
            d.update(moved=False,reason='saved_G_prefix_geometry_preserved')
            d['new_tip']=d['old_tip']
    return updated,loc,decisions


def dump_raw(output):
    return [{'component':c.key,'type':c.type,'terminals':terms} for c,terms in output.terminals]


def added_attribution(front,old,new,diagnostics,target,height,prediction):
    mapping=dict(_strict_box_pairs(component_payload(front,height),target['components']))
    events=event_lookup(new,diagnostics)
    rows=[]
    for before,after in zip(old,new):
        if before['type']!='box':continue
        key=after['component'];gkey=mapping.get(key)
        gt=_gt_pins(target,gkey,height) if gkey else []
        m5=constrained_matches(gt,after['terminals'],5)
        m20=constrained_matches(gt,after['terminals'],20)
        look5={i:(r,d) for r,(i,d) in m5.items()};look20={i:(r,d) for r,(i,d) in m20.items()}
        for i in range(len(before['terminals']),len(after['terminals'])):
            term=after['terminals'][i];event=events[key][i]
            match=look5.get(i,look20.get(i))
            record={'component':key,'candidate_index':i,**term,'number':event.get('number'),
                'pinname':event.get('pinname'),'exportable':event.get('exportable',True),
                'matched_GT':match[0] if match else None,'distance':match[1] if match else None,
                'geometry_5':i in look5,'geometry_20':i in look20,'strict_component_matched':bool(gkey)}
            if not gkey:record['bucket']='COMPONENT_NOT_STRICT_MATCHED'
            elif match is None:record['bucket']='NO_OWNER_SIDE_MATCH_20'
            elif i not in look5:record['bucket']='POINT_OFFSET_5_TO_20'
            else:
                gtkey=match[0][len(gkey)+1:]
                reference=target['pins'][gkey][gtkey]
                numberkey='pin_'+str(event.get('number',''))
                exported=prediction['pins'][key].get(numberkey)
                record.update(gt_key=gtkey,gt_name=reference['pinname'],key_match=gtkey==numberkey,
                              name_match=reference['pinname']==event.get('pinname',''))
                if not record['exportable'] or exported is None:record['bucket']='EXPORT_ABSTAINED'
                elif gtkey!=numberkey or not record['name_match']:record['bucket']='SEMANTIC_MISMATCH'
                else:record['bucket']='STRICT_TP'
            rows.append(record)
    return rows


def small_overlay(image,front,old,new,diag,path):
    decisions=[d for d in diag['decisions'] if d['added_count']]
    if not decisions:return
    chosen=max(decisions,key=lambda d:d['added_count'])
    c=next(c for c in front.components if c.key==chosen['component'])
    a,b,e,f=c.body_bbox or c.bbox
    xa,ya=max(0,round(a-80)),max(0,round(b-20));xb,yb=min(image.shape[1],round(e+80)),min(image.shape[0],round(f+20))
    panel=image[ya:yb,xa:xb].copy()
    original=next(r for r in old if r['component']==c.key)['terminals']
    after=next(r for r in new if r['component']==c.key)['terminals']
    for i,t in enumerate(after):
        p=(round(t['tip'][0]-xa),round(t['tip'][1]-ya))
        cv2.circle(panel,p,3,(255,0,0) if i<len(original) else (0,0,255),1)
    cv2.putText(panel,c.key+' blue:G red:new contact',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    write_image(path,panel)


def run(task):
    start=time.perf_counter();state=verify_formal_state();before=git_state();frozen=freeze()
    if task=='check':
        design=read_json(OUT/'design/latest.json')
        if not design['guards']['candidate_repair_passed']:
            raise AssertionError('Design candidate repair failed; do not run Check')
        if design['metadata']['frozen_policy']!=frozen:raise AssertionError('Policy drift')
    guard=AccessGuard();cases=load_cases();keys=SETS[task]
    config=PipelineConfig.load(ROOT/'configs/current.json');model=digest(ROOT/'models/component_yolo11n_continue_v2_best.pt')
    registry=build_default_registry();expected={r['case_id']:r['target_sha256'] for r in read_json(ROOT/'reports/pin_g_wire_e1/e0/latest.json')['cases']}
    folder=OUT/task/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True,exist_ok=False)
    evaluations={v:[] for v in VARIANTS};loc_totals={v:Counter() for v in ('G','R')}
    allids={v:set() for v in VARIANTS};inputs={};records=[];new_records=[];pair_counts={v:Counter() for v in ('GF_vs_G','R_vs_G','RF_vs_GF','RF_vs_R')}
    for key in keys:
        case=cases[key];image=read_image(case.image_path);h,w=image.shape[:2]
        front=signed_frontend(case,config,model);context=PipelineContext(case.image_path.name,w,h,config,None)
        source=saved_paths(task,key)['G'];reference=read_json(source/'result.json');raw=read_json(source/'raw_terminals.json')
        for name in ('result.json','raw_terminals.json','diagnostics.json'):inputs[str(source/name)]=digest(source/name)
        scene_g,loc_g=reconstruct_scene(front,raw,image,context)
        new_raw,recovery=append_contact_candidates(image,front.components,front.texts,raw)
        if recovery['added_count']:
            scene_r,loc_r=reconstruct_scene(front,new_raw,image,context)
            scene_r,loc_r,tip_decisions=apply_G_to_new(scene_r,loc_r,raw,front.roles,image)
        else:
            scene_r,loc_r=copy.deepcopy(scene_g),copy.deepcopy(loc_g);tip_decisions=[]
        scene_r.diagnostics['pin_contact_band_e4']=recovery
        scene_r.diagnostics['pin_contact_new_tip_decisions']=tip_decisions
        new_raw=dump_raw(loc_r)
        for old,other in zip(raw,new_raw):
            if other['terminals'][:len(old['terminals'])]!=old['terminals']:raise AssertionError('G terminal identity changed')
            if old['type']!='box' and other!=old:raise AssertionError('Nonbox changed')
        scenes={'G':scene_g,'R':scene_r};wire_masks={}
        for v in ('G','R'):
            wire_masks[v]=registry.create('wire','v3').run(image,scenes[v],context)
            label=v+'F';scenes[label]=copy.deepcopy(scenes[v])
            wire_masks[label]=suppress_box_frames(image,scenes[label],wire_masks[v])
        predictions,diags={},{}
        for v in VARIANTS:
            registry.create('topology','v3').run(scenes[v],wire_masks[v],context)
            predictions[v]=registry.create('submission','official').run(scenes[v],context).data
            diags[v]=scenes[v].diagnostics;validate_strict(predictions[v],(w,h))
            if predictions[v]['components']!=reference['components']:raise AssertionError('Component changed')
            for c in front.components:
                if c.type!='box' and predictions[v]['pins'][c.key]!=reference['pins'][c.key]:raise AssertionError('Nonbox Pins changed')
        if predictions['G']!=reference:raise AssertionError('G full JSON replay mismatch')
        old_f=read_json(ROOT/'reports/pin_g_frame_e3/f_only'/task/'latest.json')
        expected_f=read_json(Path(old_f['metadata']['run_directory'])/'predictions/F'/key/'result.json')
        if predictions['GF']!=expected_f:raise AssertionError('Frozen F replay mismatch')
        for v in ('R','RF'):
            write_json(folder/'predictions'/v/key/'result.json',predictions[v])
            write_json(folder/'predictions'/v/key/'diagnostics.json',diags[v])
            write_json(folder/'predictions'/v/key/'raw_terminals.json',new_raw)
        # Both candidate providers and all four predictions complete before GT.
        with guard.evaluator():
            if digest(case.target_path)!=expected[key]:raise AssertionError('Target version drift')
            target=read_json(case.target_path)
            ev={v:evaluate_case(predictions[v],target,diags[v]) for v in VARIANTS}
            ids={v:strict_ids(predictions[v],target) for v in VARIANTS}
            if ids['G']!=ids['GF'] or ids['R']!=ids['RF']:raise AssertionError('Wire changed Pin TP identities')
            loc=followup_localization(front,{'G':loc_g,'R':loc_r},target,h)
            changes={label:pair_changes(predictions[a],predictions[b],target) for label,a,b in
                 (('GF_vs_G','G','GF'),('R_vs_G','G','R'),('RF_vs_GF','GF','RF'),('RF_vs_R','R','RF'))}
            added=added_attribution(front,raw,new_raw,diags['R'],target,h,predictions['R'])
            pin_change={'gained':sorted(ids['R']-ids['G']),'lost':sorted(ids['G']-ids['R'])}
        for v in VARIANTS:
            allids[v].update((key,*i) for i in ids[v]);evaluations[v].append({'case_id':key,'source':source_from_name(case.image_path.name),'status':'ok','evaluation':ev[v]})
        for v in ('G','R'):loc_totals[v].update(loc[v])
        for v in pair_counts:pair_counts[v].update({k:len(values) for k,values in changes[v].items()})
        new_records.extend({'case_id':key,**r} for r in added)
        row={'case_id':key,'source':source_from_name(case.image_path.name),'added_candidates':recovery['added_count'],
             'recovery':recovery,'tip_decisions':tip_decisions,'evaluation':ev,'localization':loc,
             'pin_TP_changes':pin_change,'pair_changes':changes,'G_full_JSON_replay':True,'F_full_JSON_replay':True,
             'G_candidate_prefix_preserved':True,'Component_nonbox_Pins_preserved':True,
             'wire_palette':{v:diags[v]['wire_palette_selected'] for v in ('G','R')},
             'added_buckets':dict(Counter(r['bucket'] for r in added))}
        records.append(row)
        if key in keys[:3] or key=='0087':small_overlay(image,front,raw,new_raw,recovery,folder/f'{key}_new_contacts.png')
        print(json.dumps({'E4_case':key,'added':recovery['added_count'],
              'box_tp5':{v:loc[v]['tp_5'] for v in loc},'box_tp20':{v:loc[v]['tp_20'] for v in loc},
              'Pin_TP':{v:ev[v]['metrics']['Pin']['tp'] for v in ('G','R')},
              'Pin_gained':len(pin_change['gained']),'Pin_lost':len(pin_change['lost']),
              'added_buckets':row['added_buckets']},ensure_ascii=True),flush=True)
    for name,sha in inputs.items():
        if digest(Path(name))!=sha:raise AssertionError('Frozen G input changed')
    ag={v:aggregate(evaluations[v]) for v in VARIANTS}
    gained=allids['R']-allids['G'];lost=allids['G']-allids['R']
    guards={'G_prefix_geometry_unchanged':True,'no_original_strict_Pin_TP_loss':not lost,
        'strict_Pin_micro_F1_not_lower':ag['R']['overall']['metrics']['Pin']['f1']>=ag['G']['overall']['metrics']['Pin']['f1'],
        'box_tp5_not_lower':loc_totals['R']['tp_5']>=loc_totals['G']['tp_5'],
        'box_tp20_improved':loc_totals['R']['tp_20']>loc_totals['G']['tp_20']}
    guards['candidate_repair_passed']=all(guards.values())
    payload={'metadata':{'OFFICIAL_SCORE':False,'task':task,'case_ids':keys,'case_count':len(keys),
       'run_directory':str(folder),'frozen_policy':frozen,'formal_state':verify_formal_state(),'git_before':before,
       'input_sha256':inputs,'sealed_holdout_used':False,'full_150_executed':False,'models_or_OCR_executed':False,
       'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts,
       'seconds':time.perf_counter()-start,'Check_reused_development_regression_not_unseen_holdout':True,
       'candidate_increase_allowed_only_box':True,'formal_default_modified':False},
       'end_to_end':ag,'localization':{v:localization_metrics(dict(loc_totals[v])) for v in loc_totals},
       'strict_Pin_change':{'gained':sorted(gained),'lost':sorted(lost)},'guards':guards,
       'pair_change_counts':{v:dict(c) for v,c in pair_counts.items()},
       'added_candidates':len(new_records),'added_candidate_buckets':dict(Counter(r['bucket'] for r in new_records)),
       'strict_schema_pass_count':4*len(keys),'cases':records,'new_candidate_records':new_records}
    assert len(new_records)==sum(r['added_candidates'] for r in records)
    write_json(folder/'report.json',payload);write_json(OUT/task/'latest.json',payload)
    print(json.dumps({'E4_finished':task,'guards':guards,'added':len(new_records),
                      'Pin_TP_gained':len(gained),'Pin_TP_lost':len(lost),'report':str(folder/'report.json')},ensure_ascii=True),flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=('design','check'),required=True)
    run(parser.parse_args().task)
