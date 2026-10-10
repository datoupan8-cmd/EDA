"""Pin-only regression study; original algorithms/configs remain untouched.

Stage crossover runs are diagnostic input counterfactuals, not deployable fixes.
GT is read only after inference. All inputs are fixed allowed development cases.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pin_skeleton_followup import (
    MANUAL,OUT as PREVIOUS,read_json,write_json,verify_formal_state,selected_cases,
    signed_frontend,digest,AccessGuard,BoxBBoxSemantics,PipelineConfig,
    PipelineContext,PinLocalizationOutput,build_default_registry,
    followup_localization,strict_ids,
)
from pcb.io import read_image,write_image
from pcb.schema import Scene
from pcb.submission import validate_strict
from evaluate_v2 import (component_key_map,normalize_pred_ref,net_members,pair_set,
                         evaluate_case,aggregate,source_from_name)
import numpy as np
import cv2

OUT=ROOT/'reports/pin_tip_regression'


def load_cases():
    dataset=Path(read_json(MANUAL/'outputs/design/latest.json')['metadata']['dataset_root'])
    return {f'{c.case_id:04d}':c for subset in ('design','check') for c in selected_cases(dataset,subset)}


def pin_semantic_payload(pred):
    return {owner:{key:pin['pinname'] for key,pin in pins.items()} for owner,pins in pred['pins'].items()}


def normalized_pairs(pred,target):
    mapping=component_key_map(pred['components'],target['components'])
    return pair_set([{normalize_pred_ref(ref,pred['components'],mapping)
                      for ref in net_members(net)} for net in pred['nets'].values()])


def saved_audit():
    state=verify_formal_state();guard=AccessGuard();cases=load_cases()
    rows=[];examples=[]
    for subset in ('design','check'):
        meta=read_json(PREVIOUS/('text_'+subset)/'latest.json');run=Path(meta['metadata']['run_directory'])
        for key in meta['metadata']['case_ids']:
            pred={v:read_json(run/'predictions'/v/key/'result.json') for v in ('N','TN')}
            diag={v:read_json(run/'predictions'/v/key/'diagnostics.json') for v in ('N','TN')}
            raw={v:read_json(run/'predictions'/v/key/'raw_terminals.json') for v in ('N','TN')}
            if pin_semantic_payload(pred['N'])!=pin_semantic_payload(pred['TN']):
                raise AssertionError('Scorer semantics changed between N/TN')
            for a,b in zip(raw['N'],raw['TN']):
                if a['component']!=b['component'] or len(a['terminals'])!=len(b['terminals']):
                    raise AssertionError('Candidate count/order changed')
                for x,y in zip(a['terminals'],b['terminals']):
                    if any(x[f]!=y[f] for f in ('base','side','method')):
                        raise AssertionError('Candidate identity changed')
            with guard.evaluator():
                target=read_json(cases[key].target_path)
                true=pair_set([net_members(n) for n in target['nets'].values()])
                known={f'{owner}.{pk[4:]}' for owner,pins in target['pins'].items() for pk in pins}
                pairs={v:normalized_pairs(pred[v],target) for v in pred}
                added=pairs['TN']-pairs['N'];false=added-true
                missing=sum(any(ref not in known for ref in pair) for pair in false)
                changed={}
                for v in pred:
                    d=diag[v]
                    changed[v]={k:d.get(k) for k in ('wire_palette_selected','wire_pixels','wire_paths','wire_groups',
                                'terminal_corridor_restored_pixels','unattached_pin_ratio')}
                    changed[v].update(junction_count=len(d.get('junctions',[])),bridge_count=len(d.get('short_bridges',[])))
                row={'subset':subset,'case_id':key,'exported_semantics_equal':True,
                     'pin_tp':{v:len(strict_ids(pred[v],target)) for v in pred},
                     'pair_tp':{v:len(pairs[v]&true) for v in pred},'pair_pred':{v:len(pairs[v]) for v in pred},
                     'new_false_pairs':len(false),'new_false_pairs_missing_gt_ref':missing,
                     'new_false_pairs_known_but_disconnected':len(false)-missing,
                     'wire':changed,'localization_20':{v:next(c for c in meta['cases'] if c['case_id']==key)['localization'][v]['tp_20'] for v in pred}}
                rows.append(row)
                examples.extend({'case_id':key,'pair':p,'known_refs':all(ref in known for ref in p)} for p in sorted(false)[:6])
    payload={'metadata':{'OFFICIAL_SCORE':False,'sealed_holdout_used':False,'case_count':14,
                        'no_inference_rerun':True,'formal_state':verify_formal_state()},'cases':rows,'false_pair_examples':examples}
    write_json(OUT/'saved_audit.json',payload)
    print(json.dumps({'saved_audit':str(OUT/'saved_audit.json'),
                     'major_cases':[r for r in rows if r['case_id'] in ('0014','0017','0034')]},ensure_ascii=False),flush=True)
    return payload


def reconstruct_scene(front,raw,image,context):
    local=copy.deepcopy(front)
    if [c.key for c in local.components]!=[b['component'] for b in raw]:
        raise AssertionError('Saved owner ordering mismatch')
    loc=PinLocalizationOutput([(c,copy.deepcopy(b['terminals'])) for c,b in zip(local.components,raw)])
    semantic=BoxBBoxSemantics().run(local,loc,context)
    scene=Scene(context.width,context.height,semantic.components,local.texts,
                diagnostics=copy.deepcopy(local.diagnostics))
    scene.diagnostics.update(semantic.diagnostics)
    return scene,loc


def crossovers():
    """2x2 wire-input/snap-input geometry, identical original algorithms."""
    verify_formal_state();guard=AccessGuard();cases=load_cases()
    config=PipelineConfig.load(ROOT/'configs/current.json')
    model_sha=digest(ROOT/'models/component_yolo11n_continue_v2_best.pt')
    registry=build_default_registry();records=[]
    folder=OUT/'crossover'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    definitions={'NN':('N','N'),'TT':('TN','TN'),'NT':('N','TN'),'TN':('TN','N')}
    for key in ('0014','0017'):
        case=cases[key];image=read_image(case.image_path);h,w=image.shape[:2]
        front=signed_frontend(case,config,model_sha)
        context=PipelineContext(case.image_path.name,w,h,config,None)
        previous=read_json(PREVIOUS/'text_design/latest.json')
        saved=Path(previous['metadata']['run_directory'])/'predictions'
        raws={v:read_json(saved/v/key/'raw_terminals.json') for v in ('N','TN')}
        scenes={};wires={}
        for v in ('N','TN'):
            scenes[v],_=reconstruct_scene(front,raws[v],image,context)
            wire_scene=copy.deepcopy(scenes[v])
            wires[v]=registry.create('wire','v3').run(image,wire_scene,context)
            scenes[v].diagnostics.update(wire_scene.diagnostics)
        mask_n=wires['N'].mask>0;mask_t=wires['TN'].mask>0
        diff={'different_pixels':int((mask_n!=mask_t).sum()),
              'intersection_over_union':float((mask_n&mask_t).sum()/max(1,(mask_n|mask_t).sum()))}
        results={};diags={}
        for variant,(wire_geometry,snap_geometry) in definitions.items():
            scene=copy.deepcopy(scenes[snap_geometry])
            for field in scenes[wire_geometry].diagnostics:
                if field.startswith(('wire_','terminal_corridor','suppressed_','component_barrier')):
                    scene.diagnostics[field]=copy.deepcopy(scenes[wire_geometry].diagnostics[field])
            registry.create('topology','v3').run(scene,wires[wire_geometry],context)
            # Diagnostic counterfactual only: separate scoring position from
            # topology input to measure its causal contribution. No Stage code
            # or official JSON contract is changed or promoted.
            if variant=='TN':
                for c,actual in zip(scene.components,scenes['TN'].components):
                    for p,q in zip(c.pins,actual.pins):p.tip=q.tip
            results[variant]=registry.create('submission','official').run(scene,context).data
            validate_strict(results[variant],(w,h));diags[variant]=scene.diagnostics
            if variant in ('NN','TT'):
                reference=read_json(saved/('N' if variant=='NN' else 'TN')/key/'result.json')
                if results[variant]!=reference:raise AssertionError('Reconstructed baseline JSON parity failure')
            write_json(folder/variant/key/'result.json',results[variant])
            write_json(folder/variant/key/'diagnostics.json',diags[variant])
        with guard.evaluator():
            target=read_json(case.target_path)
            evaluations={v:evaluate_case(results[v],target,diags[v]) for v in definitions}
            record={'case_id':key,'mask_difference':diff,'evaluation':evaluations,
                    'palette':{v:scenes[v].diagnostics['wire_palette_selected'] for v in scenes}}
            records.append(record)
        print(json.dumps({'case':key,'mask_diff':diff,
              'pairs':{v:{k:e['metrics']['PinPair'][k] for k in ('tp','pred','f1')} for v,e in evaluations.items()}},ensure_ascii=False),flush=True)
    payload={'metadata':{'OFFICIAL_SCORE':False,'case_ids':['0014','0017'],'sealed_holdout_used':False,
                         'input_counterfactual_only':True,'definitions':definitions,
                         'formal_state':verify_formal_state(),'run_directory':str(folder),
                         'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts},
             'cases':records}
    write_json(folder/'report.json',payload);write_json(OUT/'crossover/latest.json',payload)
    return payload


def run_policies(subset,resume=None):
    from pin_tip_evidence import refine_scene_tips,CONFIDENCE_MIN
    verify_formal_state();guard=AccessGuard();cases=load_cases()
    config=PipelineConfig.load(ROOT/'configs/current.json')
    model_sha=digest(ROOT/'models/component_yolo11n_continue_v2_best.pt')
    registry=build_default_registry()
    previous=read_json(PREVIOUS/('text_'+subset)/'latest.json')
    saved=Path(previous['metadata']['run_directory'])/'predictions'
    folder=resume.resolve() if resume else OUT/subset/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if resume and folder.parent!=(OUT/subset).resolve():raise ValueError('Resume outside task folder')
    if not resume:folder.mkdir(parents=True,exist_ok=False)
    reused_predictions=0
    variant_names=('N','TN','G','E');evaluations={v:[] for v in variant_names}
    tp_sets={v:set() for v in variant_names};records=[];loc_totals={v:Counter() for v in variant_names}
    for key in previous['metadata']['case_ids']:
        case=cases[key];image=read_image(case.image_path);h,w=image.shape[:2]
        front=signed_frontend(case,config,model_sha);context=PipelineContext(case.image_path.name,w,h,config,None)
        raw_n=read_json(saved/'N'/key/'raw_terminals.json')
        scene_n,loc_n=reconstruct_scene(front,raw_n,image,context)
        results={v:read_json(saved/v/key/'result.json') for v in ('N','TN')}
        diagnostics={v:read_json(saved/v/key/'diagnostics.json') for v in ('N','TN')}
        locs={'N':loc_n,'TN':PinLocalizationOutput([(c,copy.deepcopy(b['terminals'])) for c,b in zip(front.components,read_json(saved/'TN'/key/'raw_terminals.json'))])}
        changes={}
        for v,policy in (('G','joint'),('E','endpoint')):
            scene,locs[v],decisions=refine_scene_tips(scene_n,loc_n,front.roles,image,policy)
            changes[v]=decisions
            dest=folder/'predictions'/v/key
            raw_expected=[{'component':c.key,'type':c.type,'terminals':r} for c,r in locs[v].terminals]
            if resume and all((dest/f).exists() for f in ('result.json','diagnostics.json','raw_terminals.json')):
                if read_json(dest/'raw_terminals.json')!=json.loads(json.dumps(raw_expected)):
                    raise AssertionError('Saved policy geometry changed')
                results[v]=read_json(dest/'result.json');diagnostics[v]=read_json(dest/'diagnostics.json')
                reused_predictions+=1
            else:
                wire=registry.create('wire','v3').run(image,scene,context)
                registry.create('topology','v3').run(scene,wire,context)
                results[v]=registry.create('submission','official').run(scene,context).data
                diagnostics[v]=scene.diagnostics
                write_json(dest/'result.json',results[v])
                write_json(dest/'diagnostics.json',diagnostics[v])
                write_json(dest/'raw_terminals.json',raw_expected)
            validate_strict(results[v],(w,h))
            if pin_semantic_payload(results[v])!=pin_semantic_payload(results['N']):
                raise AssertionError('Refinement changed exported number/name/key')
            if results[v]['components']!=results['N']['components']:raise AssertionError('Component changed')
            for comp in front.components:
                if comp.type!='box' and results[v]['pins'][comp.key]!=results['N']['pins'][comp.key]:
                    raise AssertionError('Non-box Pin JSON changed')
        with guard.evaluator():
            target=read_json(case.target_path)
            localization=followup_localization(front,locs,target,h)
            counts={}
            for v in variant_names:
                ev=evaluate_case(results[v],target,diagnostics[v])
                evaluations[v].append({'case_id':key,'source':source_from_name(case.image_path.name),'status':'ok','evaluation':ev})
                ids=strict_ids(results[v],target);tp_sets[v].update((key,*r) for r in ids)
                counts[v]=len(ids)
                loc_totals[v].update(localization[v])
            records.append({'case_id':key,'strict_tp':counts,'localization':localization,
                            'changes':{v:{'moved':sum(r['moved'] for r in values),'reasons':dict(Counter(r['reason'] for r in values))} for v,values in changes.items()},
                            'policy_decisions':changes})
        print(json.dumps({'case':key,'strict_tp':counts,'box_tp5':{v:localization[v]['tp_5'] for v in variant_names},
                         'box_tp20':{v:localization[v]['tp_20'] for v in variant_names}},ensure_ascii=False),flush=True)
    from tools.run_box_skeleton_experiment import metrics
    payload={'metadata':{'OFFICIAL_SCORE':False,'subset':subset,'case_ids':previous['metadata']['case_ids'],
            'parameters':{'joint_confidence_min':CONFIDENCE_MIN,'tip_formula':'existing max(6,6*scale)',
                          'endpoint_profile':'existing P3 normal support, 20*scale, span>=8*scale, occupancy>=0.75'},
            'all_params_frozen_before_check':True,'sealed_holdout_used':False,
            'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts,
            'formal_state':verify_formal_state(),'run_directory':str(folder),'reused_prediction_count':reused_predictions},
            'end_to_end':{v:aggregate(r) for v,r in evaluations.items()},
            'localization':{v:metrics(dict(r)) for v,r in loc_totals.items()},
            'tp_changes':{v:{'gained_vs_N':sorted(tp_sets[v]-tp_sets['N']),'lost_vs_N':sorted(tp_sets['N']-tp_sets[v]),
                           'gained_vs_TN':sorted(tp_sets[v]-tp_sets['TN']),'lost_vs_TN':sorted(tp_sets['TN']-tp_sets[v])} for v in variant_names},
            'cases':records}
    write_json(folder/'report.json',payload);write_json(OUT/subset/'latest.json',payload)
    lines=[f'# Pin Tip最小回归实验：{subset}','','本地非官方诊断；不是150例分数。','',
        '|组|严格Pin TP|Pin宏F1|box TP5|box TP20|Pair TP|Pair预测|Pair宏F1|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in variant_names:
        m=payload['end_to_end'][v]['overall']['metrics'];loc=payload['localization'][v]
        lines.append(f"|{v}|{m['Pin']['tp']}|{m['Pin']['macro_f1']:.6f}|{loc['tp_5']}|{loc['tp_20']}|{m['PinPair']['tp']}|{m['PinPair']['pred']}|{m['PinPair']['macro_f1']:.6f}|")
    lines+=['','N：上一轮BBox脚号距离+原V3 Tip；TN：全部box统一短Tip；G：可信联合语义+连续线段才短Tip；E：有可靠局部末端才改Tip。',
            '无GT参数或case规则，无新增/删减candidate，无角色/编号/脚名改动。Wire/Topology/正式配置保持不变。']
    (OUT/subset/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"REPORT: {OUT/subset/'summary.md'}",flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=('audit','crossover','design','check'),default='audit')
    parser.add_argument('--resume-directory',type=Path)
    args=parser.parse_args()
    if args.task=='audit':saved_audit()
    elif args.task=='crossover':crossovers()
    else:run_policies(args.task,args.resume_directory)
