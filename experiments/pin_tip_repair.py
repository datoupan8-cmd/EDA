"""Local Pin regression investigation. Frozen production algorithms and inputs."""
from __future__ import annotations
import argparse
from collections import deque,Counter
import copy
from datetime import datetime
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pin_tip_regression import (load_cases,reconstruct_scene,OUT as PRIOR,PREVIOUS,
                               read_json,write_json,verify_formal_state,AccessGuard,
                               PipelineConfig,PipelineContext,signed_frontend,digest,
                               build_default_registry)
from pcb.io import read_image,write_image
from pcb.submission import validate_strict
from evaluate_v2 import evaluate_case,aggregate,source_from_name
from pcb.wire import trace_paths
import cv2
import numpy as np

OUT=ROOT/'reports/pin_tip_repair'


def saved_paths(subset,key):
    old=Path(read_json(PREVIOUS/('text_'+subset)/'latest.json')['metadata']['run_directory'])/'predictions'
    new=Path(read_json(PRIOR/subset/'latest.json')['metadata']['run_directory'])/'predictions'
    return {'N':old/'N'/key,'G':new/'G'/key}


def cc_at(mask,label,point):
    x,y=map(round,point);h,w=mask.shape
    return sorted(set(int(v) for v in label[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)].ravel())-{0})


def shortest_path(adjacency,a,b):
    queue=deque([a]);previous={a:None}
    while queue:
        point=queue.popleft()
        if point==b:
            result=[]
            while point is not None:result.append(point);point=previous[point]
            return result[::-1]
        for neighbor in adjacency.get(point,[]):
            if neighbor not in previous:previous[neighbor]=point;queue.append(neighbor)
    return []


def audit():
    verify_formal_state();guard=AccessGuard();cases=load_cases()
    config=PipelineConfig.load(ROOT/'configs/current.json')
    model_sha=digest(ROOT/'models/component_yolo11n_continue_v2_best.pt')
    registry=build_default_registry()
    folder=OUT/'audit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    key='0011';case=cases[key];image=read_image(case.image_path);h,w=image.shape[:2]
    front=signed_frontend(case,config,model_sha);context=PipelineContext(case.image_path.name,w,h,config,None)
    saved=saved_paths('check',key)
    scenes={};wires={};traces={};predictions={};metrics={}
    for v in ('N','G'):
        scene,_=reconstruct_scene(front,read_json(saved[v]/'raw_terminals.json'),image,context)
        wires[v]=registry.create('wire','v3').run(image,scene,context)
        scenes[v]=scene;traces[v]=trace_paths(wires[v].mask)
    combos={'NN':('N','N'),'GG':('G','G'),'NG':('N','G'),'GN':('G','N')}
    for label,(wire_input,pin_input) in combos.items():
        scene=copy.deepcopy(scenes[pin_input])
        for field,value in scenes[wire_input].diagnostics.items():
            if field.startswith(('wire_','terminal_corridor','suppressed_','component_barrier')):
                scene.diagnostics[field]=copy.deepcopy(value)
        registry.create('topology','v3').run(scene,wires[wire_input],context)
        pred=registry.create('submission','official').run(scene,context).data
        validate_strict(pred,(w,h));predictions[label]=pred
        if label in ('NN','GG') and pred!=read_json(saved[wire_input]/'result.json'):
            raise AssertionError('Saved JSON parity failed')
        write_json(folder/label/'result.json',pred)
        write_json(folder/label/'diagnostics.json',scene.diagnostics)
        if label in ('NN','GG'):
            mask=wires[wire_input].mask
            _,cc=cv2.connectedComponents(mask,8)
            snaps={s['pin']:s for s in scene.diagnostics['snaps']}
            refs=('U2.4','U2.9');paths,sk,adj=traces[wire_input]
            snapped={ref:snaps[ref] for ref in refs}
            cc_ids={ref:cc_at(mask,cc,snaps[ref]['to']) for ref in refs}
            points=[]
            for ref in refs:
                q=snaps[ref]['to'];path=paths[snaps[ref]['path']]
                points.append(min(path,key=lambda p:(p[1]-q[0])**2+(p[0]-q[1])**2))
            route=shortest_path(adj,*points)
            metrics[label]={'snaps':snapped,'mask_cc_ids':cc_ids,
                            'same_mask_cc':bool(set(cc_ids[refs[0]])&set(cc_ids[refs[1]])),
                            'skeleton_route_exists':bool(route),'skeleton_route_length':len(route),
                            'route_junctions':[{'point':(p[1],p[0]),'degree':len(adj[p])} for p in route if len(adj[p])>=3],
                            'nonconnecting_crossings':scene.diagnostics['nonconnecting_crossings'],
                            'palette':scene.diagnostics['wire_palette_selected']}
            if route:write_json(folder/label/'route.json',[(p[1],p[0]) for p in route])
    with guard.evaluator():
        gt=read_json(case.target_path)
        evals={v:evaluate_case(predictions[v],gt,read_json(folder/v/'diagnostics.json')) for v in combos}
    component=next(c for c in front.components if c.key=='U2');x1,y1,x2,y2=component.body_bbox or component.bbox
    xa=max(0,round(x1-130));xb=min(w,round(x2+35));ya=max(0,round(y1-30));yb=min(h,round(y2+40))
    panels=[image[ya:yb,xa:xb].copy()]
    for v in ('N','G'):
        canvas=cv2.cvtColor(wires[v].mask[ya:yb,xa:xb],cv2.COLOR_GRAY2BGR)
        for ref,snap in metrics[v+v]['snaps'].items():
            q=(round(snap['to'][0]-xa),round(snap['to'][1]-ya))
            cv2.circle(canvas,q,5,(0,0,255),1)
            cv2.putText(canvas,ref,q,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
        panels.append(canvas)
    diff=wires['N'].mask[ya:yb,xa:xb]!=wires['G'].mask[ya:yb,xa:xb]
    canvas=image[ya:yb,xa:xb].copy();canvas[diff]=(0,0,255);panels.append(canvas)
    for index,panel in enumerate(panels):
        panels[index]=cv2.copyMakeBorder(panel,25,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(panels[index],('Image','N wire mask','G wire mask','Red: changed pixels')[index],(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
    write_image(folder/'0011_U2_wire_trace.png',np.concatenate(panels,axis=1))
    payload={'metadata':{'case_ids':[key],'OFFICIAL_SCORE':False,'sealed_holdout_used':False,
                         'diagnostic_counterfactual_only':True,'run_directory':str(folder),
                         'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts,
                         'formal_state':verify_formal_state()},'trace':metrics,'evaluation':evals,
             'combination_definitions':combos,'mask_difference_pixels':int((wires['N'].mask!=wires['G'].mask).sum())}
    write_json(folder/'report.json',payload);write_json(OUT/'audit/latest.json',payload)
    print({'trace':metrics,'pair_metrics':{v:e['metrics']['PinPair'] for v,e in evals.items()},'folder':str(folder)},flush=True)
    return payload


def run_local_geometry(subset):
    """One frozen local-length policy; Check is a known regression set, not holdout."""
    from pin_tip_local_geometry import refine_existing_changes,PARAMETERS
    from pin_skeleton_followup import followup_localization,strict_ids,PinLocalizationOutput
    from pin_tip_regression import pin_semantic_payload
    from pin_tip_regression_finalize import geometry_guard
    from tools.run_box_skeleton_experiment import metrics
    verify_formal_state();guard=AccessGuard();cases=load_cases()
    config=PipelineConfig.load(ROOT/'configs/current.json')
    model_sha=digest(ROOT/'models/component_yolo11n_continue_v2_best.pt')
    registry=build_default_registry();prior=read_json(PRIOR/subset/'latest.json')
    folder=OUT/subset/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True,exist_ok=False)
    if subset=='check':
        design=read_json(OUT/'design/latest.json')
        if design['metadata']['policy_source_sha256']!=digest(ROOT/'experiments/pin_tip_local_geometry.py'):
            raise AssertionError('Policy changed after Design')
    evaluations={v:[] for v in ('N','G','H')};counts={v:Counter() for v in evaluations}
    tp_sets={v:set() for v in evaluations};records=[];reused=0
    for key in prior['metadata']['case_ids']:
        case=cases[key];image=read_image(case.image_path);h,w=image.shape[:2]
        front=signed_frontend(case,config,model_sha);context=PipelineContext(case.image_path.name,w,h,config,None)
        saved=saved_paths(subset,key)
        raws={v:read_json(saved[v]/'raw_terminals.json') for v in ('N','G')}
        scene_g,loc_g=reconstruct_scene(front,raws['G'],image,context)
        _,loc_n=reconstruct_scene(front,raws['N'],image,context)
        scene_h,loc_h,decisions=refine_existing_changes(scene_g,loc_g,loc_n,image)
        raw_h=[{'component':c.key,'type':c.type,'terminals':r} for c,r in loc_h.terminals]
        geometry_guard(raws['N'],raw_h)
        preds={v:read_json(saved[v]/'result.json') for v in ('N','G')}
        diags={v:read_json(saved[v]/'diagnostics.json') for v in ('N','G')}
        if not any(d['changed'] for d in decisions):
            preds['H']=copy.deepcopy(preds['G']);diags['H']=copy.deepcopy(diags['G']);reused+=1
            diags['H']['pin_tip_local_geometry']=scene_h.diagnostics['pin_tip_local_geometry']
        else:
            wire=registry.create('wire','v3').run(image,scene_h,context)
            registry.create('topology','v3').run(scene_h,wire,context)
            preds['H']=registry.create('submission','official').run(scene_h,context).data
            diags['H']=scene_h.diagnostics
        validate_strict(preds['H'],(w,h))
        if pin_semantic_payload(preds['H'])!=pin_semantic_payload(preds['G']) or preds['H']['components']!=preds['G']['components']:
            raise AssertionError('Semantics/component changed')
        dest=folder/'predictions/H'/key
        write_json(dest/'result.json',preds['H']);write_json(dest/'diagnostics.json',diags['H'])
        write_json(dest/'raw_terminals.json',raw_h)
        with guard.evaluator():
            target=read_json(case.target_path)
            local=followup_localization(front,{'N':loc_n,'G':loc_g,'H':loc_h},target,h)
            pin_counts={};pair_counts={}
            for v in evaluations:
                e=evaluate_case(preds[v],target,diags[v]);pin_counts[v]=e['metrics']['Pin']['tp']
                pair_counts[v]=e['metrics']['PinPair']['tp']
                evaluations[v].append({'case_id':key,'source':source_from_name(case.image_path.name),
                                        'status':'ok','evaluation':e})
                counts[v].update(local[v]);tp_sets[v].update((key,*r) for r in strict_ids(preds[v],target))
            records.append({'case_id':key,'strict_tp':pin_counts,'pair_tp':pair_counts,
                            'localization':local,'decisions':decisions,
                            'moved_count':sum(d['changed'] for d in decisions)})
        print({'case':key,'tp':pin_counts,'pair_tp':pair_counts,'box_tp20':{v:local[v]['tp_20'] for v in local},
               'changed_tips':sum(d['changed'] for d in decisions)},flush=True)
    payload={'metadata':{'case_ids':prior['metadata']['case_ids'],'OFFICIAL_SCORE':False,'sealed_holdout_used':False,
                        'run_directory':str(folder),'subset':subset,'reused_G_count':reused,
                        'parameters':PARAMETERS,'policy_source_sha256':digest(ROOT/'experiments/pin_tip_local_geometry.py'),
                        'check_is_previous_development_regression_set':True,
                        'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts,
                        'formal_state':verify_formal_state()},
             'end_to_end':{v:aggregate(rows) for v,rows in evaluations.items()},
             'localization':{v:metrics(dict(c)) for v,c in counts.items()},
             'tp_changes':{'gained_vs_G':sorted(tp_sets['H']-tp_sets['G']),'lost_vs_G':sorted(tp_sets['G']-tp_sets['H']),
                           'gained_vs_N':sorted(tp_sets['H']-tp_sets['N']),'lost_vs_N':sorted(tp_sets['N']-tp_sets['H'])},
             'cases':records}
    write_json(folder/'report.json',payload);write_json(OUT/subset/'latest.json',payload)
    print({'report':str(OUT/subset/'latest.json'),
           'pin_tp':{v:p['overall']['metrics']['Pin']['tp'] for v,p in payload['end_to_end'].items()},
           'lost_vs_G':len(payload['tp_changes']['lost_vs_G'])},flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=('audit','design','check'),default='audit')
    args=parser.parse_args()
    if args.task=='audit':audit()
    else:run_local_geometry(args.task)
