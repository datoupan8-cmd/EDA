"""Frozen-front-end four-way box localization / full-pipeline experiment.

Inference completes before target reads. Frontend snapshots contain only
image-derived Components/Text/roles, validated against image/config/model
hashes. Development cases 0001..0150 are the only permitted dataset inputs.
"""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from contextlib import contextmanager

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_v2 import aggregate, evaluate_case, prf, source_from_name
from pcb.component.v4 import ComponentStageV4
from pcb.component_detector_yolo import YoloComponentDetector
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import ComponentStageOutput
from pcb.core.registry import build_default_registry
from pcb.data_policy import assert_allowed_case_id, discover_allowed_cases
from pcb.io import read_image, write_json, write_image
from pcb.ocr_backends import HybridOCR, TiledEasyOCR
from pcb.pin.localization.box_skeleton_experiment import CONFIG, preserve_strokes, raw_ink, text_mask
from pcb.schema import Component, Scene, Text
from pcb.text.stage import CurrentTextStage
from pcb.text_detection import TokenRole
from pcb.vision import OCR
from pcb.submission import validate_strict
from tools.evaluate_box_text_mask_counterfactual import _gt_pins, _strict_box_pairs
from tools.analyze_box_terminal_stage_trace import _match

VARIANTS = {'A': 'v3', 'B': 'box_scan_preserve', 'C': 'box_skeleton_masked', 'D': 'box_skeleton_preserve'}
RADII = (5.,10.,20.)
DESIGN = {'0014','0017','0018','0087','0103','0130'}
CHECK = {'0011','0034','0046','0095','0111','0129','0139','0143'}


class AccessGuard:
    def __init__(self):
        self.evaluation = False
        self.target_attempts = 0
        self.sealed_attempts = 0
        sys.addaudithook(self._audit)

    def _audit(self, event, args):
        if event != 'open' or not args or not isinstance(args[0],(str,bytes,Path)):
            return
        value = str(args[0]).replace('\\','/').lower()
        parts = value.split('/')
        if any(p in {'10_gtcase','10gt','golden','eda_pin_crossing_quicktest'} for p in parts):
            self.sealed_attempts += 1
            raise PermissionError('Sealed / QuickTest access blocked')
        if '200_train_cases' in parts:
            index = parts.index('200_train_cases')
            if index+1 < len(parts):
                case_id = int(parts[index+1])
                if not 1 <= case_id <= 150:
                    self.sealed_attempts += 1
                assert_allowed_case_id(case_id)
        if value.endswith('_target.json') and not self.evaluation:
            self.target_attempts += 1
            raise PermissionError('Target read during inference blocked')

    @contextmanager
    def evaluator(self):
        self.evaluation = True
        try:
            yield
        finally:
            self.evaluation = False


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def load_frontend(case, image, context, cache, runtime):
    signature = {'image_sha256': digest(case.image_path), 'config': context.config.as_dict(),
                 'model_sha256': runtime['model_sha256'], 'frontend_source_sha256': runtime['frontend_source_sha256']}
    path = cache / f'{case.case_id:04d}.json'
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('signature') == signature:
            components = [Component(**{**r,'bbox':tuple(r['bbox']),'body_bbox':tuple(r['body_bbox']) if r.get('body_bbox') else None}) for r in value['components']]
            texts = [Text(**{**r,'bbox':tuple(r['bbox'])}) for r in value['texts']]
            roles = [TokenRole(**{**r,'token':Text(**{**r['token'],'bbox':tuple(r['token']['bbox'])})}) for r in value['roles']]
            return ComponentStageOutput(components,texts,roles,value['diagnostics']), True
    if runtime.get('ocr') is None:
        runtime['ocr'] = HybridOCR(TiledEasyOCR(ROOT/'runs/ocr_cache_v4_1',ROOT/'models/easyocr',device='auto',allow_download=False),OCR(ROOT/'runs/ocr_cache_v4_1'))
        runtime['detector'] = YoloComponentDetector(ROOT/'models/component_yolo11n_continue_v2_best.pt','auto',cache_dir=ROOT/'runs/yolo_cache')
    context.ocr, context.detector = runtime['ocr'], runtime['detector']
    front = ComponentStageV4().run(image,CurrentTextStage().run(image,context),context)
    write_json(path, {'signature':signature,'components':[asdict(c) for c in front.components], 'texts':[asdict(t) for t in front.texts], 'roles':[asdict(r) for r in front.roles], 'diagnostics':front.diagnostics})
    return front, False


def component_payload(front,height):
    from tools.analyze_pin_localization import predicted_component_payload
    return {c.key:predicted_component_payload(c,height) for c in front.components}


def empty_counts():
    return {'components':0,'gt':0,'pred':0,'tp_5':0,'tp_10':0,'tp_20':0,'absolute_count_error':0,'exact_count':0}


def add_counts(total, item):
    for key in total:
        total[key] += item[key]


def metrics(counts):
    return {**counts, **{f'at_{int(r)}':prf(counts[f'tp_{int(r)}'],counts['pred'],counts['gt']) for r in RADII},
            'pin_count_mae':counts['absolute_count_error']/counts['components'] if counts['components'] else None,
            'exact_pin_count_accuracy':counts['exact_count']/counts['components'] if counts['components'] else None}


def localization_case(front, outputs, target, height):
    pairs = _strict_box_pairs(component_payload(front,height),target['components'])
    lookup = {v:{c.key:rows for c,rows in out.terminals} for v,out in outputs.items()}
    totals = {v:empty_counts() for v in VARIANTS}
    per_side = {v:{s:empty_counts() for s in ('left','right','top','bottom')} for v in VARIANTS}
    per_component=[]
    for pk,gk in pairs:
        pins = _gt_pins(target,gk,height)
        for v in VARIANTS:
            candidates=lookup[v][pk]
            counts=empty_counts()
            counts.update(components=1,gt=len(pins),pred=len(candidates),absolute_count_error=abs(len(pins)-len(candidates)),exact_count=int(len(pins)==len(candidates)))
            matches={r:_match(pins,candidates,radius=r) for r in RADII}
            for radius,match in matches.items():
                counts[f'tp_{int(radius)}']=len(match)
            add_counts(totals[v],counts)
            for side in per_side[v]:
                side_gt=[p for p in pins if p['side']==side]
                side_pred=[p for p in candidates if p['side']==side]
                block=empty_counts();block.update(components=1,gt=len(side_gt),pred=len(side_pred),absolute_count_error=abs(len(side_gt)-len(side_pred)),exact_count=int(len(side_gt)==len(side_pred)))
                for radius in RADII:
                    block[f'tp_{int(radius)}']=len(_match(side_gt,side_pred,radius=radius))
                add_counts(per_side[v][side],block)
            per_component.append({'component':pk,'variant':v,**counts})
    return totals,per_side,per_component


def full_inference(image, front, loc, context, registry):
    semantics=registry.create('pin_semantics',context.config.pin_semantics).run(front,loc,context)
    scene=Scene(context.width,context.height,semantics.components,front.texts,diagnostics=copy.deepcopy(front.diagnostics))
    scene.diagnostics.update(semantics.diagnostics)
    scene.diagnostics.update(loc.diagnostics)
    wire=registry.create('wire',context.config.wire).run(image,scene,context)
    registry.create('topology',context.config.topology).run(scene,wire,context)
    result=registry.create('submission',context.config.submission).run(scene,context).data
    return result,scene.diagnostics


def debug_case(image,front,outputs,target,height,outdir):
    pairs=_strict_box_pairs(component_payload(front,height),target['components'])
    if not pairs:
        return
    gt_by_component={p:_gt_pins(target,g,height) for p,g in pairs}
    for v,output in outputs.items():
        panel=image.copy()
        for comp,rows in output.terminals:
            if comp.type!='box':continue
            x1,y1,x2,y2=map(lambda x:int(round(x)),comp.body_bbox or comp.bbox)
            cv2.rectangle(panel,(x1,y1),(x2,y2),(210,100,0),1)
            for pin in gt_by_component.get(comp.key,[]):
                cv2.circle(panel,tuple(int(round(x)) for x in pin['point']),4,(0,180,0),1)
            for t in rows:
                base=tuple(int(round(x)) for x in t['base']);tip=tuple(int(round(x)) for x in t['tip'])
                cv2.line(panel,base,tip,(0,0,220),1);cv2.circle(panel,tip,2,(0,0,220),-1)
        write_image(outdir/f'{v}_terminals.png',panel)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--case-ids',default=None)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('localization','full'),default='localization')
    p.add_argument('--frontend-cache',type=Path,default=ROOT.parent/'tmp/pin_sina_frontend')
    p.add_argument('--prediction-root',type=Path,default=ROOT.parent/'tmp/pin_sina_predictions')
    p.add_argument('--debug-case-ids',default='0002,0014,0087,0034')
    args=p.parse_args(argv)
    config=PipelineConfig.load(ROOT/'configs/current.json')
    if config != PipelineConfig.current():raise AssertionError('Formal current config changed')
    guard=AccessGuard(); started=time.perf_counter()
    selected={int(n) for n in args.case_ids.split(',')} if args.case_ids else set(range(1,151))
    for number in selected:assert_allowed_case_id(number)
    cases=[c for c in discover_allowed_cases(args.dataset_root) if c.case_id in selected]
    if len(cases)!=len(selected):raise AssertionError('Case selection incomplete')
    source_files=[ROOT/'pcb/component_detection_v4.py',ROOT/'pcb/component/v4.py',ROOT/'pcb/text_detection.py',ROOT/'pcb/component_text_assignment.py',ROOT/'pcb/ocr_backends.py',ROOT/'pcb/text/stage.py']
    runtime={'model_sha256':digest(ROOT/'models/component_yolo11n_continue_v2_best.pt'), 'frontend_source_sha256':{str(f.relative_to(ROOT)):digest(f) for f in source_files}}
    registry=build_default_registry()
    all_rows=[];evaluations={v:[] for v in VARIANTS};failures=[];baseline_parity=0;nonbox_parity=0
    for case in cases:
        case_id=f'{case.case_id:04d}'
        image=read_image(case.image_path);height,width=image.shape[:2]
        context=PipelineContext(case.image_path.name,width,height,config,None)
        front,hit=load_frontend(case,image,context,args.frontend_cache,runtime)
        outputs={v:registry.create('pin_localization',name).run(image,copy.deepcopy(front),context) for v,name in VARIANTS.items()}
        for v in ('B','C','D'):
            for (ac,ar),(bc,br) in zip(outputs['A'].terminals,outputs[v].terminals):
                if ac.key!=bc.key or ac.body_bbox!=bc.body_bbox:raise AssertionError('Component changed')
                if ac.type!='box' and ar!=br:raise AssertionError('Non-box candidate changed')
        nonbox_parity+=1
        results={};diags={}
        if args.mode=='full':
            for v in VARIANTS:
                local_front=copy.deepcopy(front)
                # Rebind row owners to this independent component copy.
                loc=copy.deepcopy(outputs[v]);loc.terminals=[(c,loc.terminals[i][1]) for i,c in enumerate(local_front.components)]
                t=time.perf_counter()
                results[v],diags[v]=full_inference(image,local_front,loc,replace(context,config=replace(config,pin_localization=VARIANTS[v])),registry)
                diags[v]['seconds']=time.perf_counter()-t
                validate_strict(results[v],(width,height))
                folder=args.prediction_root/v/case_id
                write_json(folder/'result.json',results[v]);write_json(folder/'diagnostics.json',diags[v])
            reference=ROOT.parent/'tmp/pin_p6_5_formal_predictions'/case_id/'result.json'
            if reference.exists():
                if json.loads(reference.read_text(encoding='utf-8'))!=results['A']:raise AssertionError(f'{case_id}: baseline prediction parity failed')
                baseline_parity+=1
        # No GT was opened during frontend, localization or downstream inference.
        with guard.evaluator():
            target=json.loads(case.target_path.read_text(encoding='utf-8'))
            loc,side,components=localization_case(front,outputs,target,height)
            if args.mode=='full':
                for v in VARIANTS:
                    evaluation=evaluate_case(results[v],target,diags[v])
                    evaluations[v].append({'case_id':case_id,'source':source_from_name(case.image_path.name),'status':'ok','evaluation':evaluation})
            if case_id in args.debug_case_ids.split(','):
                debug_case(image,front,outputs,target,height,ROOT/'debug/pin_sina'/args.output.stem/case_id)
        category='design' if case_id in DESIGN else 'check' if case_id in CHECK else 'other_dev'
        all_rows.append({'case_id':case_id,'source':source_from_name(case.image_path.name),'partition':category,'localization':loc,'by_side':side,'per_component':components,'frontend_cache_hit':hit})
        print(json.dumps({'case_id':case_id,'status':'ok','cache':hit,'localization':{v:{'pred':r['pred'],'tp5':r['tp_5'],'tp20':r['tp_20']} for v,r in loc.items()} }),flush=True)
    summary={}
    for group in ('overall','design','check','KiCad','Altium Designer','Datasheet','jlc','other'):
        subset=[r for r in all_rows if group=='overall' or r['partition']==group or r['source']==group]
        if not subset:continue
        summary[group]={}
        for v in VARIANTS:
            counts=empty_counts()
            sides={s:empty_counts() for s in ('left','right','top','bottom')}
            for row in subset:
                add_counts(counts,row['localization'][v])
                for s in sides:add_counts(sides[s],row['by_side'][v][s])
            summary[group][v]={'case_count':len(subset),**metrics(counts),'by_side':{s:metrics(r) for s,r in sides.items()}}
    payload={'metadata':{'OFFICIAL_SCORE':False,'sealed_holdout_used':False,'mode':args.mode,'case_ids':[f'{c.case_id:04d}' for c in cases],'success_count':len(all_rows),'failure_count':len(failures),'formal_config':config.as_dict(),'box_only':True,'tip_formula':'max(6,13*V3 scale)','parameters':asdict(CONFIG),'parameter_sha256':digest(ROOT/'pcb/pin/localization/box_skeleton_experiment.py'),'runtime_seconds':time.perf_counter()-started,'target_open_during_inference_attempts':guard.target_attempts,'sealed_open_attempts':guard.sealed_attempts,'non_box_parity_case_count':nonbox_parity,'baseline_prediction_parity_case_count':baseline_parity,'matching':'evaluate_v2 strict matched boxes + side-consistent per-component one-to-one threshold-gated Hungarian'},'localization':summary,'cases':all_rows,'end_to_end':{v:aggregate(rows) for v,rows in evaluations.items()} if args.mode=='full' else {}}
    write_json(args.output,payload)
    per_case=[]
    for row in all_rows:
        for v in VARIANTS:per_case.append({'case_id':row['case_id'],'source':row['source'],'partition':row['partition'],'variant':v,**{k:val for k,val in metrics(row['localization'][v]).items() if not isinstance(val,dict)}})
    csv_path=args.output.with_suffix('.csv');csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_case[0]));writer.writeheader();writer.writerows(per_case)
    print(json.dumps({'output':str(args.output),'success':len(all_rows),'summary':{v:{k:r[k] for k in ('gt','pred','at_5','at_20')} for v,r in summary['overall'].items()}},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
