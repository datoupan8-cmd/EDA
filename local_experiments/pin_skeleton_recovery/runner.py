"""Independent A/B/C runner. All generated files stay inside this package."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from evaluate_v2 import aggregate, evaluate_case, linear_sum_assignment, source_from_name
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import ComponentStageOutput
from pcb.core.registry import build_default_registry
from pcb.data_policy import CaseFiles, assert_allowed_case_id
from pcb.io import read_image, write_image, write_json
from pcb.schema import Component, Text
from pcb.submission import validate_strict
from pcb.text_detection import TokenRole
from tools.run_box_skeleton_experiment import (
    AccessGuard, add_counts, component_payload, digest, empty_counts,
    full_inference, load_frontend, metrics,
)
from tools.evaluate_box_text_mask_counterfactual import _strict_box_pairs, _gt_pins
from local_experiments.pin_skeleton_recovery.candidates import RecoveryConfig, RecoveryStage

SETS = {
    'smoke': ['0017','0087','0143'],
    'design': ['0014','0017','0018','0087','0103','0130'],
    'check': ['0011','0034','0046','0095','0111','0129','0139','0143'],
    'full': [f'{i:04d}' for i in range(1,151)],
}
VARIANTS = {'A': 'v3', 'B': 'local_skeleton_append', 'C': 'local_skeleton_split_append'}
RADII = (5,10,20)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_formal_state():
    before = read_json(PACKAGE/'formal_state_before.json')
    changed = [name for name, sha in before['protected_sha256'].items()
               if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
    if changed:
        raise AssertionError(f'Protected formal files changed; do not run: {changed}')
    return {'protected_file_count': len(before['protected_sha256']), 'unchanged': True,
            'head_at_package_creation': before['head']}


class ExperimentGuard(AccessGuard):
    """Block target access in inference, sealed data, and writes into formal code/caches."""
    def _audit(self, event, args):
        super()._audit(event, args)
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        mode = args[1] if len(args) > 1 else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(PACKAGE):
                raise PermissionError(f'Experiment cannot write formal project files: {path}')


def selected_cases(dataset, task):
    """List only requested allowed directories; never discover sealed folders."""
    dataset = Path(dataset).resolve()
    if dataset.name != '200_train_cases':
        raise ValueError('Dataset must be the 200_train_cases directory')
    output = []
    for key in SETS[task]:
        number = int(key)
        assert_allowed_case_id(number)
        folder = dataset/key
        images, targets = sorted(folder.glob('*.png')), sorted(folder.glob('*_target.json'))
        if len(images) != 1 or len(targets) != 1:
            raise ValueError(f'{key}: expected exactly one PNG and one target')
        output.append(CaseFiles(number, folder, images[0], targets[0], 'dev'))
    return output


def constrained_matches(gt, pred, radius):
    """Within one owner: same side, maximum cardinality first, distance second.

    A high dummy cost makes losing any feasible match more expensive than all
    real distances combined. This is localization diagnostics only; the formal
    evaluate_v2.py and its scoring behavior are unchanged.
    """
    if not gt or not pred:
        return {}
    unmatched = (len(gt)+1)*(radius+1)
    cost = np.full((len(gt), len(pred)+len(gt)), unmatched*10)
    for i, pin in enumerate(gt):
        for j, terminal in enumerate(pred):
            distance = math.dist(pin['point'], terminal['tip'])
            if pin['side'] == terminal['side'] and distance <= radius:
                cost[i,j] = distance
        cost[i,len(pred)+i] = unmatched
    rows, columns = linear_sum_assignment(cost)
    return {gt[int(i)]['id']: (int(j), float(cost[i,j]))
            for i,j in zip(rows,columns) if j < len(pred) and cost[i,j] <= radius}


def check_invariants(a, other):
    if len(a.terminals) != len(other.terminals):
        raise AssertionError('Component owner list changed')
    added = 0
    for (owner, original), (new_owner, rows) in zip(a.terminals, other.terminals):
        if (owner.key, owner.bbox, owner.body_bbox) != (new_owner.key, new_owner.bbox, new_owner.body_bbox):
            raise AssertionError('Component geometry/identity changed')
        if rows[:len(original)] != original:
            raise AssertionError('V3 candidate prefix was changed/lost/reordered')
        if owner.type != 'box' and rows != original:
            raise AssertionError('Non-box output changed')
        added += len(rows)-len(original)
    return added


def localization(front, outputs, target, height):
    pairs = _strict_box_pairs(component_payload(front,height), target['components'])
    totals = {v: empty_counts() for v in VARIANTS}
    sides = {v: {s:empty_counts() for s in ('left','right','top','bottom')} for v in VARIANTS}
    records = []
    lookup = {v:{owner.key:rows for owner,rows in out.terminals} for v,out in outputs.items()}
    for pk,gk in pairs:
        gt = _gt_pins(target,gk,height)
        baseline_matches = {r:constrained_matches(gt,lookup['A'][pk],r) for r in RADII}
        for variant in VARIANTS:
            candidates = lookup[variant][pk]
            matches = {r:constrained_matches(gt,candidates,r) for r in RADII}
            block = empty_counts()
            block.update(components=1, gt=len(gt), pred=len(candidates),
                         absolute_count_error=abs(len(gt)-len(candidates)), exact_count=int(len(gt)==len(candidates)))
            block.update({f'tp_{r}':len(matches[r]) for r in RADII})
            # All V3 points remain eligible; an appendix cannot reduce maximum match count.
            if any(len(matches[r]) < len(baseline_matches[r]) for r in RADII):
                raise AssertionError('Maximum-cardinality localization regressed despite retained prefix')
            add_counts(totals[variant],block)
            for side in sides[variant]:
                gs = [p for p in gt if p['side']==side]
                ps = [p for p in candidates if p['side']==side]
                sb = empty_counts()
                sb.update(components=1,gt=len(gs),pred=len(ps),absolute_count_error=abs(len(gs)-len(ps)),exact_count=int(len(gs)==len(ps)))
                sb.update({f'tp_{r}':len(constrained_matches(gs,ps,r)) for r in RADII})
                add_counts(sides[variant][side],sb)
            records.append({'component':pk,'gt_component':gk,'variant':variant,**block,
                            'tp_gain_5':len(matches[5])-len(baseline_matches[5]),
                            'tp_gain_20':len(matches[20])-len(baseline_matches[20])})
    return totals,sides,records


def cached_frontend(case, image, context, runtime):
    """Reuse validated old image-only snapshots read-only; new cache stays here."""
    signature = {'image_sha256':digest(case.image_path),'config':context.config.as_dict(),
                 'model_sha256':runtime['model_sha256'],'frontend_source_sha256':runtime['frontend_source_sha256']}
    for base in (PACKAGE/'cache/frontend', ROOT.parent/'tmp/pin_sina_frontend'):
        path = base/f'{case.case_id:04d}.json'
        if path.is_file():
            value = read_json(path)
            if value.get('signature') == signature:
                components = [Component(**{**r,'bbox':tuple(r['bbox']),'body_bbox':tuple(r['body_bbox']) if r.get('body_bbox') else None}) for r in value['components']]
                texts = [Text(**{**r,'bbox':tuple(r['bbox'])}) for r in value['texts']]
                roles = [TokenRole(**{**r,'token':Text(**{**r['token'],'bbox':tuple(r['token']['bbox'])})}) for r in value['roles']]
                return ComponentStageOutput(components,texts,roles,value['diagnostics']), str(base)
    if runtime.get('ocr') is None:
        from pcb.ocr_backends import TiledEasyOCR, HybridOCR
        from pcb.vision import OCR
        from pcb.component_detector_yolo import YoloComponentDetector, short_cache_key

        class ReadThroughTiledOCR(TiledEasyOCR):
            def _read_tile(self, tile):
                path = self._tile_cache_path(tile)
                old = ROOT/'runs/ocr_cache_v4_1/easyocr_tiles'/path.name
                if not path.is_file() and old.is_file():
                    shutil.copyfile(old,path)
                return super()._read_tile(tile)

        class ReadThroughDetector(YoloComponentDetector):
            def detect(self, image):
                payload = json.dumps({'weights_sha256':self.weight_sha256,'config':asdict(self.config)},sort_keys=True).encode()
                name = short_cache_key(image,payload)+'.json'
                old, new = ROOT/'runs/yolo_cache'/name, self.cache_dir/name
                if not new.is_file() and old.is_file():
                    shutil.copyfile(old,new)
                return super().detect(image)

        cache = PACKAGE/'cache/ocr'
        runtime['ocr'] = HybridOCR(ReadThroughTiledOCR(cache,ROOT/'models/easyocr',device='auto',allow_download=False),OCR(cache))
        runtime['detector'] = ReadThroughDetector(ROOT/'models/component_yolo11n_continue_v2_best.pt','auto',cache_dir=PACKAGE/'cache/yolo')
    # This helper uses the explicitly supplied OCR/detector and local snapshot dir.
    return load_frontend(case,image,context,PACKAGE/'cache/frontend',runtime)[0], 'recomputed_local_cache'


def experiment_fingerprint():
    paths = [PACKAGE/'runner.py',PACKAGE/'candidates.py',PACKAGE/'config.json',PACKAGE/'formal_state_before.json']
    return hashlib.sha256(json.dumps({p.name:digest(p) for p in paths},sort_keys=True).encode()).hexdigest()


def full_gate(fingerprint, dataset=None):
    reports = []
    for task in ('design','check'):
        path = PACKAGE/'outputs'/task/'latest.json'
        if not path.is_file():
            raise ValueError('Full150 blocked: run fixed Design and Check first')
        report = read_json(path)
        if report['metadata']['experiment_sha256'] != fingerprint or report['metadata']['case_ids'] != SETS[task]:
            raise ValueError('Full150 blocked: experiment changed or incomplete Design/Check')
        reports.append(report)
    roots = {r['metadata']['dataset_root'] for r in reports}
    if len(roots) != 1 or (dataset is not None and str(Path(dataset).resolve()) not in roots):
        raise ValueError('Full150 blocked: Design/Check/full dataset roots differ')
    winners = []
    for variant in ('B','C'):
        valid = True
        for report in reports:
            a,b = report['localization']['A'], report['localization'][variant]
            ea,eb = report['end_to_end']['A']['overall']['metrics'],report['end_to_end'][variant]['overall']['metrics']
            if (not a['gt'] or b['tp_5'] <= a['tp_5']
                    or b['at_5']['f1'] < a['at_5']['f1']
                    or eb['Pin']['tp'] < ea['Pin']['tp']
                    or any(eb[key]['macro_f1'] < ea[key]['macro_f1']-1e-12 for key in ('Pin','NetHypergraph','NetLine','PinPair'))):
                valid = False
        if valid:
            winners.append(variant)
    if not winners:
        raise ValueError('Full150 blocked: no variant passes Design/Check localization + downstream regression gates. Share latest_summary.md first.')
    return winners


def debug_overlay(image, front, outputs, target, height, directory):
    pairs = _strict_box_pairs(component_payload(front,height),target['components'])
    if not pairs:
        return
    pk,gk = pairs[0]
    owner = next(c for c in front.components if c.key==pk)
    x1,y1,x2,y2 = map(lambda v:int(round(v)),owner.body_bbox or owner.bbox)
    a,b,c,d = max(0,x1-45),max(0,y1-45),min(image.shape[1],x2+46),min(height,y2+46)
    panels=[]
    for variant,out in outputs.items():
        panel=image[b:d,a:c].copy()
        cv2.rectangle(panel,(x1-a,y1-b),(x2-a,y2-b),(180,100,0),1)
        rows=next(rows for comp,rows in out.terminals if comp.key==pk)
        for gt in _gt_pins(target,gk,height):
            cv2.circle(panel,(round(gt['point'][0])-a,round(gt['point'][1])-b),4,(0,160,0),1)
        original_count=len(next(rows for comp,rows in outputs['A'].terminals if comp.key==pk))
        for index,terminal in enumerate(rows):
            pt=tuple(round(v)-offset for v,offset in zip(terminal['tip'],(a,b)))
            cv2.circle(panel,pt,2,(0,0,220) if index<original_count else (220,0,220),-1)
        cv2.putText(panel,f'{variant} {pk}',(6,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        panels.append(panel)
    write_image(directory/'ABC_first_matched_box.png',np.hstack(panels))


def markdown(report):
    m = report['metadata']
    lines = [f"# Pin骨架补充实验：{m['task']}",'',f"完成 {m['success_count']}/{len(m['case_ids'])}；失败 {m['failure_count']}；本地非官方诊断。",
             f"正式文件哈希一致：{m['formal_state']['unchanged']}（{m['formal_state']['protected_file_count']}个）；V3候选前缀全部保留。",
             f"基线最终JSON历史对照：{m['baseline_parity_count']}例；缺少历史对照：{m['baseline_reference_missing']}。",
             f"sealed/推理读取GT阻止计数：{m['sealed_open_attempts']}/{m['target_open_during_inference_attempts']}。",'',
             '定位表只统计严格匹配的box，按owner与side一对一；Pin表统计所选case的全部原始target。空box集合不可作为成功依据。','',
             '|组|GT terminal|Pred terminal|TP@5|Recall@5|F1@5|Recall@10|Recall@20|Count MAE|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v,r in report['localization'].items():
        mae=f"{r['pin_count_mae']:.3f}" if r['pin_count_mae'] is not None else 'N/A'
        lines.append(f"|{v}|{r['gt']}|{r['pred']}|{r['tp_5']}|{r['at_5']['recall']:.4f}|{r['at_5']['f1']:.4f}|{r['at_10']['recall']:.4f}|{r['at_20']['recall']:.4f}|{mae}|")
    lines += ['', '|组|新增box候选(全部预测box)|严格Pin TP|Pin微平均F1|Pin宏平均F1|Hypergraph宏F1|Line宏F1|PinPair宏F1|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        s=report['end_to_end'][v]['overall']['metrics']
        lines.append(f"|{v}|{m['added_candidates'][v]}|{s['Pin']['tp']}|{s['Pin']['f1']:.4f}|{s['Pin']['macro_f1']:.4f}|{s['NetHypergraph']['macro_f1']:.4f}|{s['NetLine']['macro_f1']:.4f}|{s['PinPair']['macro_f1']:.4f}|")
    lines += ['', 'A=V3；B=V3+环带骨架补充；C=V3+向外分支拆分补充。B/C使用相同文字处理、支持检查、去重和tip公式。',
              '定位TP可以增加而最终Pin TP不增加；新增候选也可能降低精度或影响脚号分配。因此不会自动替换正式配置。',
              f"完整数据：{m['run_directory']}",'',
              '下一步：smoke通过→Design→冻结参数→Check。Design/Check同向改善且下游不回归，才解锁full。请把本摘要发回讨论，勿反复用Check调参。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=tuple(SETS),required=True)
    parser.add_argument('--dataset-root',type=Path,required=True)
    args=parser.parse_args(argv)
    state=verify_formal_state()
    config=PipelineConfig.load(ROOT/'configs/current.json')
    if config != PipelineConfig.current():
        raise AssertionError('Expected frozen Component BEST + V3 pin/wire/topology baseline')
    recovery=RecoveryConfig(**read_json(PACKAGE/'config.json'))
    fingerprint=experiment_fingerprint()
    if args.task=='full':
        full_gate(fingerprint,args.dataset_root)
    guard=ExperimentGuard()
    cases=selected_cases(args.dataset_root,args.task)
    run=PACKAGE/'outputs'/args.task/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    sources=['pcb/component_detection_v4.py','pcb/component/v4.py','pcb/text_detection.py',
             'pcb/component_text_assignment.py','pcb/ocr_backends.py','pcb/text/stage.py']
    runtime={'model_sha256':digest(ROOT/'models/component_yolo11n_continue_v2_best.pt'),
             'frontend_source_sha256':{name:digest(ROOT/name) for name in sources}}
    registry=build_default_registry()
    registry.register('pin_localization',VARIANTS['B'],lambda:RecoveryStage(recovery,False))
    registry.register('pin_localization',VARIANTS['C'],lambda:RecoveryStage(recovery,True))
    totals={v:empty_counts() for v in VARIANTS}
    evals={v:[] for v in VARIANTS}
    per_case=[]
    added={v:0 for v in VARIANTS}
    parity=0
    missing=[]
    try:
        for index,case in enumerate(cases):
            key=f'{case.case_id:04d}'
            image=read_image(case.image_path)
            height,width=image.shape[:2]
            context=PipelineContext(case.image_path.name,width,height,config,None)
            front,cache=cached_frontend(case,image,context,runtime)
            outputs={v:registry.create('pin_localization',name).run(image,copy.deepcopy(front),context) for v,name in VARIANTS.items()}
            for v in ('B','C'):
                added[v]+=check_invariants(outputs['A'],outputs[v])
            results,diagnostics={},{}
            for v in VARIANTS:
                local=copy.deepcopy(front)
                loc=copy.deepcopy(outputs[v])
                loc.terminals=[(owner,loc.terminals[i][1]) for i,owner in enumerate(local.components)]
                t=time.perf_counter()
                results[v],diagnostics[v]=full_inference(image,local,loc,replace(context,config=replace(config,pin_localization=VARIANTS[v])),registry)
                diagnostics[v]['seconds']=time.perf_counter()-t
                validate_strict(results[v],(width,height))
                directory=run/'predictions'/v/key
                write_json(directory/'result.json',results[v])
                write_json(directory/'raw_terminals.json',[{'component':owner.key,'type':owner.type,'terminals':terms} for owner,terms in outputs[v].terminals])
                write_json(directory/'diagnostics.json',diagnostics[v])
            reference=ROOT.parent/'tmp/pin_p6_5_formal_predictions'/key/'result.json'
            if reference.is_file():
                if read_json(reference)!=results['A']:
                    raise AssertionError(f'{key}: A final JSON differs from formal reference')
                parity+=1
            else:
                missing.append(key)
            with guard.evaluator():
                target=read_json(case.target_path)
                loc,sides,components=localization(front,outputs,target,height)
                for v in VARIANTS:
                    add_counts(totals[v],loc[v])
                    ev=evaluate_case(results[v],target,diagnostics[v])
                    evals[v].append({'case_id':key,'source':source_from_name(case.image_path.name),'status':'ok','evaluation':ev})
                if index<3:
                    debug_overlay(image,front,outputs,target,height,run/'debug'/key)
            per_case.append({'case_id':key,'source':source_from_name(case.image_path.name),'frontend_cache':cache,
                             'image_sha256':digest(case.image_path),'target_sha256':None,
                             'localization':loc,'by_side':sides,'per_component':components})
            # target hashing is also an evaluator-only operation.
            with guard.evaluator():
                per_case[-1]['target_sha256']=digest(case.target_path)
            print(json.dumps({'case':key,'status':'ok','box_tp5':{v:loc[v]['tp_5'] for v in VARIANTS},
                              'box_pred':{v:loc[v]['pred'] for v in VARIANTS}},ensure_ascii=False),flush=True)
        state=verify_formal_state()
    except Exception as exc:
        write_json(run/'failure.json',{'error':repr(exc),'completed_cases':[r['case_id'] for r in per_case],
                                     'OFFICIAL_SCORE':False,'formal_state_after':verify_formal_state()})
        raise
    report={'metadata':{'task':args.task,'case_ids':SETS[args.task],'success_count':len(per_case),'failure_count':0,
                        'OFFICIAL_SCORE':False,'sealed_holdout_used':False,'formal_state':state,
                        'experiment_sha256':fingerprint,'parameters':asdict(recovery),
                        'dataset_root':str(args.dataset_root.resolve()),'formal_config':config.as_dict(),
                        'baseline_parity_count':parity,'baseline_reference_missing':missing,
                        'target_open_during_inference_attempts':guard.target_attempts,'sealed_open_attempts':guard.sealed_attempts,
                        'added_candidates':added,'v3_prefix_preserved':True,'non_box_unchanged':True,
                        'runtime_seconds':time.perf_counter()-started,'run_directory':str(run)},
            'localization':{v:metrics(counts) for v,counts in totals.items()},
            'end_to_end':{v:aggregate(records) for v,records in evals.items()},'cases':per_case}
    write_json(run/'report.json',report)
    rows=[{'case_id':c['case_id'],'source':c['source'],'variant':v,**{k:x for k,x in metrics(c['localization'][v]).items() if not isinstance(x,dict)},
           'recall_5':metrics(c['localization'][v])['at_5']['recall'],'f1_5':metrics(c['localization'][v])['at_5']['f1']} for c in per_case for v in VARIANTS]
    with (run/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary=markdown(report)
    (run/'summary_to_share.md').write_text(summary,encoding='utf-8')
    write_json(run.parent/'latest.json',report)
    (run.parent/'latest_summary.md').write_text(summary,encoding='utf-8')
    print(f"SUMMARY: {run.parent/'latest_summary.md'}",flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
