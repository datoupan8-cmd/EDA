"""Save the completed agent experiment decision, per-case table and a few overlays."""
from __future__ import annotations
import ast
import csv
from pathlib import Path
import sys
import json

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pin_skeleton_followup import (
    OUT,MANUAL,read_json,write_json,selected_cases,verify_formal_state,
    AccessGuard,event_lookup,
    _strict_box_pairs,_gt_pins,constrained_matches,
)
from pcb.io import read_image,write_image
from pcb.coordinates import target_to_opencv
from pcb.submission import validate_strict
from evaluate_v2 import evaluate_case,aggregate,source_from_name
import cv2


def overlay(image,component,old,new,target_pins,height,path):
    """Evaluation-only display. It never feeds inference or parameter selection."""
    x1,y1,x2,y2=map(float,component['bbox'])
    h,w=image.shape[:2]
    left=max(0,int(x1-80));top=max(0,int(y1-65))
    right=min(w,int(x2+80));bottom=min(h,int(y2+65))
    canvas=image[top:bottom,left:right].copy()
    def p(point):return (round(point[0]-left),round(point[1]-top))
    cv2.rectangle(canvas,p((x1,y1)),p((x2,y2)),(170,60,170),1)
    for a,b in zip(old,new):
        cv2.circle(canvas,p(a['base']),3,(120,120,120),-1)
        cv2.drawMarker(canvas,p(a['tip']),(255,80,0),cv2.MARKER_CROSS,9,1)
        cv2.circle(canvas,p(b['tip']),4,(0,170,0),1)
    for key,pin in target_pins.items():
        q=target_to_opencv(pin['point']['x'],pin['point']['y'],height)
        cv2.drawMarker(canvas,p(q),(0,0,240),cv2.MARKER_TILTED_CROSS,9,1)
    canvas=cv2.copyMakeBorder(canvas,42,0,0,max(0,520-canvas.shape[1]),cv2.BORDER_CONSTANT,value=(255,255,255))
    cv2.putText(canvas,'Blue: V3 tip  Green: experiment tip  Red: GT',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
    cv2.putText(canvas,'Purple: body bbox  Gray: base  (evaluation only)',(8,34),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
    write_image(path,canvas)


def main():
    guard=AccessGuard();formal=verify_formal_state()
    dataset=Path(read_json(MANUAL/'outputs/design/latest.json')['metadata']['dataset_root'])
    tasks={task:read_json(OUT/task/'latest.json') for task in ('smoke','design','check','text_design','text_check')}
    image_cache={};case_cache={}
    for subset in ('design','check'):
        for case in selected_cases(dataset,subset):
            key=f'{case.case_id:04d}';case_cache[key]=case
            image_cache[key]=read_image(case.image_path)
    validations=0
    for task,report in tasks.items():
        folder=Path(report['metadata']['run_directory'])
        for key in report['metadata']['case_ids']:
            if key not in case_cache: raise AssertionError('Case outside fixed subsets')
            h,w=image_cache[key].shape[:2]
            for variant in report['metadata']['stage_factors']:
                pred=read_json(folder/'predictions'/variant/key/'result.json')
                validate_strict(pred,(w,h));validations+=1
    per_case=[];all_records={'A':[],'TN':[]};subsets={};overlays=[]
    for subset in ('design','check'):
        report=tasks['text_'+subset];folder=Path(report['metadata']['run_directory'])
        subsets[subset]={'case_ids':report['metadata']['case_ids'],
            'variants':{v:{'localization':report['localization'][v],
                          'overall':report['end_to_end'][v]['overall'],
                          'tp_changes':report['tp_changes'][v]} for v in report['metadata']['stage_factors']}}
        for key in report['metadata']['case_ids']:
            case=case_cache[key];height=image_cache[key].shape[0]
            with guard.evaluator():
                gt=read_json(case.target_path)
                predictions={v:read_json(folder/'predictions'/v/key/'result.json') for v in ('A','TN')}
                evals={}
                for v,pred in predictions.items():
                    diag=read_json(folder/'predictions'/v/key/'diagnostics.json')
                    evals[v]=evaluate_case(pred,gt,diag)
                    all_records[v].append({'case_id':key,'source':source_from_name(case.image_path.name),
                        'status':'ok','evaluation':evals[v]})
                a,b=evals['A']['metrics'],evals['TN']['metrics']
                record={'subset':subset,'case_id':key,'gt_pins':a['Pin']['gt'],
                    'pred_pin_before':a['Pin']['pred'],'pred_pin_after':b['Pin']['pred'],
                    'strict_tp_before':a['Pin']['tp'],'strict_tp_after':b['Pin']['tp'],
                    'pin_f1_before':a['Pin']['f1'],'pin_f1_after':b['Pin']['f1'],
                    'pair_tp_before':a['PinPair']['tp'],'pair_tp_after':b['PinPair']['tp'],
                    'pair_pred_before':a['PinPair']['pred'],'pair_pred_after':b['PinPair']['pred'],
                    'pair_f1_before':a['PinPair']['f1'],'pair_f1_after':b['PinPair']['f1']}
                record['pin_f1_status']=('improved' if record['pin_f1_after']>record['pin_f1_before']+1e-12 else
                    'regressed' if record['pin_f1_after']<record['pin_f1_before']-1e-12 else 'unchanged')
                per_case.append(record)
                owner={'0014':'U14','0017':'U1','0018':'U2','0129':'U1'}.get(key)
                if key=='0034':
                    data={v:{r['component']:r['terminals'] for r in read_json(folder/'predictions'/v/key/'raw_terminals.json')} for v in ('A','TN')}
                    losses=[]
                    for pk,gk in _strict_box_pairs(predictions['A']['components'],gt['components']):
                        points=_gt_pins(gt,gk,height)
                        losses.append((len(constrained_matches(points,data['A'][pk],20))-
                                       len(constrained_matches(points,data['TN'][pk],20)),pk))
                    owner=max(losses,default=(0,None))[1]
                if owner and owner in predictions['A']['components'] and owner in gt['components']:
                    raws={v:{r['component']:r for r in read_json(folder/'predictions'/v/key/'raw_terminals.json')} for v in ('A','TN')}
                    out=OUT/'debug'/f'{key}_{owner}.png'
                    comp=predictions['A']['components'][owner]
                    # Submission bbox is lower-left; raw snapshots retain OpenCV geometry.
                    from pcb.coordinates import target_bbox_to_opencv
                    comp={**comp,'bbox':target_bbox_to_opencv(comp['bbox'],height)}
                    overlay(image_cache[key],comp,raws['A'][owner]['terminals'],raws['TN'][owner]['terminals'],
                            gt['pins'].get(owner,{}),height,out)
                    overlays.append(str(out))
    combined={v:aggregate(records) for v,records in all_records.items()}
    if any(r['metadata']['target_in_inference_attempts'] or r['metadata']['sealed_attempts'] for r in tasks.values()):
        raise AssertionError('An experiment attempted forbidden dataset access')
    formal_after=verify_formal_state()
    source_files=[ROOT/'experiments'/name for name in (
        'pin_skeleton_followup.py','pin_skeleton_followup_audit.py','pin_skeleton_followup_finalize.py')]
    for path in source_files:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    gained=sum(len(subsets[s]['variants']['TN']['tp_changes']['gained']) for s in subsets)
    lost=sum(len(subsets[s]['variants']['TN']['tp_changes']['lost']) for s in subsets)
    validation={'formal_files_before':formal,'formal_files_after':formal_after,
        'strict_submission_pass':validations,'strict_submission_fail':0,'baseline_json_parity':14,
        'fixed_unique_development_cases':14,'full150_run':False,'sealed_holdout_used':False,
        'new_models_or_ocr_calls':False,'new_source_ast_pass':len(source_files),
        'unit_tests':{'count':28,'passed':28,'command':'python -B -m unittest discover -s local_experiments/pin_skeleton_recovery/tests -q'},
        'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts}
    result={'metadata':{'OFFICIAL_SCORE':False,'dataset_root':str(dataset),'formal_config_changed':False,
             'stage_scope':'box localization tip and number-text distance; no component/wire/topology algorithm changes',
             'evaluation_scope':'fixed previously-used 6 Design + 8 Check development cases; not out-of-fold or sealed test'},
        'decision':'KEEP_EXPERIMENTAL_DO_NOT_PROMOTE_FORMAL',
        'subsets':subsets,'combined_14':combined,'strict_tp_gained':gained,'strict_tp_lost':lost,
        'per_case':per_case,'validation':validation,'debug_overlays':overlays,
        'next_priority':['Design regression: diagnose false pairs in 0014/0017 while freezing downstream code.',
                         'Pin semantics: number/name evidence-aware recovery with abstention, no internal-ID guessing.',
                         'Only after regression guard passes, run the fixed 150-case comparison.']}
    write_json(OUT/'decision.json',result);write_json(OUT/'validation.json',validation)
    with (OUT/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_case[0]));writer.writeheader();writer.writerows(per_case)
    a=combined['A']['overall'];b=combined['TN']['overall']
    lines=['# Pin骨架恢复后续实验：结论与下一步','',
        '## 结论','',
        '不切换正式baseline。骨架补候选暂未带来严格Pin TP；已找到一个可复现的关联缺陷：脚号文字框中心常高于引脚行，中心距离门控会拒绝正确OCR。改用文字框距离，并结合已有Tip校正后，固定14例严格Pin TP从235增至267（+32，旧TP未丢失）。但Design网络误报仍明显增加，不能称为完整成功。','',
        '本轮未运行150例，下面F1都是对应小子集的本地非官方诊断，不能替换已有150例基线成绩。','',
        '## 实验顺序','',
        '1. 回读原6例骨架结果，确认B/C都新增0、丢失0严格TP，确实是新增候选未转化。',
        '2. 在Design做10组候选/Tip/保序语义对照。骨架B/C提高定位，但没有额外严格TP；不继续扩大候选池。',
        '3. 对A/S/T/TS做固定8例Check，参数不变。Tip改善5px位置，但语义仍阻塞。',
        '4. 检查准确5px候选的OCR证据，发现已有脚号被文字中心距离门控拒绝。',
        '5. 新增N：使用文字框的法线/切向距离，角色/正则/阈值/脚名评分/配对/顺序逻辑全部沿用。TN同时使用已有Tip公式；Design和Check分开运行，Check不调参。','',
        'A=正式V3；S=box保序语义；T=box Tip校正；TS=T+S；N=S+脚号框距离；TN=T+N。','',
        '## 结果','',
        '|子集|例数|严格TP A→TN|Pin宏F1 A→TN|box定位Recall@5 A→TN|PinPair宏F1 A→TN|',
        '|---|---:|---:|---:|---:|---:|']
    for s,data in subsets.items():
        av=data['variants']['A'];bv=data['variants']['TN']
        am=av['overall']['metrics'];bm=bv['overall']['metrics']
        lines.append(f"|{s}|{len(data['case_ids'])}|{am['Pin']['tp']}→{bm['Pin']['tp']}|{am['Pin']['macro_f1']:.6f}→{bm['Pin']['macro_f1']:.6f}|{av['localization']['at_5']['recall']:.4%}→{bv['localization']['at_5']['recall']:.4%}|{am['PinPair']['macro_f1']:.6f}→{bm['PinPair']['macro_f1']:.6f}|")
    lines+=['',f"14例合并：GT Pin={a['metrics']['Pin']['gt']}，导出Pin {a['metrics']['Pin']['pred']}→{b['metrics']['Pin']['pred']}；严格TP {a['metrics']['Pin']['tp']}→{b['metrics']['Pin']['tp']}；Pin微F1 {a['metrics']['Pin']['f1']:.6f}→{b['metrics']['Pin']['f1']:.6f}；宏F1 {a['metrics']['Pin']['macro_f1']:.6f}→{b['metrics']['Pin']['macro_f1']:.6f}。",
        f"严格TP新增{gained}，丢失{lost}。逐例Pin F1：{sum(r['pin_f1_status']=='improved' for r in per_case)}改善/{sum(r['pin_f1_status']=='unchanged' for r in per_case)}不变/{sum(r['pin_f1_status']=='regressed' for r in per_case)}回退。逐例详情见per_case.csv。",'',
        '## 改了什么、没改什么','',
        '- 仅新增独立实验Stage，使用原V3候选、冻结的P1 Tip公式和V6分侧保序关联。新评分只换距离的计算方式，不用GT均值作offset。',
        '- N/TN没有新增或删除terminal；base/side/method/order和切向坐标保持；非box Pin JSON逐字段不变。',
        '- Component/OCR/模型/权重/正式V3源码/current配置/持久Registry/Wire/Topology/Submission都没有修改。',
        '- 已签名的image-only前端快照复用，不调用新的模型或OCR；GT只在独立评测窗口读入。',
        '- 新scorer通过进程内临时引用仅用于串行实验，finally恢复；不能未经接口整理就放到并行生产推理。','',
        '## 仍然存在的问题','',
        '1. 14例matched-box共742个GT terminal，预测候选269个，TN在5px仅定位161个（约21.7%），复杂器件候选不足仍未解决。',
        '2. 定位正确也不保证脚号/脚名正确。原TS的161个准确位置中，100个所需脚号字符串不在局部OCR缓存；这不能自动证明图中不可见或GT错误。',
        '3. Design的Pair正确数量18→22，但预测Pair 133→573；更密集的错误连接抵消了收益。回归集中于0014/0017，不能只报告TP上涨。',
        '4. 统一Tip缩短会影响既有Wire corridor与snapping输入；Check的20px定位Recall 29.30%→26.56%。0034的20px匹配8→0；0129、0143各丢失1个5px匹配，不能用总体提升掩盖回归。',
        '5. Check是曾用于开发的固定子集，不是真正未见测试；已有YOLO权重训练历史也没有在本轮变为OOF证据。','',
        '## 下一步','',
        '优先审计0014/0017新增错误Pair的Pin端点、导出语义和几何证据，保留当前下游源码。验证是缺少可信脚号/脚名却仍导出，还是Tip与导线锚点不一致；每次只改变一个实验因素。先通过这项回归保护，再考虑150例固定基线对照。暂不叠加骨架候选，不扩大OCR调用，不推断不可见内部ID。','',
        '## 验证与文件','',
        f"28项专项测试通过；{validations}/{validations}份本轮预测通过strict submission validator；14/14个正式A预测与已有基线JSON一致；209个受保护正式文件SHA256未变。",
        '未读取0151～0200、Golden、10GT或QuickTest；没有commit/push/PR/训练/环境升级/ZIP/DOCX。',
        '本轮新文件：experiments/pin_skeleton_followup.py、pin_skeleton_followup_audit.py、pin_skeleton_followup_finalize.py及test_followup.py；报告和少量图都在本目录。',
        '原手动run.ps1和config.json未改动，也没有调用它们。组员暂不需要再手工运行。',
        '结果入口：decision.json、per_case.csv、validation.json；完整消融：design/check/text_design/text_check目录；语义证据：text_distance_audit/audit.json。']
    (OUT/'decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'strict_tp_gained':gained,'strict_tp_lost':lost,'validation':validation,
        'combined_pin_before':a['metrics']['Pin'],'combined_pin_after':b['metrics']['Pin'],
        'report':str(OUT/'decision.md'),'overlays':overlays},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
