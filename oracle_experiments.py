"""Oracle ablations for cases 0001..0150 only. Never imported by main.py."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import sys
local_runtime=Path(__file__).resolve().parent.parent/'work/runtime'
if local_runtime.exists():sys.path.insert(0,str(local_runtime))
import argparse,copy,json,math,time,traceback
import cv2,numpy as np
from pcb.io import read_image,write_json
from pcb.schema import Component,Pin,Scene
from pcb.coordinates import target_bbox_to_opencv,target_to_opencv
from pcb.vision import OCR
from pcb.vision_v2 import parse_pins_for_components,associate_text_for_known_components
from pcb.wire_v2 import extract_wire_v2
from pcb.topology_v2 import build_topology_v2
from pcb.submission import export,validate_strict
from evaluate_v2 import evaluate_case,aggregate,source_from_name

HOLDOUT_START=151
OFFICIAL_TO_INTERNAL={'r':'resistor','c':'capacitor','l':'inductor','box':'ic','v':'vcc','d':'diode','pin':'connector','battary':'battery','amp':'amplifier','net_input':'vbus_in','net_output':'vbus_out','net_bidirection':'vbus_bi'}
LABEL_TYPES={'net_input','net_output','net_bidirection','net_short','v','gnd','testpoint'}

def side_base(tip,bbox):
    x,y=tip;x1,y1,x2,y2=bbox;dist={'left':abs(x-x1),'right':abs(x-x2),'top':abs(y-y1),'bottom':abs(y-y2)};side=min(dist,key=dist.get)
    base={'left':(x1,y),'right':(x2,y),'top':(x,y1),'bottom':(x,y2)}[side]
    return side,base

def gt_scene(data,width,height,include_pin_tips=False,include_pin_semantics=False,include_component_values=True):
    comps=[]
    for key,item in data['components'].items():
        b=list(target_bbox_to_opencv(item['bbox'],height));b[0]=max(0,min(width-1,b[0]));b[2]=max(b[0]+1,min(width,b[2]));b[1]=max(0,min(height-1,b[1]));b[3]=max(b[1]+1,min(height,b[3]))
        raw_type=item['type'];typ=OFFICIAL_TO_INTERNAL.get(raw_type,raw_type)
        c=Component(key,typ,tuple(b),name=item.get('Name'),value=item.get('value') if include_component_values else None,body_bbox=tuple(b),
            net_label=item.get('Name') if raw_type in LABEL_TYPES else None,observable_designator=None if key.startswith('∅') else item.get('Name'),internal_identifier=key if key.startswith('∅') else None)
        if include_pin_tips:
            for pk,p in data['pins'].get(key,{}).items():
                suffix=pk[4:] if pk.startswith('pin_') else pk;tip=target_to_opencv(p['point']['x'],p['point']['y'],height);side,base=side_base(tip,b)
                c.pins.append(Pin(suffix,p.get('pinname','') if include_pin_semantics else '',tip,base=base,side=side,exportable=True,
                    observable_number=suffix if include_pin_semantics else None,internal_key=pk,number_source='gt' if include_pin_semantics else 'oracle_tip_only'))
        comps.append(c)
    return Scene(width,height,comps)

def gt_edge_mask(data,width,height):
    mask=np.zeros((height,width),np.uint8)
    for net in data['nets'].values():
        for edge in net.get('edges',{}).values():
            if len(edge)!=2:continue
            a=target_to_opencv(edge[0]['x'],edge[0]['y'],height);b=target_to_opencv(edge[1]['x'],edge[1]['y'],height)
            cv2.line(mask,tuple(round(v) for v in a),tuple(round(v) for v in b),255,1,cv2.LINE_8)
    return mask

def run_variant(scene,mask,case_dir,mode='directional',merge_labels=False):
    start=time.perf_counter();build_topology_v2(scene,mask.copy(),mode,merge_labels=merge_labels);result=export(scene);validate_strict(result,(scene.width,scene.height))
    case_dir.mkdir(parents=True,exist_ok=True);write_json(case_dir/'result.json',result);scene.diagnostics['seconds']=round(time.perf_counter()-start,3);write_json(case_dir/'diagnostics.json',scene.diagnostics);return result,scene.diagnostics

def write_error_budget(path,aggregates):
    names=[('full_prediction','Full Prediction'),('gt_component','GT Component'),('gt_component_pin_tip','GT Component + GT Pin Tip'),('gt_components_pins','GT Component + GT Pins'),('gt_components_pins_edges_graph','GT Component + GT Pins + GT Edges -> Graph')]
    rows=[]
    for key,label in names:
        block=aggregates.get(key,{}).get('overall',{});m=block.get('metrics',{})
        rows.append((key,label,block.get('FinalScore',0),m.get('Component',{}).get('macro_f1',0),m.get('Pin',{}).get('macro_f1',0),m.get('NetHypergraph',{}).get('macro_f1',0),m.get('NetLine',{}).get('macro_f1',0),m.get('PinPair',{}).get('macro_f1',0)))
    deltas=[]
    for (ka,la,sa,*_),(kb,lb,sb,*__) in zip(rows,rows[1:]):deltas.append((f'{la} -> {lb}',sb-sa))
    ranked=sorted(deltas,key=lambda x:x[1],reverse=True)
    text=['# V2 Oracle Error Budget','','`OFFICIAL_SCORE = FALSE`。全部实验仅使用 0001～0150；0151～0200 与 10_GTcase 未读取。','',
          '| Variant | FinalScore | ComponentF1 | PinF1 | NetHypergraphF1 | NetLineF1 | PinPairF1 |','|---|---:|---:|---:|---:|---:|---:|']
    for _,label,score,*vals in rows:text.append('| '+label+' | '+f'{score:.2f} | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    text+=['','## 相邻 Oracle 增益','']+[f'- {name}: {delta:+.2f} 分' for name,delta in deltas]
    text+=['','## 瓶颈排序','']+[f'{i}. {name}: 可恢复分差 {delta:+.2f}' for i,(name,delta) in enumerate(ranked,1)]
    text+=['','这些差值是条件性误差预算：上游 GT 注入会改变下游输入分布，因此不能把各增益机械相加成独立因果贡献。GT edges 实验主要测 graph builder 上限。']
    path.write_text('\n'.join(text)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--image_root',type=Path,required=True);ap.add_argument('--baseline_dir',type=Path,required=True);ap.add_argument('--output_dir',type=Path,default=Path('experiments/oracle'));ap.add_argument('--report',type=Path,default=Path('reports/v2_oracle_metrics.json'));ap.add_argument('--case_start',type=int,default=1);ap.add_argument('--case_end',type=int,default=150);ap.add_argument('--cache_dir',type=Path,default=Path('runs/ocr_cache'));a=ap.parse_args()
    if a.image_root.name!='200_train_cases' or not(1<=a.case_start<=a.case_end<HOLDOUT_START):raise SystemExit('Holdout guard: exact 200_train_cases root and range 0001..0150 required')
    variants=('full_prediction','gt_component','gt_component_pin_tip','gt_components_pins','gt_components_pins_edges_graph');records={v:[] for v in variants};ocr=OCR(a.cache_dir)
    for n in range(a.case_start,a.case_end+1):
        cid=f'{n:04d}';folder=a.image_root/cid
        try:
            ip=next(folder.glob('*.png'));jp=next(folder.glob('*_target.json'));image=read_image(ip);h,w=image.shape[:2];gt=json.loads(jp.read_text(encoding='utf-8'))
            base_pred=json.loads((a.baseline_dir/cid/'result.json').read_text(encoding='utf-8'));base_diag=json.loads((a.baseline_dir/cid/'diagnostics.json').read_text(encoding='utf-8'))
            records['full_prediction'].append({'case_id':cid,'source':source_from_name(ip.name),'status':'ok','evaluation':evaluate_case(base_pred,gt,base_diag)})
            cached={}
            for variant in variants[1:]:
                rp=a.output_dir/variant/cid/'result.json';dp=a.output_dir/variant/cid/'diagnostics.json'
                if not (rp.exists() and dp.exists()):cached={};break
                result=json.loads(rp.read_text(encoding='utf-8'));diag=json.loads(dp.read_text(encoding='utf-8'));validate_strict(result,(w,h));cached[variant]=(result,diag)
            if len(cached)==len(variants)-1:
                for variant,(result,diag) in cached.items():records[variant].append({'case_id':cid,'source':source_from_name(ip.name),'status':'ok','evaluation':evaluate_case(result,gt,diag)})
                print(json.dumps({'case_id':cid,'status':'cached'},ensure_ascii=False),flush=True);continue
            texts=ocr.recognize(image)
            template=gt_scene(gt,w,h,False,False);template.texts=texts;wire_mask,_,_=extract_wire_v2(image,template)
            # B: bbox/key/type are oracle inputs; value/model text remains predicted.
            b=gt_scene(gt,w,h,False,False,include_component_values=False);b.texts=texts
            associate_text_for_known_components(image,ocr,b.components,texts)
            parse_pins_for_components(image,ocr,b.components,texts,known_tips=False);pred,diag=run_variant(b,wire_mask,a.output_dir/'gt_component'/cid);records['gt_component'].append({'case_id':cid,'source':source_from_name(ip.name),'status':'ok','evaluation':evaluate_case(pred,gt,diag)})
            c=gt_scene(gt,w,h,True,False);c.texts=texts;parse_pins_for_components(image,ocr,c.components,texts,known_tips=True);pred,diag=run_variant(c,wire_mask,a.output_dir/'gt_component_pin_tip'/cid);records['gt_component_pin_tip'].append({'case_id':cid,'source':source_from_name(ip.name),'status':'ok','evaluation':evaluate_case(pred,gt,diag)})
            d=gt_scene(gt,w,h,True,True);d.texts=texts;pred,diag=run_variant(d,wire_mask,a.output_dir/'gt_components_pins'/cid);records['gt_components_pins'].append({'case_id':cid,'source':source_from_name(ip.name),'status':'ok','evaluation':evaluate_case(pred,gt,diag)})
            e=gt_scene(gt,w,h,True,True);e.texts=texts;pred,diag=run_variant(e,gt_edge_mask(gt,w,h),a.output_dir/'gt_components_pins_edges_graph'/cid,merge_labels=True);records['gt_components_pins_edges_graph'].append({'case_id':cid,'source':source_from_name(ip.name),'status':'ok','evaluation':evaluate_case(pred,gt,diag)})
            print(json.dumps({'case_id':cid,'status':'ok'},ensure_ascii=False),flush=True)
        except Exception as exc:
            err={'case_id':cid,'status':'failed','error':repr(exc),'traceback':traceback.format_exc()}
            for v in variants:
                if not records[v] or records[v][-1].get('case_id')!=cid:records[v].append(err)
            print(json.dumps(err,ensure_ascii=False),flush=True)
    aggregates={v:aggregate(rows) for v,rows in records.items()};failures=sorted({r['case_id'] for rows in records.values() for r in rows if r['status']!='ok'})
    output={'OFFICIAL_SCORE':False,'case_start':a.case_start,'case_end':a.case_end,'holdout_used':False,'variants':aggregates,'failure_count':len(failures),'failed_cases':failures,'records':records}
    write_json(a.report,output);write_error_budget(a.report.parent/'v2_error_budget.md',aggregates);print(json.dumps({'failure_count':len(failures),'variants':{k:v.get('overall',{}).get('FinalScore') for k,v in aggregates.items()}},ensure_ascii=False,indent=2));return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
