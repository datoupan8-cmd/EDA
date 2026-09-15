"""Frozen-set stage matrix and oracle ablation. All reported scores are non-official."""
from pathlib import Path
import argparse,copy,json,sys,time,statistics,traceback,hashlib
ROOT=Path(__file__).resolve().parent;RUNTIME=ROOT.parent/'work/runtime'
if RUNTIME.exists():sys.path.insert(0,str(RUNTIME))
from pcb.io import read_image,write_json,write_image
from pcb.vision import OCR,detect_scene
from pcb.vision_v2 import detect_scene_v2,refine_v1_scene,sanitize_low_confidence_pins,parse_pins_for_components,associate_text_for_known_components
from pcb.annotation import load_annotation,reference_nets
from pcb.wire import extract_wire
from pcb.wire_v2 import extract_wire_v2
from pcb.topology import build_topology
from pcb.topology_v2 import build_topology_v2
from pcb.submission import export
from pcb.evaluate import evaluate
from main import overlay

STAGES=['v1_baseline','v1_improved','v2_adaptive_fixed_raw','v2_directional_raw','v2_safe_text_fixed','v2_safe_text_sanitized_fixed','v2_refined_old_wire','v2_refined_adaptive_fixed','v2_refined_directional','v2_refined_labels']
ORACLES=['A_current_all_image','B_gt_component_pred_text_pin','C_gt_component_tip_pred_pin_semantics','D_gt_component_pins_pred_wire_topology','E_complete_prediction_v2']

def snap_oracle_diagnostic(scene,reference,tolerance=5.0):
    """Check whether a snapped point lies on the reference net carrying that pin."""
    import math
    def segdist(p,a,b):
        import numpy as np
        p=np.asarray(p,float);a=np.asarray(a,float);b=np.asarray(b,float);v=b-a;t=float(np.clip(np.dot(p-a,v)/max(np.dot(v,v),1e-9),0,1));return float(np.linalg.norm(p-(a+t*v)))
    by_pin={}
    for net in reference['nets'].values():
        edges=[((x['x'],x['y']),(y['x'],y['y'])) for x,y in net['edges'].values()]
        for ref in net['hyperGraph'][1:-1].split(',') if net['hyperGraph'][1:-1] else []:by_pin.setdefault(ref,[]).extend(edges)
    snaps=scene.diagnostics.get('snaps',[]);tp=0
    for s in snaps:
        q=s.get('to');ref=s.get('pin')
        if q is None or ref not in by_pin:continue
        out=(float(q[0]),scene.height-float(q[1]))
        if any(segdist(out,a,b)<=tolerance for a,b in by_pin[ref]):tp+=1
    return {'snap_oracle_tp':tp,'snap_oracle_pred':len(snaps),'snap_oracle_reference_pins':len(by_pin),
        'snap_oracle_precision':tp/max(1,len(snaps)),'snap_oracle_recall':tp/max(1,len(by_pin)),'snap_oracle_tolerance_px':tolerance}

def run_stage(name,image,visual_v1,visual_safe,visual_safe_sanitized,visual_refined,wire_cache=None):
    source=visual_v1 if name.startswith('v1_') or name.endswith('_raw') else visual_safe if name=='v2_safe_text_fixed' else visual_safe_sanitized if name=='v2_safe_text_sanitized_fixed' else visual_refined
    scene=copy.deepcopy(source)
    color=suppress=None
    if name in ('v1_baseline','v1_improved','v2_refined_old_wire'):
        mask=extract_wire(image,scene);sk=build_topology(scene,mask,'baseline' if name=='v1_baseline' else 'improved')
    else:
        if wire_cache is None:mask,color,suppress=extract_wire_v2(image,scene)
        else:
            mask,color,suppress,wire_diag,traced=wire_cache;scene.diagnostics.update(copy.deepcopy(wire_diag))
        mode='fixed' if name in ('v2_adaptive_fixed_raw','v2_safe_text_fixed','v2_safe_text_sanitized_fixed','v2_refined_adaptive_fixed') else 'directional';sk=build_topology_v2(scene,mask,mode,merge_labels=name=='v2_refined_labels',traced=traced if wire_cache is not None else None)
    return scene,mask,sk,color,suppress

def oracle_scenes(image,ocr,ref_scene,visual_v1,visual_refined):
    # A is the frozen current all-image pipeline; E is the complete V2 image
    # pipeline.  B-D replace exactly one or more upstream prediction layers.
    out={ORACLES[0]:copy.deepcopy(visual_v1)}
    b=copy.deepcopy(ref_scene)
    for c in b.components:c.name=None;c.value=None;c.pins=[]
    b.texts=ocr.recognize(image);b.diagnostics={'oracle':'GT component bbox/key/type only; predicted text, terminal, pin, wire, topology'}
    b.diagnostics.update(associate_text_for_known_components(image,ocr,b.components,b.texts));b.diagnostics.update(parse_pins_for_components(image,ocr,b.components,b.texts,False));out[ORACLES[1]]=b
    cscene=copy.deepcopy(ref_scene)
    for c in cscene.components:c.name=None;c.value=None
    cscene.texts=ocr.recognize(image);cscene.diagnostics={'oracle':'GT component and pin tip/base only; predicted text, pin number/name, wire, topology'}
    cscene.diagnostics.update(associate_text_for_known_components(image,ocr,cscene.components,cscene.texts));cscene.diagnostics.update(parse_pins_for_components(image,ocr,cscene.components,cscene.texts,True));out[ORACLES[2]]=cscene
    d=copy.deepcopy(ref_scene);d.texts=ocr.recognize(image);d.diagnostics={'oracle':'GT component and complete pins; predicted wire/topology'};out[ORACLES[3]]=d
    e=copy.deepcopy(visual_refined);e.diagnostics['oracle']='complete retained V2 prediction from image only';out[ORACLES[4]]=e
    return out

def summarize_variant(scene,metrics,seconds,reference=None,prediction=None):
    dl=metrics['diagnostic_layers'];pins=sum(len(c.pins) for c in scene.components);diag=scene.diagnostics
    result={'metrics':metrics,'runtime_seconds':seconds,'components':len(scene.components),'pins':pins,'nets':len(scene.nets),
        'edges':sum(len(n.segments) for n in scene.nets),'unattached_pins':len(diag.get('unattached_pins',[])),
        'unattached_pin_ratio':diag.get('unattached_pin_ratio',len(diag.get('unattached_pins',[]))/max(1,pins)),
        'wire_pixels':diag.get('wire_pixels',0),'wire_segments':diag.get('wire_segments',0),'wire_groups':diag.get('wire_groups',0),
        'snap_count':len(diag.get('snaps',[])),'false_merge_count':dl['net_pair_false_merges'],'singleton_nets':dl['pred_singleton_nets']}
    if prediction is not None:
        result.update({'exported_pins':sum(len(x) for x in prediction['pins'].values()),'exported_nets':len(prediction['nets']),
            'exported_edges':sum(len(x['edges']) for x in prediction['nets'].values())})
    if reference is not None:result.update(snap_oracle_diagnostic(scene,reference))
    return result

def aggregate(records,names):
    out={}
    for name in names:
        rows=[r[name] for r in records if name in r]
        if not rows:continue
        out[name]={}
        for k in ['runtime_seconds','components','pins','nets','edges','exported_pins','exported_nets','exported_edges','unattached_pin_ratio','wire_pixels','wire_segments','wire_groups','snap_count','false_merge_count','singleton_nets','snap_oracle_precision','snap_oracle_recall']:
            if all(k in x for x in rows):out[name][k]=statistics.mean(x[k] for x in rows)
        for k in ['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','weighted_diagnostic_score']:
            out[name][k]=statistics.mean(x['metrics'][k] for x in rows)
        for k in ['component_key_precision','component_key_recall','component_bbox_precision','component_bbox_recall','pin_number_precision','pin_number_accuracy_over_gt','pin_name_accuracy_over_gt','pin_position_accuracy_over_gt','NetPairF1_singleton_resistant','ExactNetF1']:
            out[name][k]=statistics.mean(x['metrics']['diagnostic_layers'][k] for x in rows)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=ROOT/'splits/v2_dev.json');ap.add_argument('--output_dir',type=Path,default=ROOT/'runs/v2_matrix');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--skip_oracle',action='store_true');ap.add_argument('--oracle_only',action='store_true');ap.add_argument('--stages',help='Comma-separated subset for a final confirmation run')
    a=ap.parse_args();active_stages=a.stages.split(',') if a.stages else STAGES
    if any(x not in STAGES for x in active_stages):ap.error('Unknown stage in --stages')
    cases=json.loads(a.manifest.read_text(encoding='utf8'))['cases'];cases=cases[:a.limit or None];ocr=OCR(a.output_dir/'ocr_cache');stage_rows=[];oracle_rows=[]
    for row in cases:
        case=row['case_id'];folder=a.output_dir/case;start=time.perf_counter()
        try:
            image=read_image(row['image'])
            # Both visual paths complete before any annotation is opened.
            v1=detect_scene(image,ocr);safe=refine_v1_scene(image,ocr,copy.deepcopy(v1),True,False,False);safe_sanitized=sanitize_low_confidence_pins(copy.deepcopy(safe));refined=refine_v1_scene(image,ocr,copy.deepcopy(v1)) if any('refined_' in x for x in active_stages) and not a.oracle_only else safe_sanitized
            ref_scene,data=load_annotation(Path(row['annotation']));ref_scene.nets=reference_nets(ref_scene,data);reference=export(ref_scene)
            write_json(folder/'derived_reference_NOT_OFFICIAL.json',reference)
            sr={'case_id':case}
            if not a.oracle_only:
                wire_scene=copy.deepcopy(refined);cached_mask,cached_color,cached_suppress=extract_wire_v2(image,wire_scene)
                wire_keys=('wire_extractor','wire_palette_candidates','wire_palette_selected','wire_stroke_width','suppressed_pixels')
                from pcb.wire import trace_paths
                wire_cache=(cached_mask,cached_color,cached_suppress,{k:wire_scene.diagnostics[k] for k in wire_keys},trace_paths(cached_mask))
                for name in active_stages:
                    ts=time.perf_counter();scene,mask,sk,color,suppress=run_stage(name,image,v1,safe,safe_sanitized,refined,wire_cache);pred=export(scene);metrics=evaluate(pred,reference);sr[name]=summarize_variant(scene,metrics,time.perf_counter()-ts,reference,pred)
                    write_json(folder/name/'result.json',pred);write_json(folder/name/'diagnostics.json',scene.diagnostics)
                    if case=='0001' and name in ('v1_baseline','v2_refined_labels'):
                        write_image(folder/name/'overlay.png',overlay(image,scene));write_image(folder/name/'wire_mask.png',mask);write_image(folder/name/'skeleton.png',sk.astype('uint8')*255)
                        if color is not None:write_image(folder/name/'wire_color_debug.png',color);write_image(folder/name/'suppressed_regions.png',suppress)
                stage_rows.append(sr)
            if not a.skip_oracle:
                ors={'case_id':case}
                for name,scene in oracle_scenes(image,ocr,ref_scene,v1,safe_sanitized).items():
                    ts=time.perf_counter()
                    if name==ORACLES[0]:
                        mask=extract_wire(image,scene);build_topology(scene,mask,'improved')
                    else:
                        # B-E use the retained V2 topology configuration:
                        # adaptive wire + fixed snapping, with label merge off.
                        mask,_,_=extract_wire_v2(image,scene);build_topology_v2(scene,mask,'fixed',False)
                    pred=export(scene);metrics=evaluate(pred,reference);ors[name]=summarize_variant(scene,metrics,time.perf_counter()-ts,reference,pred)
                oracle_rows.append(ors)
            print(json.dumps({'case':case,'seconds':round(time.perf_counter()-start,2),'v2':sr.get('v2_refined_labels',{}).get('metrics',{}).get('weighted_diagnostic_score')},ensure_ascii=False),flush=True)
        except Exception as e:
            stage_rows.append({'case_id':case,'error':str(e),'traceback':traceback.format_exc()});print(json.dumps(stage_rows[-1]),flush=True)
        if not a.oracle_only:write_json(a.output_dir/'stage_details.json',{'official':False,'cases':stage_rows})
        if not a.skip_oracle:write_json(a.output_dir/'oracle_details.json',{'official':False,'cases':oracle_rows})
    valid_stage=[r for r in stage_rows if 'error' not in r];valid_oracle=[r for r in oracle_rows if 'error' not in r]
    if not a.oracle_only:write_json(ROOT/'reports/v2_metrics.json',{'OFFICIAL':False,'reference':'CVJsonStd-derived diagnostic reference','split_manifest':str(a.manifest),'case_count':len(valid_stage),'stages':aggregate(valid_stage,active_stages)})
    if not a.skip_oracle:write_json(ROOT/'reports/v2_oracle_ablation.json',{'OFFICIAL':False,'reference':'CVJsonStd-derived diagnostic reference','split_manifest':str(a.manifest),'case_count':len(valid_oracle),'oracles':aggregate(valid_oracle,ORACLES)})
    expected=valid_oracle if a.oracle_only else valid_stage
    return 1 if len(expected)!=len(cases) else 0
if __name__=='__main__':raise SystemExit(main())
