from __future__ import annotations
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
from docs.network.wire.experiments.palette_recall_20261026_144402.run_wire_palette_replay import (
    DATA, edge_coverage, make_scene, mask_distance, read_gt_edges, saved_prediction,
)
from pcb.core.context import PipelineContext
from pcb.core.team_config import load_team_config
from pcb.io import read_image
from pcb.network_shared.selection import selected_versions
from pcb.topology_stage.versions import TopologyStageV3
from pcb.wire_stage.retained import (
    WireStageV3FrameGuard, WireStageV8BaselineAnchoredRecall, WireStageV9BranchSalvageRecall,
)
import pcb.network_shared.composite as composite

OUT=Path(__file__).with_name('results.json')
REPORT=Path(__file__).with_name('report.md')
VERSIONS={
 'v3_frame_guard':WireStageV3FrameGuard,
 'v8_baseline_anchored_recall':WireStageV8BaselineAnchoredRecall,
 'v9_branch_salvage_recall':WireStageV9BranchSalvageRecall,
}
STAGES=['wire_mask','graph_paths','graph_reference_segments','plan_selected_paths',
        'plan_selected_reference_segments','geometry_result','assembled_scene_nets']
PIN_RE=re.compile(r'([^(),\s]+)\.([^(),\s]+)')


def gt_data(target,height):
    membership=defaultdict(set); nets={}; edges=[]
    for net_id,net in target.get('nets',{}).items():
        members={f'{c}.{p}' for c,p in PIN_RE.findall(net.get('hyperGraph',''))}
        nets[str(net_id)]=members
        for ref in members: membership[ref].add(str(net_id))
        for edge_id,ends in net.get('edges',{}).items():
            a,b=ends
            edges.append({'net_id':str(net_id),'edge_id':str(edge_id),
                          'a_cv':[float(a['x']),height-float(a['y'])],
                          'b_cv':[float(b['x']),height-float(b['y'])]})
    ambiguous={pin for pin,owners in membership.items() if len(owners)>1}
    return nets,membership,ambiguous,edges


def pairset(partitions,allowed):
    out=set()
    for group in partitions:
        members=sorted(set(group)&allowed)
        out.update(tuple(pair) for pair in itertools.combinations(members,2))
    return out


def connection_metrics(gt_nets,gt_membership,ambiguous,pred_pin_ids,pred_partitions):
    matched=(set(gt_membership)&pred_pin_ids)-ambiguous
    unmatched_gt=set(gt_membership)-pred_pin_ids
    unmatched_pred=pred_pin_ids-set(gt_membership)
    expected=set()
    excluded_gt_pairs=0
    for members in gt_nets.values():
        all_pairs=set(itertools.combinations(sorted(members),2))
        expected.update(tuple(p) for p in itertools.combinations(sorted(members&matched),2))
        excluded_gt_pairs += sum(1 for pair in all_pairs if pair[0] not in matched or pair[1] not in matched)
    predicted=pairset(pred_partitions,matched)
    tp=expected&predicted; fp=predicted-expected; fn=expected-predicted
    return {
       'matched_unambiguous_pin_count':len(matched), 'unmatched_gt_pin_count':len(unmatched_gt),
       'unmatched_predicted_pin_count':len(unmatched_pred), 'ambiguous_gt_pin_count':len(ambiguous),
       'excluded_gt_pair_count_due_to_unmatched_or_ambiguous_pins':excluded_gt_pairs,
       'eligible_gt_pin_pair_count':len(expected), 'eligible_predicted_pin_pair_count':len(predicted),
       'correct_connected_gt_pin_pairs_tp':len(tp), 'missed_required_gt_pin_pairs_fn':len(fn),
       'incorrect_connected_pin_pairs_fp':len(fp),
       'connection_recall':len(tp)/len(expected) if expected else None,
       'connection_precision':len(tp)/len(predicted) if predicted else None,
       'incorrect_connection_rate_fp_over_predicted_pairs':len(fp)/len(predicted) if predicted else None,
       'tp_pairs':[list(p) for p in sorted(tp)], 'fp_pairs':[list(p) for p in sorted(fp)],
       'fn_pairs':[list(p) for p in sorted(fn)],
       'unmatched_gt_pins':sorted(unmatched_gt), 'unmatched_predicted_pins':sorted(unmatched_pred),
       'ambiguous_gt_pins':sorted(ambiguous),
    }


def raster_segments(shape,segments):
    mask=np.zeros(shape,np.uint8)
    for a,b in segments:
        aa=tuple(np.rint(a).astype(int)); bb=tuple(np.rint(b).astype(int))
        cv2.line(mask,aa,bb,255,1)
    return mask


def paths_to_segments(paths):
    result=[]
    for path in paths:
        for (r1,c1),(r2,c2) in zip(path,path[1:]):
            result.append(((float(c1),float(r1)),(float(c2),float(r2))))
    return result


def coverage(mask,edges):
    dist=mask_distance(mask)
    vals=[edge_coverage(dist,e,5) for e in edges]
    return {'macro_gt_edge_coverage_5px':float(np.mean(vals)) if vals else None,
            'gt_edges_ge_80pct_5px':sum(v>=.8 for v in vals),
            'per_edge':vals}


def capture_run(topology,scene,wire,context):
    captured={}
    old_build=composite.build_graph
    def build_capture(mask,traced=None):
        g=old_build(mask,traced)
        captured['graph']=g
        return g
    composite.build_graph=build_capture
    h_orig=topology.hypergraph.run
    l_orig=topology.lines.run
    def h_run(*args,**kwargs):
        logical=h_orig(*args,**kwargs); captured['logical']=logical; return logical
    def l_run(*args,**kwargs):
        geometry=l_orig(*args,**kwargs); captured['geometry']=geometry; return geometry
    topology.hypergraph.run=h_run
    topology.lines.run=l_run
    try:
        topology.run(scene,wire,context)
    finally:
        topology.hypergraph.run=h_orig
        topology.lines.run=l_orig
        composite.build_graph=old_build
    return captured


def main():
    if OUT.exists() or REPORT.exists(): raise FileExistsError('Refusing to overwrite experiment outputs')
    config=load_team_config(ROOT/'configs/team.json')
    net_versions=selected_versions()
    data={'experiment':'V3/V8/V9 electrical-pair and wire-to-NetLine stage-loss diagnostics',
          'runtime':{'python':sys.version,'wire_default':json.loads((ROOT/'configs/wire_topology/current.json').read_text(encoding='utf-8'))['wire'],
                     'net_versions':net_versions,'gt_tolerance_px':5},
          'input':{'dataset_root':str(DATA),'sample_ids':[f'{i:04d}' for i in range(1,14)],
                   'saved_component_pin_results_reused':True,'component_pin_inference_rerun':False,
                   'ground_truth_used_only_for_offline_scoring':True},
          'definitions':{'connectivity_universe':'exact component.pin string match between GT hyperGraph members and saved predicted component pins; exclude ambiguous GT pin IDs',
                         'connection_precision':'TP/(TP+FP) over predicted pairs whose two pin IDs are matched and unambiguous',
                         'connection_recall':'TP/(TP+FN) over GT pairs whose two pin IDs are matched and unambiguous',
                         'unmatched_or_ambiguous':'reported separately and excluded from pair scoring; never assumed correct',
                         'geometry':'fraction of sampled GT edge points within 5 px of stage geometry mask; diagnostic not official score'},
          'limitations':['current Wire checkout WireGraph has no reference_segment_edge_ranges provenance field; source-edge index attribution is unavailable',
                         'path IDs are current replay IDs only; spatial overlap is not proof of exact source provenance',
                         'pair metrics are limited to exact-ID matched pins; unmatchable pins remain unevaluated',
                         'development set only; not official competition scores'],
          'samples':[]}
    conn_sets={name:{'tp':set(),'fp':set(),'fn':set()} for name in VERSIONS}
    edge_stage_totals={name:{s:[] for s in STAGES} for name in VERSIONS}
    first_loss={name:Counter() for name in VERSIONS}; final_loss_examples=[]; pin_match_counts=Counter()
    for case_id in range(1,14):
        imgpath=DATA/str(case_id)/f'{case_id}.png'; image=read_image(imgpath); h,w=image.shape[:2]
        pred,diag=saved_prediction(case_id)
        target=json.loads((DATA/str(case_id)/f'{case_id}.json').read_text(encoding='utf-8'))
        gt_nets,gt_membership,ambiguous,edges=gt_data(target,h)
        ctx=PipelineContext(imgpath.name,w,h,config,None)
        row={'sample_id':f'{case_id:04d}','gt_edge_count':len(edges),'versions':{}}
        for name,stage_cls in VERSIONS.items():
            scene=make_scene(image,pred,diag); wire=stage_cls().run(image,scene,ctx)
            topology=TopologyStageV3(network_versions=net_versions)
            captured=capture_run(topology,scene,wire,ctx)
            graph=captured['graph']; plan=captured['logical'].plan; geometry=captured['geometry']
            path_segments=paths_to_segments(graph.paths)
            ref_segments=[seg for group in graph.reference_segments for seg in group]
            selected_ids=sorted({pid for group in plan.groups for pid in group.path_ids})
            selected_path_segments=paths_to_segments([graph.paths[pid] for pid in selected_ids])
            selected_refs=[seg for group in plan.groups for pid in group.path_ids for seg in graph.reference_segments[pid]]
            geometry_segments=[seg for group in geometry.groups for seg in group.segments]
            assembled_segments=[tuple(seg) for net in scene.nets for seg in net.segments]
            stage_masks={
              'wire_mask':wire.mask,
              'graph_paths':raster_segments((h,w),path_segments),
              'graph_reference_segments':raster_segments((h,w),ref_segments),
              'plan_selected_paths':raster_segments((h,w),selected_path_segments),
              'plan_selected_reference_segments':raster_segments((h,w),selected_refs),
              'geometry_result':raster_segments((h,w),geometry_segments),
              'assembled_scene_nets':raster_segments((h,w),assembled_segments),
            }
            stage_cov={s:coverage(m,edges) for s,m in stage_masks.items()}
            for s in STAGES: edge_stage_totals[name][s].extend(stage_cov[s]['per_edge'])
            thresholds={s:[v>=.8 for v in stage_cov[s]['per_edge']] for s in STAGES}
            for i,was_wire in enumerate(thresholds['wire_mask']):
                if not was_wire: continue
                chain=[thresholds[s][i] for s in STAGES]
                for j in range(1,len(chain)):
                    if chain[j-1] and not chain[j]:
                        first_loss[name][STAGES[j]]+=1; break
                if chain[0] and not chain[-1] and len(final_loss_examples)<120:
                    e=edges[i]
                    final_loss_examples.append({'version':name,'sample_id':f'{case_id:04d}','net_id':e['net_id'],'edge_id':e['edge_id'],
                        'coverage_by_stage':{s:round(stage_cov[s]['per_edge'][i],4) for s in STAGES},
                        'nearby_graph_path_ids':[pid for pid,path in enumerate(graph.paths)
                          if min((np.linalg.norm(np.array((pt[1],pt[0]))-np.array(e['a_cv'])) for pt in path),default=1e9)<10
                             or min((np.linalg.norm(np.array((pt[1],pt[0]))-np.array(e['b_cv'])) for pt in path),default=1e9)<10]})
            pred_ids={f'{c.key}.{p.number}' for c in scene.components for p in c.pins}
            partitions=[tuple(net.pins) for net in scene.nets]
            cm=connection_metrics(gt_nets,gt_membership,ambiguous,pred_ids,partitions)
            for k in ('tp_pairs','fp_pairs','fn_pairs'):
                conn_sets[name][k[:2] if k=='tp_pairs' else ('fp' if k=='fp_pairs' else 'fn')].update(
                  (f'{case_id:04d}:{a}',f'{case_id:04d}:{b}') for a,b in cm[k])
            pin_match_counts[name]+=cm['matched_unambiguous_pin_count']
            geometry_by_id={g.group_id:g.segments for g in geometry.groups}
            plan_group_records=[{'group_id':g.group_id,'pins':list(g.pins),'path_ids':list(g.path_ids),
                'geometry_segment_count':len(geometry_by_id.get(g.group_id,())),
                'geometry_exported_by_assemble':bool(g.pins)} for g in plan.groups]
            row['versions'][name]={
              'graph_path_count':len(graph.paths),'graph_reference_segment_count':len(ref_segments),
              'plan_group_count':len(plan.groups),'plan_selected_path_count':len(selected_ids),
              'selected_path_ids':selected_ids,
              'group_path_ids':{str(g.group_id):list(g.path_ids) for g in plan.groups},
              'plan_groups':plan_group_records,
              'geometry_segments_in_groups_without_pins':sum(g['geometry_segment_count'] for g in plan_group_records if not g['geometry_exported_by_assemble']),
              'geometry_result_segment_count':len(geometry_segments),
              'assembled_scene_net_segment_count':len(assembled_segments),
              'graph_has_reference_segment_edge_ranges':hasattr(graph,'reference_segment_edge_ranges'),
              'stage_geometry':{s:{'macro_gt_edge_coverage_5px':stage_cov[s]['macro_gt_edge_coverage_5px'],
                                    'gt_edges_ge_80pct_5px':stage_cov[s]['gt_edges_ge_80pct_5px']} for s in STAGES},
              'connectivity':cm}
        data['samples'].append(row)
    pair_summary={}
    for name,sets in conn_sets.items():
        tp,fp,fn=sets['tp'],sets['fp'],sets['fn']; denom_pred=len(tp)+len(fp); denom_gt=len(tp)+len(fn)
        pair_summary[name]={'tp':len(tp),'fn':len(fn),'fp':len(fp),
            'recall':len(tp)/denom_gt if denom_gt else None,
            'precision':len(tp)/denom_pred if denom_pred else None,
            'incorrect_connection_rate_fp_over_predicted_pairs':len(fp)/denom_pred if denom_pred else None,
            'matched_unambiguous_pins_across_samples':pin_match_counts[name],
            'tp_pairs':[list(p) for p in sorted(tp)],'fp_pairs':[list(p) for p in sorted(fp)],
            'fn_pairs':[list(p) for p in sorted(fn)]}
    base=conn_sets['v8_baseline_anchored_recall']; cand=conn_sets['v9_branch_salvage_recall']
    base_correct=base['tp']; candidate_correct=cand['tp']; base_wrong=base['fp']; candidate_wrong=cand['fp']
    data['connectivity_summary']=pair_summary
    data['v9_vs_v8_connectivity_delta']={
      'new_correct_pairs':len(candidate_correct-base_correct),
      'new_correct_pair_list':[list(p) for p in sorted(candidate_correct-base_correct)],
      'lost_correct_pairs':len(base_correct-candidate_correct),
      'new_incorrect_pairs':len(candidate_wrong-base_wrong),
      'new_incorrect_pair_list':[list(p) for p in sorted(candidate_wrong-base_wrong)],
      'removed_incorrect_pairs':len(base_wrong-candidate_wrong),
    }
    data['stage_loss_summary']={name:{s:{'macro_gt_edge_coverage_5px':float(np.mean(vals)) if vals else None,
                                    'gt_edges_ge_80pct_5px':sum(v>=.8 for v in vals)} for s,vals in by_stage.items()}
                                for name,by_stage in edge_stage_totals.items()}
    data['first_loss_transition_counts_after_wire_covered_edges']={name:dict(counts) for name,counts in first_loss.items()}
    data['wire_covered_but_final_not_covered_examples']=final_loss_examples
    data['netline_provenance_available']=False
    OUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# V9 connectivity and downstream stage-loss replay','',
      'Reused the authorized 0001–0013 images and saved Component/Pin predictions. Wire, Graph, Topology, NetLine and assembly ran in memory; Component/Pin inference was not rerun. GT was read only after predictions for scoring.', '',
      f"Active network versions: `{net_versions}`. The NetLine backend was `{net_versions.get('net_lines')}`.", '',
      '## Exact-ID connectivity comparison','',
      'Only GT pin identifiers that exactly match a predicted `component.pin` and are not ambiguous across GT nets enter pair scoring. Unmatched predicted/GT pins and ambiguous GT IDs are listed separately and are not counted as correct. Precision/recall are pairwise over that eligible universe.', '',
      '| Wire | TP correct pairs | FN required pairs missed | FP incorrect pairs | Recall | Precision | FP / predicted | matched unambiguous pins |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in pair_summary.items(): lines.append(f"| {name} | {m['tp']} | {m['fn']} | {m['fp']} | {m['recall']:.2%} | {m['precision']:.2%} | {m['incorrect_connection_rate_fp_over_predicted_pairs']:.2%} | {m['matched_unambiguous_pins_across_samples']} |")
    delta=data['v9_vs_v8_connectivity_delta']; lines += ['',f"V9 vs V8: {delta['new_correct_pairs']} newly correct pairs, {delta['new_incorrect_pairs']} newly incorrect pairs, {delta['lost_correct_pairs']} previously correct pairs lost.",'']
    lines += ['## Geometry by pipeline stage','']
    for version, stages in data['stage_loss_summary'].items():
        lines += [f'### {version}', '', '| Stage | mean coverage @5px | edges >=80% |','|---|---:|---:|']
        for stage_name, metrics in stages.items():
            lines.append(f"| {stage_name} | {metrics['macro_gt_edge_coverage_5px']:.2%} | {metrics['gt_edges_ge_80pct_5px']}/744 |")
        lines.append('')
    lines += ['First loss transitions for edges with Wire mask coverage >=80%:', '']
    for version, transitions in data['first_loss_transition_counts_after_wire_covered_edges'].items():
        lines.append(f'**{version}:** ' + (', '.join(f'{k}={v}' for k,v in transitions.items()) or 'none'))
    lines += ['', '## Evidence boundary','',
      'This checkout’s `WireGraph` has `paths` and `reference_segments` but no `reference_segment_edge_ranges`; exact source-edge provenance is unavailable. The saved result records graph path IDs near endpoints only as spatial hints. They are not asserted to be exact source mappings. `reference_segments` are generated from paths by `cv2.approxPolyDP` with epsilon 1 px. Stage coverage can locate geometric loss but not identify a particular source edge with certainty.', '',
      'Connectivity results are limited to exactly matched, unambiguous pin IDs. They do not validate unmatched pins, labels without parseable pin identities, or ambiguous GT memberships. Geometric overlap does not establish net identity. No official score is claimed.', '',
      'Default Wire remains `v3_frame_guard`; no other-owner code or config was modified. No full Component/Pin inference was run.', '',
      'Output files: `results.json` (per-sample stage coverage, selected path IDs, connectivity TP/FN/FP and unmatched/ambiguous pin lists); this report.','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(REPORT),'results':str(OUT),'connectivity':pair_summary,'delta':data['v9_vs_v8_connectivity_delta'],'stages':data['stage_loss_summary'],'loss_transitions':dict(first_loss)},ensure_ascii=False))

if __name__=='__main__': main()
