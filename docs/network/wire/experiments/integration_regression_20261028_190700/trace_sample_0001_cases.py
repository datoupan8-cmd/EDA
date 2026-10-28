from __future__ import annotations
import copy, json, sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from docs.network.wire.experiments.palette_recall_20261026_144402.run_wire_palette_replay import DATA,edge_coverage,make_scene,mask_distance,read_gt_edges,saved_prediction
from pcb.core.team_config import load_team_config
from pcb.core.context import PipelineContext
from pcb.io import read_image
from pcb.network_shared.selection import selected_versions
from pcb.topology_stage.versions import TopologyStageV3
from pcb.wire_stage.registry import register_stages as register_wire
from pcb.core.registry import StageRegistry
import pcb.network_shared.composite as composite
OUT=Path(__file__).with_name("sample_0001_cases.json")

def raster(shape,segments):
 m=np.zeros(shape,np.uint8)
 for a,b in segments:cv2.line(m,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),255,1)
 return m

def path_segments(graph,ids):
 return [((float(a[1]),float(a[0])),(float(b[1]),float(b[0]))) for pid in ids for a,b in zip(graph.paths[pid],graph.paths[pid][1:])]

def main():
 image_path=DATA/'1'/'1.png'; image=read_image(image_path); h,w=image.shape[:2]
 result,diag=saved_prediction(1); target=json.loads((DATA/'1'/'1.json').read_text(encoding='utf-8'))
 edges=read_gt_edges(target,h)
 selected={(e['net_id'],e['edge_id']):e for e in edges if (e['net_id'],e['edge_id']) in {('net_4','edge_4'),('net_12','edge_2')}}
 if len(selected)!=2:raise RuntimeError('requested GT edges missing')
 registry=StageRegistry();register_wire(registry)
 config=load_team_config(ROOT/'configs/team.json',root=ROOT); versions=selected_versions(ROOT)
 output={'sample_id':'0001','model_inference_rerun':False,'cases':{}}
 for wire_id in ('v3_frame_guard','v9_branch_salvage_recall'):
  base=make_scene(image,result,diag); context=PipelineContext(image_path.name,w,h,config,None)
  wire=registry.create('wire',wire_id).run(image,base,context)
  for nl_id in ('legacy_v2','path_collinear_v1'):
   scene=copy.deepcopy(base); selected_versions_map=dict(versions);selected_versions_map['net_lines']=nl_id
   stage=TopologyStageV3(network_versions=selected_versions_map);capture={}
   ov,oa=composite.validate_plan,composite.assemble
   def cv(plan,view,graph):ov(plan,view,graph);capture['plan']=plan;capture['graph']=graph
   def ca(plan,geometry):capture['geometry']=geometry;return oa(plan,geometry)
   composite.validate_plan,composite.assemble=cv,ca
   try:stage.run(scene,wire,context)
   finally:composite.validate_plan,composite.assemble=ov,oa
   graph,plan,geometry=capture['graph'],capture['plan'],capture['geometry']
   _,trace=stage.lines.run_with_provenance(graph,plan)
   trace_by_group={g.group_id:g for g in trace.groups}
   allpaths=path_segments(graph,range(len(graph.paths)))
   selectedids=sorted({pid for g in plan.groups for pid in g.path_ids})
   refs=[s for group in graph.reference_segments for s in group]
   geom_by_group={g.group_id:g.segments for g in geometry.groups}
   assembled_group_ids={g.group_id for g in plan.groups if g.pins}
   name=f'{wire_id}__{nl_id}'
   item={'wire_mask_pixels':int((np.asarray(wire.mask)>0).sum()),'graph_path_count':len(graph.paths),
         'graph_reference_segment_count':sum(map(len,graph.reference_segments)),
         'valid_reference_edge_ranges':sum(1 for row in graph.reference_segment_edge_ranges for x in row if x is not None),
         'plan_group_count':len(plan.groups),'groups_without_pins':sum(not g.pins for g in plan.groups),
         'scene_net_count':len(scene.nets),'edges':{}}
   for key,edge in selected.items():
    edge_name=f'{key[0]}/{key[1]}'
    stage_masks={'wire_mask':wire.mask,'graph_paths':raster((h,w),allpaths),
                 'geometry_result':raster((h,w),[s for group in geometry.groups for s in group.segments]),
                 'scene_nets':raster((h,w),[s for net in scene.nets for s in net.segments])}
    masks={k:mask_distance(v) for k,v in stage_masks.items()}
    group_rows=[]
    for group in plan.groups:
     segments=geom_by_group.get(group.group_id,())
     cov=edge_coverage(mask_distance(raster((h,w),segments)),edge,5) if segments else 0.0
     if cov<=0.001:continue
     tg=trace_by_group[group.group_id]
     source_records=[]
     for record in tg.segments:
      for source in record.sources:
       source_records.append({'segment_index':record.segment_index,'output_endpoints':record.endpoints,
         'source_type':source.source_type,'path_id':source.path_id,
         'source_edge_indices':source.source_edge_indices,
         'reference_segment_index':source.reference_segment_index,
         'resolution':source.resolution,'reason':source.reason})
     group_rows.append({'group_id':group.group_id,'pins':group.pins,'path_ids':group.path_ids,
       'geometry_segment_count':len(segments),'spatial_coverage_of_gt_edge_5px':cov,
       'included_in_scene_nets_by_assemble':bool(group.pins),
       'provenance_source_record_count':len(source_records),
       'provenance_resolution_counts':{state:sum(r['resolution']==state for r in source_records) for state in ('complete','unknown')},
       'provenance_records':source_records})
    item['edges'][edge_name]={'gt_endpoints_cv':[edge['a_cv'],edge['b_cv']],
      'coverage_5px_by_stage':{k:edge_coverage(masks[k],edge,5) for k in stage_masks},
      'groups_with_spatial_overlap':group_rows,
      'interpretation':'Group/path/provenance fields are exact pipeline records; overlap with this GT edge is spatial only and does not prove GT electrical membership.'}
   output['cases'][name]=item
 OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'output':str(OUT),'cases':{name:{edge:{'coverage':v['coverage_5px_by_stage'],'groups':[(g['group_id'],g['pins'],g['path_ids'],g['geometry_segment_count'],g['included_in_scene_nets_by_assemble']) for g in v['groups_with_spatial_overlap']]} for edge,v in row['edges'].items()} for name,row in output['cases'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
