"""Deployment adapter for the frozen rotation_vertical + N1 + N2 recipe.

All recovery runs before pin/network inference and uses image observations only.
The source rules and thresholds are copied unchanged from the verified project.
"""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path

from pcb.component.ocr_ablation import PairedPPOcrV5Backend
from pcb.component.alternative_ocr import ROOT
from pcb.component.ocr_server import ServerPPOcrV5Backend, prepare_windows_ort_cuda
from pcb.component.selective_filter import ComponentStageSelectiveP01
from pcb.core.interfaces import ComponentStageOutput, TextStageOutput
from pcb.schema import Scene, Text
from pcb.submission import export, validate_strict
from .candidate_pool import merge_literal_candidates, rescue_observed_proposals
from .power_recovery import recover_power_values, restore_voltage_symbols
from .tokens import OcrToken
from .text_candidate_views import generate_method_tokens
from .raw_candidates import raw_server_candidates

PROFILE='rotation_vertical_n1_n2_v1'
TEXT_PROFILE='recovery_rotation_vertical_v1'

def release_ocr(backend):
    backend.engine=backend.decoder=None
    gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()

class RecoveryTextStage:
    def __init__(self):self.mobile=self.server=None

    def run(self,image,context):
        import torch
        if torch.cuda.is_available():prepare_windows_ort_cuda()
        settings=context.resources.get('component',{})
        root=Path(settings.get('cache_dir',ROOT/'runs/component_recovery_cache'))/'updated_text'
        if self.mobile is None:
            self.mobile=PairedPPOcrV5Backend(root/'mobile',normalization='legacy',use_cuda=torch.cuda.is_available())
            self.server=ServerPPOcrV5Backend(root/'server',profile='server_full',use_cuda=torch.cuda.is_available(),decoding='greedy')
        pair=self.mobile.recognize_pair(image)
        mobile_diagnostics=dict(self.mobile.last_diagnostics)
        release_ocr(self.mobile)
        extra=raw_server_candidates(self.server,image,root/'raw_server.json')
        release_ocr(self.server)
        base=[OcrToken(t.text,tuple(t.bbox),t.score,'ppocrv5_mobile',variant)
              for variant in ('greedy','beam') for t in pair[variant]]
        tokens=merge_literal_candidates(base,extra)
        progress=context.resources.get('progress')
        def notify(message):
            if progress:progress('recovery_text_progress',message=message)
        rotated,rotation_diagnostics=generate_method_tokens('rotation',self.server,image,tokens,[],{},root/'views',progress=notify)
        vertical=[t for t in rotated if t.bbox_xyxy[3]-t.bbox_xyxy[1]>=1.5*(t.bbox_xyxy[2]-t.bbox_xyxy[0])]
        tokens=merge_literal_candidates(tokens,vertical)
        release_ocr(self.server)
        diagnostic={'profile':TEXT_PROFILE,'mobile':mobile_diagnostics,'rotation':rotation_diagnostics,
                    'vertical_candidates':len(vertical),'token_count':len(tokens),'GT_used_in_inference':False}
        return TextStageOutput([Text(t.text,tuple(t.bbox_xyxy),t.confidence,t.source) for t in tokens],diagnostic)

def enrich_components(prediction,before,tokens,height):
    """Replay the three unchanged, image-only recovery functions in fixed order."""
    recovered,rescue=rescue_observed_proposals(prediction,before,height)
    recovered,voltage=restore_voltage_symbols(recovered,before,height)
    recovered,values=recover_power_values(recovered,tokens,height)
    return recovered,{'observed_proposal_rescue':rescue,'N2_voltage_restoration':voltage,'N1_power_values':values}

def assemble_components(original,before_objects,prediction,height):
    """Keep native geometry and metadata while applying recovered schema fields."""
    existing={c.key:c for c in original.components}
    before={c.key:c for c in before_objects}
    components=[]
    for key,row in prediction['components'].items():
        source=existing.get(key,before.get(key))
        if source is None:
            candidates=[c for c in before_objects if c.type==row['type'] and
                        [round(v,3) for v in (c.bbox[0],height-c.bbox[3],c.bbox[2],height-c.bbox[1])]==row['bbox']]
            if len(candidates)!=1:raise ValueError('Ambiguous recovered native component: '+key)
            source=candidates[0]
        # Preserve native coordinates and detector/identity provenance for pins.
        components.append(replace(source,key=key,name=row['Name'],value=row['value'],pins=list(source.pins)))
    return components

class RecoveryComponentStage(ComponentStageSelectiveP01):
    def filter(self,original):
        self.before_objects=list(original.components)
        result=super().filter(original)
        result.diagnostics['candidate_audit_before_filter']={'components':[asdict(c) for c in original.components]}
        return result

    def run(self,image,text,context):
        output=super().run(image,text,context)
        height,width=image.shape[:2]
        prediction=export(Scene(width,height,output.components,output.texts))
        tokens=[OcrToken(t.text,tuple(t.bbox),t.score,t.source_id) for t in text.texts]
        tokens += [OcrToken(t['text'],tuple(t['bbox']),t['confidence'],'private_parseq')
                   for t in output.diagnostics['component_local_recognition']['accepted']]
        before=output.diagnostics['candidate_audit_before_filter']['components']
        recovered,events=enrich_components(prediction,before,tokens,height)
        validate_strict(recovered,(width,height))
        components=assemble_components(output,self.before_objects,recovered,height)
        actual=export(Scene(width,height,components,output.texts))
        if actual!=recovered:raise ValueError('Recovery adapter/export schema mismatch')
        cache=Path(context.resources.get('component',{}).get('cache_dir',ROOT/'runs/component_recovery_cache'))
        cache.mkdir(parents=True,exist_ok=True)
        (cache/'recovered_components.json').write_text(json.dumps(recovered,ensure_ascii=False,indent=2),encoding='utf8')
        diagnostics={**output.diagnostics,'deployment_recovery':{'profile':PROFILE,'GT_used_in_inference':False,
                     'events':events,'counts':{name:len(rows) for name,rows in events.items()},'text':text.diagnostics}}
        return replace(output,components=components,diagnostics=diagnostics)
