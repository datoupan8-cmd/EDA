"""Frozen P01 Component filter; no training data or other sweep models needed."""
from dataclasses import replace
import hashlib,json,math
from pathlib import Path
import numpy as np
from .alternative_ocr import ROOT
from .latest.text_assignment import BOX_FAMILY
from .local_combination import ComponentStageYolo11mLocalPARSeq

PROFILE='ordered_yolo11m_local_parseq_p01_v1'
MODEL_DIR=ROOT/'models/component/component_selective120_candidate'


def component_features(component):
    kind=component.type
    group=(('r','c','l').index(kind) if kind in {'r','c','l'} else 3 if kind in BOX_FAMILY
           else 4 if kind.startswith(('mosfet','bjt')) else 5 if kind in {'d','led','esd'} else 6)
    return np.asarray([component.confidence,float(component.observable_designator is not None),float(component.value is not None),
        float(component.model is not None or (component.name is not None and component.name!=component.key)),
        math.log1p((component.bbox[2]-component.bbox[0])*(component.bbox[3]-component.bbox[1]))/10]+
        [float(group==i) for i in range(7)],np.float32)


class ComponentStageSelectiveP01(ComponentStageYolo11mLocalPARSeq):
    def __init__(self,model_dir=None,**kwargs):
        super().__init__(**kwargs)
        folder=Path(model_dir or MODEL_DIR);metadata=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        raw=(folder/'model.json').read_bytes()
        if metadata['method']!='P01' or hashlib.sha256(raw).hexdigest()!=metadata['weights_sha256']:raise ValueError('P01 parameter checksum mismatch')
        model=json.loads(raw);self.mean=np.asarray(model['mean']);self.std=np.asarray(model['std']);self.weights=np.asarray(model['weights'])
        if self.mean.shape!=(12,) or self.std.shape!=(12,) or self.weights.shape!=(13,):raise ValueError('P01 feature schema mismatch')
        if not all(np.isfinite(a).all() for a in (self.mean,self.std,self.weights)) or not (self.std>0).all():raise ValueError('P01 parameters invalid')
        self.parameter_sha256=metadata['weights_sha256']

    def filter(self,original):
        if not original.components:return original
        values=np.asarray([component_features(c) for c in original.components]);logits=((values-self.mean)/self.std)@self.weights[:-1]+self.weights[-1]
        probabilities=1/(1+np.exp(-np.clip(logits,-40,40)))
        kept=[c for c,p in zip(original.components,probabilities) if c.type=='gnd' or p>=.5]
        return replace(original,components=kept,diagnostics={**original.diagnostics,'component_selective_filter':{
            'method':'P01','before':len(original.components),'after':len(kept),'threshold':.5,
            'parameters_sha256':self.parameter_sha256,'shared_texts_and_roles_preserved':True,'GT_used_in_inference':False}})

    def run(self,image,text,context):return self.filter(super().run(image,text,context))
