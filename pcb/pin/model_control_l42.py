"""Retained Check-validated old-library model chain; no L42 new catalog.

Selected-only wrapper. Default registry/configuration remain L39.
"""
from __future__ import annotations
import copy
from pathlib import Path


class L42ControlSemantics:
    def __init__(self, baseline=None, reader=None, library=None, templates=None):
        self.baseline=baseline
        self.reader=reader
        self.library=library
        self.templates=templates

    def run(self,component,localization,context):
        raise TypeError('L42 control requires an explicit image, like L39')

    def run_image(self,image,component,localization,context):
        from .frozen_l39.stages import L39Semantics,PROJECT_ROOT
        from .model_reader_l41 import ModelOCR
        from .identity_assets_l42 import load_assets,apply
        from ..schema import Scene
        from ..core.interfaces import PinSemanticsOutput
        if self.baseline is None:self.baseline=L39Semantics()
        original=self.baseline.run_image(image,component,localization,context)
        if self.library is None or self.templates is None:
            self.library,self.templates,_=load_assets(self.baseline.library,expanded=False)
        if self.reader is None:
            cfg=context.resources.get('pin',{})
            cache=Path(cfg.get('model_cache',PROJECT_ROOT/'runs/team_cache/pin_model_control_l42'))
            self.reader=ModelOCR(cache)
        source=Scene(context.width,context.height,original.components,component.texts,
                     diagnostics=copy.deepcopy(original.diagnostics))
        result=apply(image,source,self.reader,self.library,self.templates)
        result.diagnostics['L42_control']={'expanded_catalog_used':False,'GT_used':False,
                                          'selected_only':True,'default_version_changed':False}
        return PinSemanticsOutput(result.components,result.diagnostics)
