"""Dataset-free registration and before-pin integration checks for N1/N2."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import ComponentStageOutput,TextStageOutput
from pcb.core.registry import build_default_registry
from pcb.core.team_config import load_team_config
from pcb.schema import Component,Text
from pcb.component.selective_filter import ComponentStageSelectiveP01
from pcb.component.recovery import profile


class RecoveryProfileTests(unittest.TestCase):
    def test_team_default_resolves_through_owned_registry(self):
        config=load_team_config('configs/team.json')
        self.assertEqual(config.component,profile.PROFILE)
        self.assertEqual(config.text,profile.TEXT_PROFILE)
        registry=build_default_registry()
        self.assertIsInstance(registry.create('component',config.component),profile.RecoveryComponentStage)
        self.assertIsInstance(registry.create('text',config.text),profile.RecoveryTextStage)
        self.assertIn('ordered_yolo11m_local_parseq_v1',registry.versions('component'))

    def test_recovery_precedes_pins_and_preserves_shared_outputs(self):
        ground=Component('GND1','gnd',(40,50,50,60),name='GND')
        supply=Component('V1','v',(10,10,20,20),name='VCC')
        texts=[Text('GND',(38,52,52,60),.99),Text('VCC',(8,0,22,8),.99)]
        roles=[object()];debug={'kept':True}
        original=ComponentStageOutput([ground],texts,roles,{
            'component_local_recognition':{'accepted':[]},
            'candidate_audit_before_filter':{'components':[asdict(ground),asdict(supply)]}},debug)
        saved=copy.deepcopy(asdict(ground))
        stage=object.__new__(profile.RecoveryComponentStage)
        stage.before_objects=[ground,supply]
        with tempfile.TemporaryDirectory() as temporary:
            # The normal team entry has no component cache resource; default is portable.
            context=PipelineContext('fixture',100,100,PipelineConfig(),None)
            with patch.object(ComponentStageSelectiveP01,'run',return_value=original),patch.object(profile,'ROOT',Path(temporary)):
                result=stage.run(np.zeros((100,100,3),dtype=np.uint8),TextStageOutput(texts),context)
            self.assertTrue((Path(temporary)/'runs/component_recovery_cache/recovered_components.json').is_file())
        by_key={c.key:c for c in result.components}
        self.assertEqual(by_key['GND1'].value,'GND')
        self.assertEqual(by_key['VREC_0001'].value,'VCC')
        self.assertEqual(by_key['VREC_0001'].bbox,supply.bbox)
        self.assertEqual(by_key['VREC_0001'].pins,[])
        self.assertEqual(asdict(ground),saved)
        self.assertIs(result.texts,original.texts)
        self.assertIs(result.roles,roles)
        self.assertIs(result.debug,debug)
        self.assertEqual(result.diagnostics['deployment_recovery']['counts']['N2_voltage_restoration'],1)

    def test_text_union_keeps_only_vertical_rotation_candidates(self):
        class Mobile:
            last_diagnostics={}
            def recognize_pair(self,image):
                return {'greedy':[Text('R1',(1,1,5,3),.99)],'beam':[Text('R1',(1,1,5,3),.99)]}
        from pcb.component.recovery.tokens import OcrToken
        extra=[OcrToken('R2',(10,1,15,3),.95)]
        rotated=[OcrToken('C1',(20,1,22,11),.95),OcrToken('horizontal',(30,1,40,3),.99)]
        stage=profile.RecoveryTextStage()
        with tempfile.TemporaryDirectory() as temporary:
            context=PipelineContext('fixture',100,100,PipelineConfig(),None)
            with patch.object(profile,'ROOT',Path(temporary)),patch.object(profile,'PairedPPOcrV5Backend',return_value=Mobile()),patch.object(profile,'ServerPPOcrV5Backend',return_value=object()),patch.object(profile,'raw_server_candidates',return_value=extra),patch.object(profile,'generate_method_tokens',return_value=(rotated,{})),patch.object(profile,'release_ocr'),patch.object(profile,'prepare_windows_ort_cuda'):
                result=stage.run(np.zeros((100,100,3),dtype=np.uint8),context)
        self.assertEqual([t.text for t in result.texts],['R1','R2','C1'])
        self.assertEqual(result.diagnostics['vertical_candidates'],1)


if __name__=='__main__':unittest.main()
