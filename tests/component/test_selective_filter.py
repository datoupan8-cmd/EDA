import hashlib,json,tempfile
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from pcb.schema import Component,Pin,Text
from pcb.core.interfaces import ComponentStageOutput
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.component.selective_filter import ComponentStageSelectiveP01,component_features,PROFILE
from pcb.component.local_combination import ComponentStageYolo11mLocalPARSeq
from pcb.component.registry import register_stages
from pcb.core.registry import StageRegistry


class SelectiveFilterTests(unittest.TestCase):
    def model_folder(self,folder,bias=-10):
        raw=json.dumps({'mean':[0]*12,'std':[1]*12,'weights':[0]*12+[bias]}).encode()
        (folder/'model.json').write_bytes(raw)
        (folder/'manifest.json').write_text(json.dumps({'method':'P01','weights_sha256':hashlib.sha256(raw).hexdigest()}),encoding='utf-8')

    def test_frozen_model_feature_schema_for_type_families(self):
        groups={'r':0,'c':1,'l':2,'box':3,'amp':3,'opt':3,'bjt':4,'mosfet_pnp':4,'d':5,'led':5,'gnd':6,'v':6,'other':3}
        for kind,group in groups.items():
            component=Component('R1',kind,(2,3,24,29),name='R1',value='10k',observable_designator='R1',confidence=.71)
            features=component_features(component)
            np.testing.assert_array_equal(features[:4],np.asarray([.71,1,1,0],np.float32))
            self.assertAlmostEqual(float(features[4]),float(np.log(573)/10),places=6)
            np.testing.assert_array_equal(features[5:],np.eye(7,dtype=np.float32)[group])

    def test_registry_binding_is_available_without_switching_default(self):
        registry=StageRegistry();register_stages(registry)
        self.assertIn(PROFILE,registry.versions('component'))

    def test_checked_in_model_and_vendored_sources_match_their_checksums(self):
        stage=ComponentStageSelectiveP01()
        self.assertEqual(len(stage.weights),13)
        source=Path(__file__).resolve().parents[2]/'pcb/component/_vendor/parseq'
        manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
        for relative,expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((source/relative).read_bytes()).hexdigest(),expected,relative)

    def test_actual_run_filters_only_components_and_preserves_shared_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);self.model_folder(folder);stage=ComponentStageSelectiveP01(model_dir=folder)
            pin=Pin('1','IN',(2,3));ground=Component('GND1','gnd',(1,1,5,5),pins=[pin]);resistor=Component('R1','r',(10,10,20,20))
            original=ComponentStageOutput([resistor,ground],[Text('R1',(2,2,8,8))],[object()],{'control':True},{'control':True})
            context=PipelineContext('fixture',100,100,PipelineConfig(),None)
            with patch.object(ComponentStageYolo11mLocalPARSeq,'run',return_value=original):result=stage.run(None,None,context)
            self.assertEqual([c.key for c in result.components],['GND1']);self.assertIs(result.components[0],ground)
            self.assertIs(result.texts,original.texts);self.assertIs(result.roles,original.roles);self.assertIs(result.debug,original.debug)
            self.assertEqual(len(original.components),2);self.assertIs(result.components[0].pins[0],pin)

    def test_corrupt_parameter_file_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);self.model_folder(folder);(folder/'model.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'checksum'):ComponentStageSelectiveP01(model_dir=folder)

    def test_non_finite_parameters_are_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);self.model_folder(folder,bias=float('nan'))
            with self.assertRaisesRegex(ValueError,'invalid'):ComponentStageSelectiveP01(model_dir=folder)


if __name__=='__main__':unittest.main()
