import copy
from dataclasses import asdict
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from pcb.schema import Component,Pin,Scene,Net
from pcb.core.interfaces import SubmissionStageOutput
from pcb.submission import export
from pcb.pin.rc_pair_l45 import POLICY,PairResidualNet,frame,canonical,global_points,sample,valid_pair,PinPointProjectionSubmission


class PairContractTests(unittest.TestCase):
    def owner(self,vertical=False):
        return Component('R1','r',(80,80,100,120) if vertical else (80,80,120,100),
            pins=[Pin('1','1',(90,140) if vertical else (60,90),base=(90,120) if vertical else (80,90),side='bottom' if vertical else 'left'),
                  Pin('2','2',(90,60) if vertical else (140,90),base=(90,80) if vertical else (120,90),side='top' if vertical else 'right')])

    def test_both_orientations_round_trip(self):
        for vertical in (False,True):
            owner=self.owner(vertical);spec=frame(owner,owner.pins,(200,200))
            pts=np.asarray([p.tip for p in owner.pins],np.float32)
            np.testing.assert_allclose(global_points(canonical(pts,spec),spec),pts,atol=1e-5)

    def test_initial_regression_is_exactly_the_frozen_prior(self):
        owner=self.owner();image=np.full((200,200,3),255,np.uint8)
        rgb,prior,features,spec=sample(image,owner,owner.pins)
        model=PairResidualNet().eval()
        with torch.no_grad():out=model(torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/127.5-1,
                torch.from_numpy(prior)[None],torch.from_numpy(features)[None])
        np.testing.assert_array_equal(out.numpy()[0],prior)
        self.assertTrue(valid_pair(global_points(out.numpy()[0],spec),spec,image.shape))

    def test_invalid_pair_or_other_type_is_rejected(self):
        owner=self.owner();spec=frame(owner,owner.pins,(200,200))
        self.assertFalse(valid_pair([[np.nan,20],[40,20]],spec,(200,200)))
        self.assertFalse(valid_pair([[150,90],[50,90]],spec,(200,200)))
        self.assertFalse(valid_pair([[-1,90],[50,90]],spec,(200,200)))
        owner.type='box';self.assertIsNone(frame(owner,owner.pins,(200,200)))

    def test_export_projection_preserves_original_scene_and_networks(self):
        owner=self.owner();scene=Scene(200,200,[owner],nets=[Net(['R1.1','R1.2'],[((60,90),(140,90))])])
        scene.diagnostics['rc_pair_l45']={'export_points':[{'owner':'R1','number':'1','point':[70.,90.]}]}
        before=copy.deepcopy(asdict(scene));original=export(scene)
        class Baseline:
            def run(self,s,c):return SubmissionStageOutput(export(s))
        output=PinPointProjectionSubmission(Baseline()).run(scene,SimpleNamespace()).data
        self.assertEqual(asdict(scene),before)
        self.assertEqual(output['nets'],original['nets']);self.assertEqual(output['components'],original['components'])
        self.assertEqual(output['pins']['R1']['pin_1']['point'],{'x':70.,'y':110.})
        self.assertEqual(output['pins']['R1']['pin_2'],original['pins']['R1']['pin_2'])
