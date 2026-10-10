"""Coordinate, dense point, contract and data split guards for L1."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT/'experiments'),str(ROOT)]
import numpy as np
import torch
from pin_learned_locator_l1 import (POLICY, SIDES, PinPointNet, strip_specs,
    strip_to_global, global_to_strip, sample_strip, point_targets, nearest_side,
    decode_map, merge_points, point_loss, locate_boxes)
from pcb.coordinates import target_to_opencv,opencv_to_target
from pcb.core.interfaces import PinLocalizationOutput
from pcb.data_policy import assert_allowed_case_id,split_for_case
from pcb.schema import Component


class LearnedLocatorTests(unittest.TestCase):
    def setUp(self):
        self.box=(100.,100.,300.,300.); self.image=np.full((400,400,3),255,np.uint8)

    def test_four_side_inverse(self):
        specs=strip_specs(self.box,self.image.shape)
        for s in specs:
            x,y=strip_to_global(120.25,60.5,s)
            c,r=global_to_strip((x,y),s)
            self.assertAlmostEqual(c,120.25); self.assertAlmostEqual(r,60.5)
        self.assertEqual(set(s['side'] for s in specs),set(SIDES))

    def test_lower_left_flip_once(self):
        self.assertEqual(target_to_opencv(15,20,400),(15,380))
        self.assertEqual(opencv_to_target(15,380,400),(15,20))

    def test_owner_hint_and_shape(self):
        patch=sample_strip(self.image,strip_specs(self.box,self.image.shape)[0])
        self.assertEqual(patch.shape,(4,128,256)); self.assertTrue(np.isfinite(patch).all())
        self.assertGreater(patch[3].sum(),0); self.assertLess(patch[3].sum(),128*256)

    def test_exact_heatmap_fractional_offset(self):
        spec=strip_specs(self.box,self.image.shape)[0]
        point=strip_to_global(128.25,58.3,spec)
        heat,offset,mask,count=point_targets([point],spec)
        self.assertEqual(count,1); self.assertEqual(heat[0,58,128],1.)
        self.assertAlmostEqual(float(offset[0,58,128]),.25)
        self.assertAlmostEqual(float(offset[1,58,128]),.3,places=5)
        self.assertEqual(mask.sum(),1.)

    def test_dense_separate_points(self):
        spec=strip_specs(self.box,self.image.shape)[0]
        points=[strip_to_global(x,58,spec) for x in (115,125,135)]
        heat,off,mask,count=point_targets(points,spec)
        self.assertEqual(count,3); self.assertEqual(mask.sum(),3.)

    def test_peak_decode_returns_contract(self):
        spec=strip_specs(self.box,self.image.shape)[0]
        out=torch.full((3,128,256),-20.); out[1:]=0.
        out[0,58,128]=10.; rows=decode_map(out,spec,self.image.shape)
        self.assertEqual(len(rows),1); t=rows[0]
        self.assertEqual(t['side'],'left'); self.assertEqual(t['base'][0],100.)
        self.assertLess(t['tip'][0],100.); self.assertTrue(np.isfinite(t['tip']).all())

    def test_tiled_duplicate_suppression(self):
        row=dict(tip=(90.,200.),base=(100.,200.),side='left',point_confidence=.9)
        other={**row,'tip':(90.5,200.),'point_confidence':.8}
        self.assertEqual(len(merge_points([row,other],1.)),1)

    def test_nonbox_unchanged(self):
        c=Component('R1','r',self.box); original=[{'tip':(80.,200.),'base':(100.,200.),'side':'left','method':'old'}]
        output=locate_boxes(self.image,[c],PinLocalizationOutput([(c,copy.deepcopy(original))]),PinPointNet(),'cpu')
        self.assertEqual(output.terminals[0][1],original); self.assertEqual(c.body_bbox,None)

    def test_network_loss_finite_and_gradients(self):
        torch.set_num_threads(2); model=PinPointNet()
        output=model(torch.zeros((1,4,128,256)))
        self.assertEqual(output.shape,(1,3,128,256))
        heat=torch.zeros((1,1,128,256)); heat[0,0,64,64]=1.
        mask=heat.clone(); off=torch.zeros((1,2,128,256))
        loss=point_loss(output,heat,off,mask); self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_empty_label_loss(self):
        output=torch.zeros((1,3,128,256),requires_grad=True)
        loss=point_loss(output,torch.zeros((1,1,128,256)),torch.zeros((1,2,128,256)),torch.zeros((1,1,128,256)))
        self.assertTrue(torch.isfinite(loss))

    def test_split_and_sealed_guard(self):
        train={i for i in range(1,151) if split_for_case(i)=='train'}
        val={i for i in range(1,151) if split_for_case(i)=='dev'}
        self.assertEqual((len(train),len(val)),(120,30)); self.assertFalse(train&val)
        for i in (0,151,200):
            with self.assertRaises(PermissionError): assert_allowed_case_id(i)

    def test_no_gt_or_filename_in_localizer_inputs(self):
        import inspect
        signature=inspect.signature(locate_boxes)
        self.assertFalse(any(s in signature.parameters for s in ('target','gt','case_id','source_name')))
        source=(ROOT/'experiments/pin_learned_locator_l1.py').read_text(encoding='utf-8')
        self.assertNotIn('read_text(',source); self.assertNotIn('torch.load(',source)

    def test_training_refuses_validation_target_before_open(self):
        from pin_learned_locator_l1_train import Guard
        from types import SimpleNamespace
        guard=object.__new__(Guard); guard.target_case=None; guard.blocked=0
        with self.assertRaises(PermissionError):
            guard.target(SimpleNamespace(case_id=5,target_path=Path('must_not_open_target.json')))

    def test_audit_blocks_holdout_and_target_inference(self):
        from pin_learned_locator_l1_train import Guard
        guard=object.__new__(Guard); guard.target_case=None; guard.blocked=0
        with self.assertRaises(PermissionError): guard.audit('open',('C:/data/200_train_cases/0151/x.png',))
        with self.assertRaises(PermissionError): guard.audit('open',('C:/data/200_train_cases/0005/x_target.json',))
        with self.assertRaises(PermissionError): guard.audit('open',('C:/data/10_GTcase/x.png',))

    def test_large_box_tiles_cover_tangent(self):
        specs=strip_specs((100.,100.,300.,1500.),(1800,800,3))
        left=[s for s in specs if s['side']=='left']
        self.assertGreater(len(left),1)
        for y in np.arange(100,1501,25):
            self.assertTrue(any(0<=global_to_strip((90.,float(y)),s)[0]<256 for s in left))


if __name__=='__main__': unittest.main()
