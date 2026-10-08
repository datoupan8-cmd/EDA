import unittest,sys,tempfile,json,copy
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
runtime=root.parent/'work/runtime'
try:
    import scipy, skimage  # noqa: F401
except ImportError:
    if runtime.exists():sys.path.insert(0,str(runtime))
import numpy as np
import cv2
from pcb.schema import Scene,Component,Pin,Net
from pcb.topology import build_topology
from pcb.wire import extract_wire
from pcb.submission import export,validate
from pcb.annotation import unique_keys,pin_numbers,load_annotation
from pcb.evaluate import evaluate
from pcb.io import case_names,discover_images
from pcb.wire_v2 import extract_wire_v2
from pcb.topology_v2 import build_topology_v2
from pcb.vision_v2 import sanitize_low_confidence_pins

def scene_at(points,labels=None):
    return Scene(100,100,[Component('P'+str(i),'connector',(1,1,3,3),pins=[Pin('1',None,p)],net_label=labels[i] if labels else None) for i,p in enumerate(points)])

class Contracts(unittest.TestCase):
    def test_mixed_dataset_filenames_are_not_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for name in ['0001-KiCad.png','hashedimage.jpg','u1_pins_result.png']:(p/name).touch()
            self.assertEqual(len(discover_images(p)),2)
    def test_flat_image_directory_no_overwrite(self):
        self.assertEqual(case_names([Path('input/a.png'),Path('input/b.png')]),['a','b'])
    def test_output_origin_and_capital_name(self):
        s=scene_at([(20,30)]);d=export(s)
        self.assertEqual(d['pins']['P0']['pin_1']['point'],{'x':20.,'y':70.})
        self.assertIn('Name',d['components']['P0']);self.assertNotIn('name',d['components']['P0'])
    def test_all_duplicates_numbered(self):
        s=scene_at([(20,30),(30,30)]);s.components[0].key=s.components[1].key='GND'
        unique_keys(s.components);self.assertEqual([c.key for c in s.components],['GND_1','GND_2'])
    def test_pin_numbers(self):
        self.assertEqual(pin_numbers('5,6,7'),['5','6','7'])
        self.assertEqual(pin_numbers('B7'),['B7'])
        for bad in ['R12 100','RP2350B','R10 1k','596995','RM2']:self.assertEqual(pin_numbers(bad),[])
    def test_reference_identity_and_wrong_net_penalty(self):
        s=scene_at([(20,30),(60,30)]);s.nets=[Net(['P0.1'],[((20,30),(30,30))]),Net(['P1.1'],[((50,30),(60,30))])]
        d=export(s);self.assertEqual(evaluate(d,d)['NetLineF1'],1.)
        wrong=copy.deepcopy(d);wrong['nets']['net_1']['edges'],wrong['nets']['net_2']['edges']=wrong['nets']['net_2']['edges'],wrong['nets']['net_1']['edges']
        self.assertEqual(evaluate(wrong,d)['NetLineF1'],0.)
    def test_same_pin_cannot_belong_to_two_nets(self):
        s=scene_at([(20,30)]);s.nets=[Net(['P0.1'],[]),Net(['P0.1'],[])]
        with self.assertRaises(ValueError):export(s)
    def test_unrecognized_terminal_is_not_exported(self):
        s=scene_at([(20,30)]);s.components[0].pins[0].number='UNK1';s.components[0].pins[0].exportable=False
        s.nets=[Net(['P0.UNK1'],[((20,30),(30,30))])];d=export(s)
        self.assertEqual(d['pins']['P0'],{});self.assertEqual(d['nets'],{})
    def test_implausible_ic_numeric_outlier_is_not_exported(self):
        pins=[Pin(str(i),'',(10+i,10),inferred_number=False) for i in range(1,11)]+[Pin('178','',(30,10),inferred_number=False)]
        s=Scene(100,100,[Component('U1','ic',(5,5,40,30),pins=pins)]);sanitize_low_confidence_pins(s);d=export(s)
        self.assertNotIn('pin_178',d['pins']['U1'])
    def test_singletons_do_not_inflate_pair_metric(self):
        s=scene_at([(10,10),(20,20)]);gt=export(s);pred=copy.deepcopy(gt)
        self.assertEqual(evaluate(pred,gt)['diagnostic_layers']['NetPairF1_singleton_resistant'],1.)
    def test_raw_origin_explicit(self):
        d={'canvas':{'width':100,'height':100},'components':{'C_1':{'category':'R','location':{'x1':10,'y1':20,'x2':20,'y2':40},'pins':{'P_1':{'tip':[15,20]}},'associatedTexts':{}}}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.json';path.write_text(json.dumps(d))
            a,_=load_annotation(path);b,_=load_annotation(path,'bottom-left')
            self.assertEqual(a.components[0].pins[0].tip,(15.,20.));self.assertEqual(b.components[0].pins[0].tip,(15.,80.))

class Topology(unittest.TestCase):
    def test_blue_dot_preserved_on_green_wire(self):
        image=np.full((100,100,3),255,np.uint8)
        cv2.line(image,(10,50),(90,50),(0,150,0),1)
        cv2.line(image,(50,10),(50,90),(0,150,0),1)
        cv2.circle(image,(50,50),3,(180,0,0),-1)
        s=scene_at([(10,50),(90,50),(50,10),(50,90)])
        # Component placeholders lie at the image corner, outside the fixture wires.
        mask=extract_wire(image,s);build_topology(s,mask)
        self.assertEqual(s.diagnostics['blue_junction_dots'],1)
        self.assertEqual(len(s.nets),1)
    def test_t_junction(self):
        mask=np.zeros((100,100),np.uint8);cv2.line(mask,(10,50),(90,50),255,1);cv2.line(mask,(50,10),(50,50),255,1)
        s=scene_at([(10,50),(90,50),(50,10)]);build_topology(s,mask)
        self.assertEqual(len(s.nets),1);self.assertEqual(len(s.nets[0].pins),3)
    def test_cross_without_dot(self):
        mask=np.zeros((100,100),np.uint8);cv2.line(mask,(10,50),(90,50),255,1);cv2.line(mask,(50,10),(50,90),255,1)
        s=scene_at([(10,50),(90,50),(50,10),(50,90)]);baseline=copy.deepcopy(s)
        build_topology(s,mask);build_topology(baseline,mask,'baseline')
        self.assertEqual({frozenset(n.pins) for n in s.nets},{frozenset(['P0.1','P1.1']),frozenset(['P2.1','P3.1'])})
        self.assertEqual(len(baseline.nets),1)
    def test_cross_dot_connects(self):
        mask=np.zeros((100,100),np.uint8);cv2.line(mask,(10,50),(90,50),255,1);cv2.line(mask,(50,10),(50,90),255,1);cv2.circle(mask,(50,50),4,255,-1)
        s=scene_at([(10,50),(90,50),(50,10),(50,90)]);build_topology(s,mask)
        self.assertEqual(len(s.nets),1)
    def test_snap_preserves_pin_and_long_gap_not_bridged(self):
        mask=np.zeros((100,100),np.uint8);cv2.line(mask,(10,50),(35,50),255,1);cv2.line(mask,(45,50),(90,50),255,1)
        s=scene_at([(10,47),(90,50)]);build_topology(s,mask)
        self.assertEqual(len(s.nets),2);self.assertEqual(s.components[0].pins[0].tip,(10,47))
        self.assertEqual(len(s.diagnostics['unattached_pins']),0)
    def test_flying_label_no_fabricated_edge(self):
        s=scene_at([(10,10),(90,90)],['VCC','VCC']);build_topology(s,np.zeros((100,100),np.uint8))
        self.assertEqual(len(s.nets),1);self.assertEqual(s.nets[0].segments,[])
    def test_direct_contact_degenerate_line(self):
        s=scene_at([(20,30),(20,30)]);build_topology(s,np.zeros((100,100),np.uint8))
        self.assertEqual(len(s.nets),1);self.assertEqual(s.nets[0].segments,[((20,30),(20,30))])
    def test_v2_directional_snap_rejects_wire_behind_pin(self):
        mask=np.zeros((100,100),np.uint8);cv2.line(mask,(38,20),(38,80),255,1)
        c=Component('U1','ic',(40,40,60,60),pins=[Pin('1','',(40,50),base=(50,50),side='left')])
        fixed=Scene(100,100,[copy.deepcopy(c)]);directional=Scene(100,100,[copy.deepcopy(c)])
        build_topology_v2(fixed,mask,'fixed',False);build_topology_v2(directional,mask,'directional',False)
        self.assertEqual(len(fixed.diagnostics['snaps']),1);self.assertEqual(len(directional.diagnostics['snaps']),1)
        # Move the candidate behind the outward direction: fixed accepts it,
        # directional mode rejects it.
        mask[:]=0;cv2.line(mask,(42,20),(42,80),255,1)
        fixed=Scene(100,100,[copy.deepcopy(c)]);directional=Scene(100,100,[copy.deepcopy(c)])
        build_topology_v2(fixed,mask,'fixed',False);build_topology_v2(directional,mask,'directional',False)
        self.assertEqual(len(fixed.diagnostics['snaps']),1);self.assertEqual(len(directional.diagnostics['snaps']),0)
    def test_adaptive_wire_ignores_colored_page_frame(self):
        image=np.full((100,100,3),255,np.uint8);cv2.rectangle(image,(0,0),(99,99),(0,0,180),1);cv2.line(image,(20,50),(80,50),(0,160,0),1)
        c1=Component('J1','connector',(14,45,20,55),pins=[Pin('1','',(20,50),base=(15,50))]);c2=Component('J2','connector',(80,45,86,55),pins=[Pin('1','',(80,50),base=(85,50))])
        s=Scene(100,100,[c1,c2]);mask,_,_=extract_wire_v2(image,s)
        self.assertTrue(mask[49:52,49:52].any());self.assertFalse(mask[0,50]);self.assertGreater(sum(x['frame_pixels_removed'] for x in s.diagnostics['wire_palette_candidates']),0)

if __name__=='__main__':unittest.main(verbosity=2)
