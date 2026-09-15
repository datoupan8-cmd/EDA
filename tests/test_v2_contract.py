import ast,copy,sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_v2 import prf,pair_set,component_key_map
from main import official_images
from pcb.coordinates import (opencv_to_target,target_to_opencv,
    opencv_bbox_to_target,target_bbox_to_opencv)
from pcb.schema import Scene,Component,Pin,Net
from pcb.submission import export,validate_strict
from pcb.official_types import OFFICIAL_COMPONENT_TYPES
from pcb.official_target import parse_hypergraph

def sample():
    c=Component('R1','resistor',(10,20,30,40),value='1k',pins=[Pin('1','A',(10,30))])
    return export(Scene(100,80,[c],nets=[Net(['R1.1'],[((10,30),(5,30))])]))

class V2Contract(unittest.TestCase):
    def test_official_taxonomy_has_audited_43_types(self):
        self.assertEqual(len(OFFICIAL_COMPONENT_TYPES),43)
    def test_hypergraph_parser_preserves_dotted_pin_suffix(self):
        refs=parse_hypergraph('(U1.4.7μF)',{'U1'},{'U1':{'pin_4.7μF':object()}})
        self.assertEqual(refs,(('U1','4.7μF'),))
    def test_coordinate_roundtrip(self):
        p=(12.5,27.25);self.assertEqual(target_to_opencv(*opencv_to_target(*p,80),80),p)
        b=(1.5,2.5,30.5,40.5);self.assertEqual(target_bbox_to_opencv(opencv_bbox_to_target(b,80),80),b)
    def test_empty_metric_boundaries(self):
        self.assertEqual(prf(0,3,4)['f1'],0.0)
        self.assertEqual(prf(0,0,0)['f1'],1.0)
    def test_singleton_has_no_pairs(self):
        self.assertEqual(pair_set([{'R1.1'}]),set())
    def test_gnd_serial_is_bbox_matched(self):
        pred={'GND_1':{'type':'gnd','bbox':[50,2,53,4]},'GND_2':{'type':'gnd','bbox':[1,2,3,4]}}
        gt={'GND_1':{'type':'gnd','bbox':[1,2,3,4]},'GND_2':{'type':'gnd','bbox':[50,2,53,4]}}
        self.assertEqual(component_key_map(pred,gt),{'GND_1':'GND_2','GND_2':'GND_1'})
    def test_strict_rejects_nonofficial_type(self):
        d=sample();d['components']['R1']['type']='resistor'
        with self.assertRaises(ValueError):validate_strict(d,(100,80))
    def test_strict_rejects_unknown_ref(self):
        d=sample();d['nets']['net_1']['hyperGraph']='(R1.99)'
        with self.assertRaises(ValueError):validate_strict(d,(100,80))
    def test_strict_accepts_official_parenthesized_pin_suffix(self):
        d=sample();pin=d['pins']['R1'].pop('pin_1');d['pins']['R1']['pin_R(SENSE']=pin
        d['nets']['net_1']['hyperGraph']='(R1.R(SENSE)'
        self.assertTrue(validate_strict(d,(100,80)))
    def test_strict_rejects_three_point_edge(self):
        d=sample();d['nets']['net_1']['edges']['edge_1'].append({'x':1.0,'y':1.0})
        with self.assertRaises(ValueError):validate_strict(d,(100,80))
    def test_holdout_range_is_refused_before_enumeration(self):
        with self.assertRaises(ValueError):official_images(Path('does-not-need-to-exist'),1,151)
    def test_main_has_no_target_reader_dependency(self):
        source=(ROOT/'main.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        modules={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        names={a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        joined=' '.join(str(x) for x in modules|names)
        self.assertNotIn('official_target',joined)
        self.assertNotIn('annotation',joined)
        self.assertNotIn('_target.json',source)

if __name__=='__main__':unittest.main(verbosity=2)
