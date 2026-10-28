"""Meaningful structure/ambiguity/protection checks with synthetic inputs."""
from dataclasses import asdict
import copy
import unittest

from pcb.schema import Component,Pin,Scene,Text
from pcb.pin.structural_l40 import apply, independent_anchors, compile_symbol
from pcb.pin.structural_l40_r1 import apply as safe_apply,safe_transactions


def fixture():
    template={'model':'PUBLIC123','source_sha256':'independent-source','pins':[
        {'number':str(i),'name':f'SIG{i}','uv':[-.1,i/5]} for i in range(1,5)]}
    pins=[Pin(f'X{i}','',(-10,i*20),base=(0,i*20),side='left',exportable=False) for i in range(1,5)]
    c=Component('U1','box',(0,0,100,100),pins=pins,body_bbox=(0,0,100,100))
    texts=[Text(str(i),(-8,i*20-2,-2,i*20+2),.99,f't{i}') for i in (1,4)]
    scene=Scene(200,200,[c],texts,diagnostics={'L37_public_library':{'models':{'U1':{'library_keys':['PUBLIC123'],'model_confidence':.99}}}})
    return scene,{'PUBLIC123':template}


class StructureTests(unittest.TestCase):
    def test_known_orientation_and_missing_field(self):
        a,t=fixture();before=copy.deepcopy(asdict(a));b=apply(a,t)
        self.assertEqual([p.number for p in b.components[0].pins],['1','2','3','4'])
        self.assertEqual([p.name for p in b.components[0].pins],['SIG1','SIG2','SIG3','SIG4'])
        self.assertEqual(asdict(a),before)
        self.assertEqual([(p.tip,p.base,p.side) for p in a.components[0].pins],[(p.tip,p.base,p.side) for p in b.components[0].pins])
        self.assertEqual(asdict(a)['texts'],asdict(b)['texts'])

    def test_single_anchor_abstains(self):
        a,t=fixture();a.texts=a.texts[:1];b=apply(a,t)
        self.assertEqual(asdict(a.components[0]),asdict(b.components[0]))

    def test_repeated_name_alone_cannot_determine_direction(self):
        a,t=fixture()
        for p in t['PUBLIC123']['pins']:p['name']='GND'
        a.texts=[Text('GND',(-8,i*20-2,-2,i*20+2),.99) for i in (1,4)]
        self.assertEqual(asdict(a.components[0]),asdict(apply(a,t).components[0]))

    def test_one_token_cannot_anchor_two_slots(self):
        hit={'physical_id':'one','confidence':.99,'value':'GND'}
        self.assertEqual(len(independent_anchors({0:[hit],1:[hit]})),1)

    def test_gap_does_not_generate_terminal(self):
        a,t=fixture();del a.components[0].pins[1];b=apply(a,t)
        self.assertEqual(len(b.components[0].pins),3)
        self.assertEqual([p.number for p in b.components[0].pins],['1','3','4'])

    def test_mirrored_order(self):
        a,t=fixture()
        a.texts=[Text(str(5-i),(-8,i*20-2,-2,i*20+2),.99) for i in (1,4)]
        self.assertEqual([p.number for p in apply(a,t).components[0].pins],['4','3','2','1'])

    def test_no_model_and_nonbox_unchanged(self):
        a,t=fixture();a.diagnostics={}
        self.assertEqual(asdict(a.components[0]),asdict(apply(a,t).components[0]))
        a,t=fixture();a.components[0].type='r'
        self.assertEqual(asdict(a.components[0]),asdict(apply(a,t).components[0]))

    def test_unmatched_existing_number_collision_rejects_owner(self):
        a,t=fixture();a.components[0].pins.append(Pin('1','old',(130,20),base=(100,20),side='right'))
        b=apply(a,t)
        self.assertEqual(asdict(a.components[0]),asdict(b.components[0]))
        self.assertEqual(b.diagnostics['L40_structure']['components'][0]['status'],'whole_component_number_collision')

    def test_footprint_is_not_symbol(self):
        with self.assertRaisesRegex(ValueError,'not_schematic_symbol'):
            compile_symbol({'model':'X','url':'public','sha256':'source'},{'result':{'docType':4,'dataStr':{'head':{'docType':4}}}},{})

    def test_safe_subset_keeps_unmatched_owner_number(self):
        a,t=fixture();a.components[0].pins.append(Pin('1','old',(130,20),base=(100,20),side='right'))
        b=safe_apply(a,t)
        self.assertFalse(b.components[0].pins[0].exportable)
        self.assertEqual([p.number for p in b.components[0].pins[1:]],['2','3','4','1'])
        self.assertEqual(asdict(b.components[0].pins[-1]),asdict(a.components[0].pins[-1]))

    def test_mutual_number_release_and_blocking_chain(self):
        pins=[Pin('1','',(0,0)),Pin('2','',(0,0))]
        accepted,_=safe_transactions(pins,{0:('2',),1:('1',)})
        self.assertEqual(len(accepted),2)
        pins.append(Pin('3','',(0,0)))
        accepted,blocked=safe_transactions(pins,{0:('2',),1:('3',)})
        self.assertEqual(accepted,{})
        self.assertEqual(len(blocked),2)


if __name__=='__main__':unittest.main()
