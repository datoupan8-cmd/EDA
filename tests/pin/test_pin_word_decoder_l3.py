"""L3 interface/structural tests; no competition data access."""
import copy
import inspect
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT)]
from pcb.schema import Component,Pin,Text,Scene
from pin_word_reader_l3 import strip_regions,group_words,reading_options,canonical_tokens,same_instance,WholeWordOCR,stage_b
from pin_word_decoder_l3 import replacement_allowed,solve_options,role_options,geometry,stage_c


def reading(text,box,family='global',score=.99,complete=True):
    return dict(text=text,bbox=list(box),family=family,score=score,complete=complete,side=None,angle=None)


def option(i,number,word,score=4.,side='left',tangent=10.,fallback=False):
    return dict(terminal=i,number=number,name='GND',number_word=word,name_word=None,
                number_tangent=tangent if word else None,name_tangent=None,
                score=score,side=side,fallback=fallback)


class L3Tests(unittest.TestCase):
    def test_strip_includes_complete_old_word(self):
        c=Component('U1','box',(100,100,200,200));ts=[dict(side='left',base=(100,110))]
        t=Text('GND',(140,102,170,120))
        regions=strip_regions(c,ts,[t],(300,300,3))
        self.assertTrue(any(r['roi'][2]>=172 for r in regions))

    def test_words_never_midpoint_cut(self):
        c=Component('U1','box',(100,100,200,200));ts=[dict(side='left',base=(100,110)),dict(side='left',base=(100,120))]
        regions=strip_regions(c,ts,[],(300,300,3))
        self.assertEqual(len(regions),1)
        self.assertLess(regions[0]['roi'][1],100);self.assertGreater(regions[0]['roi'][3],120)

    def test_fragment_and_full_word_one_instance(self):
        words=group_words([reading('GND',(100,20,130,35)),reading('GN',(100,20,120,35),'strip',complete=False)])
        self.assertEqual(len(words),1);self.assertEqual(canonical_tokens(words)[0].text,'GND')

    def test_incomplete_new_word_cannot_export(self):
        words=group_words([reading('GN',(100,20,120,35),'strip',complete=False)])
        self.assertEqual(canonical_tokens(words),[])

    def test_global_preserved_against_single_new_guess(self):
        words=group_words([reading('GND',(100,20,130,35)),reading('GN',(100,20,130,35),'strip',.999)])
        self.assertEqual(canonical_tokens(words)[0].text,'GND')

    def test_two_families_corroboration(self):
        words=group_words([reading('GND',(100,20,130,35)),reading('GND',(100,20,130,35),'direct')])
        self.assertTrue(reading_options(words[0])[0]['corroborated'])

    def test_adjacent_separate_words_not_fused(self):
        self.assertFalse(same_instance((10,10,20,20),(21,10,45,20)))

    def test_proper_substring_replacement_rejected(self):
        old=Pin('2','GND',(0,0));n={'text':'2','families':['strip','direct']};s={'text':'GN','corroborated':True}
        self.assertFalse(replacement_allowed(old,n,s,5.,3.)[0])

    def test_number_change_needs_direct_and_strip(self):
        old=Pin('9','CE',(0,0));n={'text':'3','families':['global','direct']};s={'text':'CE','corroborated':True}
        self.assertFalse(replacement_allowed(old,n,s,5.,3.)[0])

    def test_solver_word_and_number_uniqueness(self):
        terms=[dict(side='left',base=(0,10)),dict(side='left',base=(0,20))]
        ops=[option(0,'1','word',5.),option(0,None,None,0.,fallback=True),
             option(1,'2','word',4.),option(1,None,None,0.,fallback=True)]
        chosen,trace=solve_options(ops,terms)
        self.assertEqual(trace['status'],'optimal');self.assertEqual(len(chosen),2)
        self.assertEqual(sum(p['number_word']=='word' for p in chosen),1)

    def test_cross_side_word_uniqueness(self):
        terms=[dict(side='left',base=(0,10)),dict(side='top',base=(10,0))]
        ops=[option(0,'1','word',5.),option(0,None,None,0.,fallback=True),
             option(1,'2','word',4.,side='top'),option(1,None,None,0.,side='top',fallback=True)]
        chosen,_=solve_options(ops,terms)
        self.assertEqual(sum(p['number_word']=='word' for p in chosen),1)

    def test_order_preserving_with_gaps(self):
        terms=[dict(side='left',base=(0,10)),dict(side='left',base=(0,20))]
        ops=[option(0,'1','a',5.,tangent=20),option(0,None,None,0.,fallback=True),
             option(1,'2','b',4.,tangent=10),option(1,None,None,0.,fallback=True)]
        chosen,_=solve_options(ops,terms)
        self.assertEqual(sum(p['number'] is not None for p in chosen),1)

    def test_inside_signal_is_not_number(self):
        c=Component('U1','box',(100,100,200,200));term=dict(side='left',base=(100,120))
        words=group_words([reading('P72',(107,111,130,123))])
        self.assertEqual(role_options(words,c,term,[term],'number'),[])

    def test_component_model_is_not_name(self):
        c=Component('U1','box',(100,100,200,200),name='XC4003');term=dict(side='left',base=(100,120))
        words=group_words([reading('XC4003',(107,111,150,123))])
        self.assertEqual(role_options(words,c,term,[term],'name'),[])

    def test_decoder_no_image_io_or_ocr(self):
        import pin_word_decoder_l3 as module
        source=inspect.getsource(module)
        self.assertNotIn('read_image(',source);self.assertNotIn('read_text(',source)
        self.assertNotIn('RapidOCR',source);self.assertNotIn('_target.json',source)

    def test_b_and_c_preserve_geometry_and_nonbox(self):
        term=dict(side='left',tip=[90.,110.],base=[100.,110.],method='frozen')
        pin=Pin('1','GND',term['tip'],base=term['base'],side='left')
        box=Component('U1','box',(100,100,200,200),pins=[pin])
        resistor=Component('R1','r',(200,100,230,110),pins=[copy.deepcopy(pin)])
        raw=[{'component':c.key,'terminals':[copy.deepcopy(term)]} for c in (box,resistor)]
        scene=Scene(300,300,[box,resistor],diagnostics={'pin_events':[
            {**term,'component':c.key,'number':'1','pinname':'GND','exportable':True} for c in (box,resistor)]})
        pool={'U1':{'words':[]}}
        before=copy.deepcopy(scene)
        b,_=stage_b(scene,raw,pool,[]);c,_=stage_c(scene,raw,pool)
        self.assertEqual(scene,before)
        for new in (b,c):
            self.assertEqual(new.components[0].pins[0].tip,pin.tip)
            self.assertEqual(new.components[0].pins[0].base,pin.base)
            self.assertEqual(new.components[1],resistor)
        self.assertEqual(c.components[0].pins,box.pins)

    def test_word_crop_coordinates_and_cache(self):
        class Fake:
            def __call__(self,image):return [],0.
            def text_rec(self,crops):return [('GND',.99) for _ in crops],0.
        image=np.full((300,300,3),255,np.uint8)
        c=Component('U1','box',(100,100,200,200));terms=[dict(side='left',base=(100,110))]
        with tempfile.TemporaryDirectory() as folder:
            reader=WholeWordOCR(Path(folder));reader.backend.engine=Fake();reader.backend.recognizer=reader.backend.engine.text_rec
            word=Text('GND',(105,102,135,118),.99)
            result=reader.read_owner(image,c,terms,[word])
            self.assertEqual(result['words'][0]['bbox'],list(word.bbox))
            self.assertEqual(canonical_tokens(result['words'])[0].text,'GND')
            reader.read_owner(image,c,terms,[word]);self.assertEqual(reader.cache_hits,1)


if __name__=='__main__':unittest.main()
