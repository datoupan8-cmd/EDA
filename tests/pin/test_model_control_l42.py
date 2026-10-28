import copy
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from pcb.core.interfaces import PinSemanticsOutput
from pcb.pin.model_control_l42 import L42ControlSemantics
from tests.pin.test_model_reader_l41 import fixture,FakeReader,LIBRARY
from pcb.schema import Text


class Baseline:
    def __init__(self,scene):self.scene=scene
    def run_image(self,image,component,localization,context):
        return PinSemanticsOutput(copy.deepcopy(self.scene.components),copy.deepcopy(self.scene.diagnostics))


class ControlStageTests(unittest.TestCase):
    def test_image_stage_retains_old_library_gain_and_input_geometry(self):
        scene,image=fixture([Text('TEST123',(10,10,60,20),.99)])
        original=copy.deepcopy(asdict(scene))
        stage=L42ControlSemantics(baseline=Baseline(scene),reader=FakeReader(),library=LIBRARY,templates={})
        component=SimpleNamespace(texts=scene.texts)
        context=SimpleNamespace(width=scene.width,height=scene.height,resources={})
        result=stage.run_image(image,component,None,context)
        self.assertEqual(result.components[0].pins[0].number,'7')
        self.assertEqual(result.components[0].pins[0].tip,scene.components[0].pins[0].tip)
        self.assertEqual(asdict(scene),original)
        self.assertFalse(result.diagnostics['L42_control']['expanded_catalog_used'])
        self.assertEqual(stage.reader.calls,0)

    def test_unselected_constructor_is_lazy(self):
        stage=L42ControlSemantics()
        self.assertIsNone(stage.baseline)
        self.assertIsNone(stage.reader)
        self.assertIsNone(stage.library)
        self.assertIsNone(stage.templates)


if __name__=='__main__':unittest.main()
