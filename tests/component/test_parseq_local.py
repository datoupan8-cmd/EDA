"""Local recognition controls, using synthetic pixels and no dataset/model."""
from dataclasses import asdict
import copy
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from pcb.component.parseq_local import (
    ParseqConfig, accepted_readings, clipped_box, refine_component, select_crops, word_boxes,
)
from pcb.component.proposal_fusion import ComponentProposal
from pcb.component.value_groups_v2 import ComponentStageOrderedValues
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import TextStageOutput
from pcb.schema import Component, Text


class ParseqLocalTests(unittest.TestCase):
    def scene(self, score=.70, kind="r"):
        return SimpleNamespace(texts=[Text("RI",(40,20,60,32),score)],diagnostics={},
            components=[Component("unknown",kind,(40,40,60,60),confidence=.95)])

    def reading(self, **changes):
        row = {"bbox":(38,18,62,34),"source":"ocr_box","token_index":0,
               "text":"R1","confidence":.99,"minimum_probability":.97,"has_eos":True}
        return {**row,**changes}

    def test_uncertain_ascii_designator_can_be_reread_without_input_mutation(self):
        original=self.scene();before=copy.deepcopy(original)
        texts,events=accepted_readings(original,[self.reading()])
        self.assertEqual(texts[0].text,"R1");self.assertEqual(len(events),1)
        self.assertEqual(asdict(original.texts[0]),asdict(before.texts[0]))

    def test_trusted_original_is_preserved_despite_confident_alternative(self):
        original=self.scene(score=.98)
        texts,events=accepted_readings(original,[self.reading()])
        self.assertEqual(texts,original.texts);self.assertFalse(events)

    def test_low_character_score_or_missing_eos_is_rejected(self):
        original=self.scene()
        for changes in ({"minimum_probability":.59},{"has_eos":False},{"confidence":.89}):
            with self.subTest(changes=changes):
                texts,events=accepted_readings(original,[self.reading(**changes)])
                self.assertEqual(texts,original.texts);self.assertFalse(events)

    def test_type_incompatible_designator_and_engineering_value_are_rejected(self):
        original=self.scene(kind="c")
        for label in ("R1","10µF","100Ω","100","random"):
            with self.subTest(label=label):
                texts,events=accepted_readings(original,[self.reading(text=label)])
                self.assertEqual(texts,original.texts);self.assertFalse(events)

    def test_new_word_is_exclusive_to_recovery_and_cannot_override_trusted_region(self):
        original=self.scene();row=self.reading(source="cc_word",token_index=None)
        self.assertFalse(accepted_readings(original,[row],recover=False)[1])
        self.assertEqual(len(accepted_readings(original,[row],recover=True)[1]),1)
        original.texts[0].score=.99
        self.assertFalse(accepted_readings(original,[row],recover=True)[1])

    def test_model_recognition_requires_chip_and_keeps_literal_text(self):
        row=self.reading(text="STM32F103C8T6")
        self.assertFalse(accepted_readings(self.scene(),[row])[1])
        texts,events=accepted_readings(self.scene(kind="box"),[row])
        self.assertEqual(texts[0].text,row["text"]);self.assertEqual(events[0]["accepted_as"],"model")

    def test_ascii_quantities_near_chip_are_not_model_readings(self):
        original=self.scene(kind="box")
        for value in ("100k","100nF","100mV","40.2k","160R","0.98F"):
            with self.subTest(value=value):
                texts,events=accepted_readings(original,[self.reading(text=value)])
                self.assertEqual(texts,original.texts);self.assertFalse(events)

    def test_crop_bounds_and_work_quota_are_enforced(self):
        image=np.full((100,100,3),255,np.uint8)
        self.assertEqual(clipped_box((-2,-3,105,110),image.shape),(0,0,100,100))
        original=self.scene()
        original.texts=[Text("R1",(20,10,40,20),.7) for _ in range(20)]
        crops=select_crops(image,original,ParseqConfig(max_ocr_crops=3,max_new_crops=2))
        self.assertLessEqual(len(crops),5)
        self.assertEqual(sum(r["source"]=="ocr_box" for r in crops),3)

    def test_new_words_come_from_pixels_and_blank_regions_produce_none(self):
        image=np.full((90,240,3),255,np.uint8)
        regions=[{"bbox":(0,0,240,90)}]
        self.assertFalse(word_boxes(image,regions,16))
        cv2.putText(image,"R12",(30,45),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1,cv2.LINE_8)
        boxes=word_boxes(image,regions,16)
        self.assertTrue(boxes)
        self.assertTrue(any(b[0]<=35 and b[2]>=60 for b in boxes))

    def test_fixed_proposal_rebuild_keeps_shared_text_contract_and_zero_change_parity(self):
        class Detector:
            def detect(self,image):
                return [ComponentProposal((40,40,60,60),"r",.95,"test")],{}
        image=np.full((100,100,3),255,np.uint8)
        texts=[Text("R1",(40,20,60,32),.99),Text("10k",(40,65,60,77),.99)]
        context=PipelineContext("arbitrary",100,100,PipelineConfig(),None,Detector())
        original=ComponentStageOrderedValues().run(image,TextStageOutput(texts),context)
        result=refine_component(original,list(texts),image,context)
        self.assertEqual([asdict(c) for c in result.components],[asdict(c) for c in original.components])
        self.assertIs(result.texts,original.texts);self.assertIs(result.roles,original.roles)


if __name__=="__main__":unittest.main()
