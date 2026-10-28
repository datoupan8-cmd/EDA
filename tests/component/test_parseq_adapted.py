import hashlib,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pcb.component.parseq_adapted import AdaptedParseqReader
from pcb.component.parseq_local import ParseqReader


class AdaptedReaderTests(unittest.TestCase):
    def fixture(self,root,content=b"trained"):
        weights=root / "best.pt";weights.write_bytes(content)
        metadata={"eligible_train120":[f"{i:04d}" for i in range(1,151) if i%5],
            "internal_check30":[f"{i:04d}" for i in range(1,151) if not i%5],
            "ACTUAL_TRAIN_CASE_IDS":["0001"],"best_weights_sha256":hashlib.sha256(content).hexdigest(),
            "source_weights_sha256":"official_pretrained","best_epoch":6}
        path=root / "complete.json";path.write_text(json.dumps(metadata),encoding="utf-8")
        return weights,path,metadata

    @staticmethod
    def parent(reader,*a,**kw):reader.signature=json.dumps({"weights":"official_pretrained","config":{}}).encode()

    def test_cache_identity_includes_trained_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weights,path,metadata=self.fixture(root)
            with patch.object(ParseqReader,"__init__",self.parent):
                reader=AdaptedParseqReader(root / "cache",weights,path)
            signature=json.loads(reader.signature)
            self.assertEqual(signature["weights"],hashlib.sha256(weights.read_bytes()).hexdigest())
            self.assertEqual(signature["fine_tuning_metadata_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(reader.weights,weights);self.assertEqual(signature["best_epoch"],6)

    def test_modified_candidate_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weights,path,_=self.fixture(root);weights.write_bytes(b"changed")
            with patch.object(ParseqReader,"__init__",self.parent),self.assertRaisesRegex(ValueError,"checksum"):
                AdaptedParseqReader(root / "cache",weights,path)

    def test_checkpoint_using_check_or_reserved_cases_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weights,path,metadata=self.fixture(root)
            for case in ("0005","0151"):
                metadata["ACTUAL_TRAIN_CASE_IDS"]=[case];path.write_text(json.dumps(metadata),encoding="utf-8")
                with self.subTest(case=case),patch.object(ParseqReader,"__init__",self.parent),self.assertRaises(PermissionError):
                    AdaptedParseqReader(root / "cache",weights,path)


if __name__=="__main__":unittest.main()
