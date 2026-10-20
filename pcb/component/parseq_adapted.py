"""PARSeq with a verified, development-only fine-tuned checkpoint."""
import hashlib,json
from pathlib import Path

from .parseq_local import ParseqReader,ParseqConfig


class AdaptedParseqReader(ParseqReader):
    def __init__(self,cache_dir,weights,training_manifest,device="cuda",config=ParseqConfig()):
        super().__init__(cache_dir,device=device,config=config)
        weights=Path(weights);metadata_path=Path(training_manifest)
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        ids=[f"{i:04d}" for i in range(1,151)]
        if metadata["eligible_train120"]!=[c for c in ids if int(c)%5] or metadata["internal_check30"]!=[c for c in ids if not int(c)%5]:
            raise PermissionError("Fine-tuned checkpoint split is not allowed")
        if any(c not in metadata["eligible_train120"] for c in metadata["ACTUAL_TRAIN_CASE_IDS"]):
            raise PermissionError("Fine-tuned checkpoint used non-training cases")
        digest=hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest!=metadata["best_weights_sha256"]:raise ValueError("Fine-tuned PARSeq checksum mismatch")
        signature=json.loads(self.signature)
        signature.update({"weights":digest,"fine_tuning_metadata_sha256":hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "fine_tuning_source_weights":metadata["source_weights_sha256"],"best_epoch":metadata["best_epoch"]})
        self.weights=weights;self.signature=json.dumps(signature,sort_keys=True).encode()
        self.training_metadata=metadata

    def recognize(self,image,crops):
        readings,diagnostics=super().recognize(image,crops)
        return readings,{**diagnostics,"backend":"PARSeq_schematic_finetuned",
            "weights_sha256":json.loads(self.signature)["weights"]}
