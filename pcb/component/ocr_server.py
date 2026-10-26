"""Larger PP-OCRv5 ONNX models, with the existing probability-based decoder."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

from .alternative_ocr import MODEL_HASHES, ROOT
from .ocr_ablation import PairedPPOcrV5Backend
from ..core.interfaces import TextStageOutput

SERVER_HASHES = {
    "ch_PP-OCRv5_det_server.onnx": "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad",
    "ch_PP-OCRv5_rec_server.onnx": "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
}
_DLL_HANDLES = []


def prepare_windows_ort_cuda():
    """Use already present CUDA12/cuDNN DLLs; never install or download here."""
    if sys.platform != "win32" or _DLL_HANDLES: return
    import os
    import torch
    import onnxruntime as ort
    torch_lib = Path(torch.__file__).parent / "lib"
    paths = list((ROOT / ".runtime/ort_gpu_debug/nvidia").glob("*/bin"))
    if torch_lib.is_dir(): paths.append(torch_lib)
    _DLL_HANDLES.extend(os.add_dll_directory(str(path.resolve())) for path in paths)
    ort.preload_dlls(cuda=True, cudnn=False, directory="")
    ort.preload_dlls(cuda=False, cudnn=True, directory=str(torch_lib))


class ServerPPOcrV5Backend(PairedPPOcrV5Backend):
    def __init__(self, cache_dir, *, profile="server_rec", use_cuda=False, decoding="greedy"):
        if profile not in {"server_rec", "server_full"}:
            raise ValueError("Unknown server OCR profile")
        if decoding not in {"greedy", "beam"}: raise ValueError("Unknown OCR decoder")
        # New server detector follows its official ImageNet normalization;
        # recognition-only keeps the already validated mobile detector recipe.
        super().__init__(cache_dir, normalization="v5" if profile == "server_full" else "legacy", use_cuda=use_cuda)
        self.profile = profile
        self.decoding = decoding
        self.server_models = ROOT / "models/component/ocr_v5_server"
        required = ["ch_PP-OCRv5_rec_server.onnx"]
        if profile == "server_full": required.append("ch_PP-OCRv5_det_server.onnx")
        for name in required:
            if hashlib.sha256((self.server_models / name).read_bytes()).hexdigest() != SERVER_HASHES[name]:
                raise ValueError("Server OCR checksum mismatch: " + name)
        hashes = dict(MODEL_HASHES)
        del hashes["ch_PP-OCRv5_rec_mobile.onnx"]
        if profile == "server_full": del hashes["ch_PP-OCRv5_det_mobile.onnx"]
        hashes.update({name: SERVER_HASHES[name] for name in required})
        signature = json.loads(self.signature)
        signature.update(models=hashes, profile=profile, decoding=decoding,
                         server_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.signature = json.dumps(signature, sort_keys=True).encode()
        self.model_hashes = hashes

    def engine_parameters(self):
        from rapidocr.utils.typings import ModelType
        params = super().engine_parameters()
        # Large OCR backbones otherwise reserve almost all 6GB on this GPU.
        params["EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy"] = "kSameAsRequested"
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "HEURISTIC"
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_use_max_workspace"] = "0"
        params["Rec.model_path"] = str(self.server_models / "ch_PP-OCRv5_rec_server.onnx")
        params["Rec.model_type"] = ModelType.SERVER
        if self.profile == "server_full":
            params["Det.model_path"] = str(self.server_models / "ch_PP-OCRv5_det_server.onnx")
            params["Det.model_type"] = ModelType.SERVER
            params["Rec.rec_batch_num"] = 1
        return params

    def _ensure_engine(self):
        if self.use_cuda: prepare_windows_ort_cuda()
        engine = super()._ensure_engine()
        if self.decoding == "greedy":
            engine.text_rec.postprocess_op = self.decoder.greedy
        return engine

    def recognize_pair(self, image):
        output = super().recognize_pair(image)
        self.last_diagnostics.update(model_profile=self.profile, models_sha256=self.model_hashes, decoding=self.decoding)
        return output


class TextStagePPOcrV5Server:
    profile = "server_rec"

    def __init__(self): self.backend = None

    def run(self, image, context):
        if self.backend is None:
            resources = context.resources.get("component", {})
            cache = Path(resources.get("cache_dir", ROOT / "runs/component_alternative_cache")) / "ocr_server"
            import torch
            self.backend = ServerPPOcrV5Backend(cache, profile=self.profile, use_cuda=torch.cuda.is_available())
        texts = self.backend.recognize_pair(image)["beam"]
        return TextStageOutput(texts, dict(self.backend.last_diagnostics))


class TextStagePPOcrV5ServerFull(TextStagePPOcrV5Server):
    profile = "server_full"
