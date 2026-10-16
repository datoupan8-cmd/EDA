"""PP-OCRv5 input-normalization and probability-based CTC decoding ablations.

No model weights, ground truth, image names, or case-specific vocabulary are used.
Each detector/recognizer pass produces BOTH greedy and beam outputs, so the
decoder comparison uses exactly the same pixels, boxes and network outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from .alternative_ocr import MODEL_HASHES, PPOcrV5Backend, ROOT
from .detector_yolo import tile_origins
from ..core.interfaces import TextStageOutput
from ..schema import Text
from ..text.ocr_backends import deduplicate_texts

V5_MEAN = (0.485, 0.456, 0.406)
V5_STD = (0.229, 0.224, 0.225)
NEG_INF = float("-inf")


@dataclass(frozen=True)
class BeamConfig:
    width: int = 8
    top_k: int = 6
    max_length: int = 24
    confidence_gate: float = .95
    # At most 1.5x prior odds for a valid, anchored electronics token. Visual
    # sequence probability remains the main criterion, including for models.
    role_bonus: float = math.log(1.5)
    max_visual_loss: float = .7


def _add(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    maximum, minimum = (a, b) if a >= b else (b, a)
    return maximum + math.log1p(math.exp(minimum - maximum))


def prefix_beam_search(probabilities, config=BeamConfig()):
    """CTC prefix beam search: accumulate blank/nonblank alignment mass.

    Blank is channel zero. Top-k pruning always includes blank; repeated
    characters can only create a new repeated label through a blank alignment.
    Scores are log probabilities, not per-character string substitutions.
    """
    probs = np.asarray(probabilities)
    beam = {(): (0., NEG_INF)}
    k = min(config.top_k, probs.shape[1])
    top = np.argpartition(probs, -k, axis=1)[:, -k:]
    for row, ids in zip(probs, top):
        candidates = sorted(set(ids.tolist()) | {0})
        logp = {i: math.log(max(float(row[i]), 1e-30)) for i in candidates}
        nxt = {}

        def update(prefix, blank=NEG_INF, nonblank=NEG_INF):
            old_b, old_n = nxt.get(prefix, (NEG_INF, NEG_INF))
            nxt[prefix] = (_add(old_b, blank), _add(old_n, nonblank))

        for prefix, (pb, pn) in beam.items():
            total = _add(pb, pn)
            update(prefix, blank=total + logp[0])
            for char in candidates:
                if char == 0:
                    continue
                if prefix and char == prefix[-1]:
                    update(prefix, nonblank=pn + logp[char])
                    if len(prefix) < config.max_length:
                        update(prefix + (char,), nonblank=pb + logp[char])
                elif len(prefix) < config.max_length:
                    update(prefix + (char,), nonblank=total + logp[char])
        beam = dict(sorted(nxt.items(), key=lambda item: _add(*item[1]), reverse=True)[:config.width])
    return [(prefix, _add(*score)) for prefix, score in beam.items()]


def sequence_probability(probabilities, labels):
    """Exact CTC forward probability and best-alignment character confidence."""
    if not labels:
        return NEG_INF, 0.
    probs = np.asarray(probabilities)
    states = [0]
    for label in labels:
        states.extend((label, 0))
    count = len(states)
    states = np.asarray(states, dtype=int)
    skip_allowed = np.zeros(count, dtype=bool)
    skip_allowed[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    forward = np.full(count, NEG_INF)
    best = np.full(count, NEG_INF)
    forward[0] = best[0] = 0.
    back = []
    for row in probs:
        previous = np.stack((forward, np.r_[NEG_INF, forward[:-1]], np.r_[NEG_INF, NEG_INF, forward[:-2]]))
        previous[2, ~skip_allowed] = NEG_INF
        best_previous = np.stack((best, np.r_[NEG_INF, best[:-1]], np.r_[NEG_INF, NEG_INF, best[:-2]]))
        best_previous[2, ~skip_allowed] = NEG_INF
        emission = np.log(np.maximum(row[states], 1e-30))
        new = np.logaddexp.reduce(previous, axis=0) + emission
        pointers = np.arange(count) - best_previous.argmax(axis=0)
        new_best = best_previous.max(axis=0) + emission
        forward, best = new, new_best
        back.append(pointers)
    score = _add(forward[-1], forward[-2])
    state = count - 1 if best[-1] >= best[-2] else count - 2
    char_scores = np.zeros(len(labels))
    for t in range(len(probs) - 1, -1, -1):
        if state % 2:
            char_scores[state // 2] = max(char_scores[state // 2], probs[t, states[state]])
        state = back[t][state]
    return score, float(char_scores.mean())


def role_anchor(text):
    # This is a generic format prior, never a dictionary of observed case text.
    from .latest.text_roles import PREFIX_TYPE
    for prefix in sorted(PREFIX_TYPE, key=len, reverse=True):
        if text.startswith(prefix) and re.fullmatch(r"[0-9OIil]+[A-Za-z]?", text[len(prefix):]):
            return "designator", prefix
    if re.match(r"^[<>+\-]?[0-9.OIil]", text) and any(c.isdigit() for c in text):
        return "value", ""
    return "generic", ""


def valid_role(text, anchor):
    kind, prefix = anchor
    if kind == "designator":
        return bool(re.fullmatch(re.escape(prefix) + r"\d+[A-Za-z]?", text))
    if kind == "value":
        from .latest.text_roles import VALUE_RE
        return bool(VALUE_RE.fullmatch(text))
    return False


class PairedCTCDecoder:
    """Keep RapidOCR's greedy API; expose paired results by exact text/score.

    RapidOCR sorts and restores crop order, then filters low-scoring strings.
    Exact greedy result keys keep alternatives aligned across those operations.
    If identical keys have differing alternatives, preserve greedy (fail safe).
    """
    def __init__(self, greedy_decoder, config=BeamConfig()):
        self.greedy = greedy_decoder
        self.character = greedy_decoder.character
        self.config = config
        self.reset()

    def reset(self):
        self.readings = {}
        self.ambiguous_keys = set()
        self.events = []
        self.considered = 0

    def choose(self, probabilities, text, confidence):
        if (confidence >= self.config.confidence_gate or not 1 < len(text) <= self.config.max_length
                or not re.fullmatch(r"[\x20-\x7eµμΩ]+", text)):
            return text, float(confidence), None
        self.considered += 1
        greedy_ids = probabilities.argmax(axis=1)
        labels = tuple(int(c) for i, c in enumerate(greedy_ids) if c and (i == 0 or c != greedy_ids[i-1]))
        greedy_score, _ = sequence_probability(probabilities, labels)
        anchor = role_anchor(text)
        options = [(text, greedy_score, float(confidence))]
        seen = {text}
        for ids, _ in prefix_beam_search(probabilities, self.config):
            candidate = "".join(self.character[i] for i in ids)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            # Never turn generic annotation text into a fabricated device ID.
            if anchor[0] != "generic" and not valid_role(candidate, anchor):
                continue
            score, char_confidence = sequence_probability(probabilities, ids)
            if score >= greedy_score - self.config.max_visual_loss:
                options.append((candidate, score, min(float(confidence), char_confidence)))
        ranked = sorted(options, key=lambda x: x[1] + self.config.role_bonus * valid_role(x[0], anchor), reverse=True)
        chosen, score, new_confidence = ranked[0]
        if chosen == text or new_confidence < .5:
            return text, float(confidence), None
        return chosen, new_confidence, {
            "greedy": text, "chosen": chosen, "greedy_confidence": float(confidence),
            "chosen_confidence": new_confidence, "log_probability_delta": score - greedy_score,
            "role": anchor[0], "candidates": [{"text": t, "log_probability": s} for t, s, _ in options]}

    def __call__(self, predictions, return_word_box=False, **kwargs):
        if return_word_box:
            raise ValueError("Paired Component decoding requires word boxes disabled")
        lines, words = self.greedy(predictions, return_word_box, **kwargs)
        for probabilities, (text, confidence) in zip(predictions, lines):
            chosen, score, event = self.choose(probabilities, text, confidence)
            key = (text, float(confidence))
            reading = (chosen, score)
            if key in self.ambiguous_keys or (key in self.readings and self.readings[key] != reading):
                self.ambiguous_keys.add(key)
                self.readings[key] = (text, float(confidence))
            else:
                self.readings[key] = reading
            if event:
                self.events.append(event)
        return lines, words


class PairedPPOcrV5Backend(PPOcrV5Backend):
    def __init__(self, cache_dir, *, normalization="legacy", use_cuda=False):
        if normalization not in {"legacy", "v5"}:
            raise ValueError("normalization must be legacy or v5")
        super().__init__(cache_dir)
        self.normalization = normalization
        self.use_cuda = use_cuda
        self.decoder = None
        import sys
        local_runtime = ROOT / ".runtime/alternative_frontend"
        if local_runtime.is_dir() and str(local_runtime) not in sys.path:
            sys.path.insert(0, str(local_runtime))
        import rapidocr
        import onnxruntime
        runtime_root = Path(rapidocr.__file__).parent
        self.signature = json.dumps({
            "runtime": "rapidocr-3.9.2", "models": MODEL_HASHES, "tiling": [768, 128, 2],
            "det_requested_limit": 2048, "normalization": normalization,
            "mean": V5_MEAN if normalization == "v5" else (.5, .5, .5),
            "std": V5_STD if normalization == "v5" else (.5, .5, .5),
            "beam": asdict(BeamConfig()), "use_cuda": use_cuda,
            "ort_version": onnxruntime.__version__, "numpy_version": np.__version__,
            "det_postprocess": {"thresh": .3, "box_thresh": .5, "unclip_ratio": 1.6, "use_dilation": True},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_sha256": hashlib.sha256((ROOT / "pcb/component/alternative_ocr.py").read_bytes()).hexdigest(),
            "runtime_files": {name: hashlib.sha256((runtime_root / name).read_bytes()).hexdigest()
                              for name in ("config.yaml", "ch_ppocr_det/main.py", "ch_ppocr_det/utils.py",
                                           "ch_ppocr_rec/main.py", "ch_ppocr_rec/utils.py")},
        }, sort_keys=True).encode()

    def engine_parameters(self):
        params = super().engine_parameters()
        params["EngineConfig.onnxruntime.use_cuda"] = self.use_cuda
        return params

    def _ensure_engine(self):
        engine = super()._ensure_engine()
        if self.decoder is None:
            if self.normalization == "v5":
                engine.text_det.mean = V5_MEAN
                engine.text_det.std = V5_STD
            self.decoder = PairedCTCDecoder(engine.text_rec.postprocess_op)
            engine.text_rec.postprocess_op = self.decoder
        if self.use_cuda:
            for module in (engine.text_det, engine.text_rec):
                if "CUDAExecutionProvider" not in module.session.session.get_providers():
                    raise RuntimeError("Requested OCR CUDA but runtime fell back to CPU")
        return engine

    def _read_pair(self, image, *, left=0, top=0, scale=1.):
        engine = self._ensure_engine()
        self.decoder.reset()
        output = engine(image, use_det=True, use_cls=False, use_rec=True)
        greedy, beam = [], []
        if output.txts is not None and output.boxes is not None:
            for polygon, value, confidence in zip(output.boxes, output.txts, output.scores):
                xs, ys = polygon[:, 0], polygon[:, 1]
                box = (float(xs.min()) / scale + left, float(ys.min()) / scale + top,
                       float(xs.max()) / scale + left, float(ys.max()) / scale + top)
                alternate, score = self.decoder.readings.get((str(value), float(confidence)), (str(value), float(confidence)))
                greedy.append(Text(str(value), box, float(confidence), "ppocrv5"))
                beam.append(Text(alternate, box, score, "ppocrv5"))
        return greedy, beam, {"considered": self.decoder.considered, "changes": list(self.decoder.events)}

    def recognize_pair(self, image):
        key = hashlib.sha256(image.tobytes() + self.signature).hexdigest()[:24]
        path = self.cache / (key + ".paired.json")
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.last_diagnostics = {**saved["diagnostics"], "cache_hit": True}
            return {v: [Text(**row) for row in saved[v]] for v in ("greedy", "beam")}
        greedy, beam, diagnostics = self._read_pair(image)
        calls = [diagnostics]
        height, width = image.shape[:2]
        for top in tile_origins(height, 768, 128):
            for left in tile_origins(width, 768, 128):
                patch = cv2.resize(image[top:top+768, left:left+768], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                a, b, detail = self._read_pair(patch, left=left, top=top, scale=2.)
                greedy.extend(a)
                beam.extend(b)
                calls.append(detail)
        results = {"greedy": deduplicate_texts(greedy, .3), "beam": deduplicate_texts(beam, .3)}
        self.last_diagnostics = {
            "backend": "ppocrv5_paired_ctc", "normalization": self.normalization, "cache_hit": False,
            "ocr_passes": len(calls), "beam_considered": sum(c["considered"] for c in calls),
            "beam_changes": [event for c in calls for event in c["changes"]],
            "effective_det_mean": list(self.engine.text_det.mean), "effective_det_std": list(self.engine.text_det.std),
            "providers": self.engine.text_rec.session.session.get_providers(),
            "tokens": {v: len(tokens) for v, tokens in results.items()}, "GT_used_in_inference": False}
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {v: [asdict(token) for token in tokens] for v, tokens in results.items()}
        payload["diagnostics"] = self.last_diagnostics
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return results


class GreedyProfilePPOcrV5Backend(PairedPPOcrV5Backend):
    """A normalization-only profile must not pay the beam-decoding cost."""
    def _ensure_engine(self):
        engine = PPOcrV5Backend._ensure_engine(self)
        if self.normalization == "v5":
            engine.text_det.mean = V5_MEAN
            engine.text_det.std = V5_STD
        if self.use_cuda:
            for module in (engine.text_det, engine.text_rec):
                if "CUDAExecutionProvider" not in module.session.session.get_providers():
                    raise RuntimeError("Requested OCR CUDA but runtime fell back to CPU")
        return engine


class TextStagePPOcrV5Ablation:
    normalization = "v5"
    decoding = "greedy"

    def __init__(self):
        self.backend = None

    def run(self, image, context):
        resources = context.resources.get("component", {})
        if self.backend is None:
            backend_class = GreedyProfilePPOcrV5Backend if self.decoding == "greedy" else PairedPPOcrV5Backend
            self.backend = backend_class(Path(resources.get("cache_dir", ROOT / "runs/component_alternative_cache")) / "ocr_paired",
                                         normalization=self.normalization)
        if self.decoding == "greedy":
            texts = self.backend.recognize(image)
            return TextStageOutput(texts, {**self.backend.last_diagnostics, "normalization": self.normalization,
                                          "decoding": "greedy"})
        results = self.backend.recognize_pair(image)
        return TextStageOutput(results[self.decoding], {**self.backend.last_diagnostics, "decoding": self.decoding})


class TextStagePPOcrV5Beam(TextStagePPOcrV5Ablation):
    normalization = "legacy"
    decoding = "beam"


class TextStagePPOcrV5NormBeam(TextStagePPOcrV5Ablation):
    decoding = "beam"
