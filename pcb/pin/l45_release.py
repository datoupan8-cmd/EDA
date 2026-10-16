"""Portable selection of the accepted L45 Pin stage; algorithms stay frozen."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA256 = "d8b1babd809f622ba4243becf135ae38e92effdc6fe9dd079526cf67e084fc71"


class L45RegressionSemantics:
    """Use the released checkpoint and unchanged known-two-point estimator."""
    def __init__(self):
        from .rc_pair_l45 import RCPointSemantics

        resources = ROOT / "models/pin/l45_rc_pair"
        with (resources / "best.pt").open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != MODEL_SHA256:
            raise ValueError("L45 checkpoint missing/changed; run git lfs pull")
        self.implementation = RCPointSemantics("regression", resources)

    def run(self, component, localization, context):
        raise TypeError("L45 requires an explicit image; use ModularPipeline")

    def run_image(self, image, component, localization, context):
        # The public runner already supplies the owner's words-cache path.
        # Keep model OCR in the same selected cache root without changing the
        # public runner or mutating another stage's shared Context.
        config = context.resources.get("pin", {})
        if "model_cache" not in config and "words_cache" in config:
            local_context = copy.copy(context)
            local_context.resources = dict(context.resources)
            local_context.resources["pin"] = {
                **config,
                "model_cache": Path(config["words_cache"]).parent / "pin_model_control_l42",
            }
        else:
            local_context = context
        return self.implementation.run_image(image, component, localization, local_context)
