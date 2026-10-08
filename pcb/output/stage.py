"""Submission boundary wrapper; export and validation remain unchanged."""
from __future__ import annotations

from ..core.interfaces import SubmissionStageOutput
from ..submission import export, validate_strict


class OfficialSubmissionStage:
    def run(self, scene, context) -> SubmissionStageOutput:
        data = export(scene)
        validate_strict(data, (scene.width, scene.height))
        return SubmissionStageOutput(data, {"strict_contract_valid": True})
