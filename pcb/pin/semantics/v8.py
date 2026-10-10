"""P6.4 targeted multi-rotation OCR; accepted V7 pins remain frozen."""
from __future__ import annotations

from pathlib import Path
import time

from ...core.interfaces import PinSemanticsOutput
from ..local_ocr import RotatedLocalPinOCR, refine_pin_semantics_local


class PinSemanticsStageV8:
    def __init__(self, reader=None):
        self.reader = reader

    def run(self, component, localization, context):
        image = localization.debug.get("pin_local_ocr_image")
        if image is None:
            raise ValueError("Pin semantics v8 requires localization v8's image handoff")
        if self.reader is None:
            rapid = getattr(context.ocr, "rapid", context.ocr)
            cache = getattr(rapid, "cache", None)
            self.reader = RotatedLocalPinOCR(Path(cache) / "pin_local_rotated" if cache else None)
        start = time.perf_counter()
        calls_before, hits_before = self.reader.calls, self.reader.cache_hits
        events, traces = [], []
        for item, terminals in localization.terminals:
            pins, rows, stats = refine_pin_semantics_local(
                image, item, terminals, component.roles, self.reader)
            item.pins = pins
            events.extend({"component": item.key, **row} for row in rows)
            if stats["attempted_terminals"]:
                traces.append({"component": item.key, **stats})
        statistics = {key: sum(row[key] for row in traces) for key in (
            "roi_count", "local_token_count", "ambiguous_token_count", "attempted_terminals", "recovered_pins")}
        statistics.update({"seconds": time.perf_counter()-start,
                           "engine_calls": self.reader.calls-calls_before,
                           "cache_hits": self.reader.cache_hits-hits_before})
        diagnostics = {
            "mode": "image-only", "pipeline": context.config.legacy_pipeline_name,
            "pin_semantics": "v8_p6_4_targeted_rotated_local_ocr_experiment",
            "ocr_backend": getattr(context.ocr, "name", "rapidocr"),
            "terminal_candidates": sum(len(item.pins) for item in component.components),
            "exportable_pins": sum(pin.exportable for item in component.components for pin in item.pins),
            "pin_events": events, "pin_local_ocr": statistics,
            "pin_local_ocr_traces": traces,
        }
        return PinSemanticsOutput(component.components, diagnostics)
