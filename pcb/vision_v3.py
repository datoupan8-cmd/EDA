"""V3 image-only frontend: symbol detection, text association, then pins."""
from __future__ import annotations
from .schema import Scene, Pin   #项目内部数据结构
from .component_detection import detect_components_v3   #元件检测
from .pin_detection import terminal_candidates_v3, TWO_TERMINAL   #引脚位置候选点
from .pin_semantics import assign_pin_semantics       #匹配引脚和名称
from .official_types import to_official_type     #统一输出格式


def _legacy_pins(component):
    """Variant A isolation: old bbox-edge placeholders with empty names."""
    x1, y1, x2, y2 = component.body_bbox or component.bbox   #bbox：bounding box:外接矩形框。 整体代表：“如果有更明确的 body_bbox，优先用它；否则使用普通 bbox。”
    if component.type in TWO_TERMINAL:  #两端口器件
        if y2 - y1 >= x2 - x1:   #如果高度大于宽度，即元件竖着摆放。
            rows = [((x1 + x2) / 2, y2, "bottom"), ((x1 + x2) / 2, y1, "top")]    #上下两条边的中间各有一个引脚。
        else:
            rows = [(x1, (y1 + y2) / 2, "left"), (x2, (y1 + y2) / 2, "right")]    #左右两边的中点各有一个引脚。
        return [Pin(str(i), "", (x, y), side=side, inferred_number=True, number_confidence=.35, exportable=True, number_source="legacy_bbox_edge") for i, (x, y, side) in enumerate(rows, 1)]
        #return：返回一个列表，列表内容由下面几组数据构成。 创造一个“Pin”，也即引脚。
    return []


def detect_scene_v3(image, ocr, image_name="", pin_v3=True, keypoint_provider=None):
    height, width = image.shape[:2]
    texts = ocr.recognize(image)
    components, roles, diagnostics, debug = detect_components_v3(image, texts, image_name)
    # Retain the established contour detector only as a missing-key fallback.
    # V3 geometry wins every duplicate, so OCR-near-contour no longer decides
    # the main body box for capacitors, boxes, or detected grounds.
    from .vision import detect_scene
    legacy_scene = detect_scene(image, ocr)
    existing = {c.key.upper() for c in components}
    fallback_events = []
    for candidate in legacy_scene.components:
        if candidate.key.upper() in existing or candidate.confidence < .10:
            continue
        candidate.type = to_official_type(candidate.type)
        if candidate.type == "other":
            continue
        candidate.source_id = "legacy_missing_key_fallback"
        candidate.pins = []
        candidate.observable_designator = candidate.observable_designator or candidate.key
        components.append(candidate); existing.add(candidate.key.upper())
        fallback_events.append({"component": candidate.key, "type": candidate.type, "bbox": candidate.bbox, "confidence": candidate.confidence})
    diagnostics["legacy_missing_key_fallback"] = fallback_events
    diagnostics["legacy_fallback_count"] = len(fallback_events)
    pin_events = []
    for component in components:
        if not pin_v3:
            component.pins = _legacy_pins(component)
            continue
        terminals = terminal_candidates_v3(image, component, texts, keypoint_provider)
        for terminal in terminals:
            terminal["tip"] = (min(width - 1.0, max(0.0, terminal["tip"][0])), min(height - 1.0, max(0.0, terminal["tip"][1])))
            terminal["base"] = (min(width - 1.0, max(0.0, terminal["base"][0])), min(height - 1.0, max(0.0, terminal["base"][1])))
        pins, events = assign_pin_semantics(component, terminals, roles)
        component.pins = pins
        pin_events.extend({"component": component.key, **event} for event in events)
    diagnostics.update({
        "mode": "image-only", "pipeline": "v3", "pin_v3_enabled": pin_v3,
        "terminal_candidates": sum(len(c.pins) for c in components),
        "exportable_pins": sum(p.exportable for c in components for p in c.pins),
        "pin_events": pin_events, "keypoint_provider": getattr(keypoint_provider, "name", "disabled"),
        "paper_mapping": {"Netlistify": "task decoupling", "Topology-Consistent": "point-first terminal candidates", "HAWP": "interface only; disabled"},
    })
    scene = Scene(width, height, components, texts, diagnostics=diagnostics)
    return scene, debug
