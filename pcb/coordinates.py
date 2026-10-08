"""Single boundary between OpenCV (top-left) and target (bottom-left) coordinates."""
from __future__ import annotations
import math

def _finite(*values: float) -> None:
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError(f"Non-finite coordinate: {values}")

def opencv_to_target(x: float, y: float, image_height: float) -> tuple[float, float]:
    _finite(x, y, image_height)
    return float(x), float(image_height) - float(y)

def target_to_opencv(x: float, y: float, image_height: float) -> tuple[float, float]:
    _finite(x, y, image_height)
    return float(x), float(image_height) - float(y)

def opencv_bbox_to_target(bbox, image_height: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox); _finite(x1, y1, x2, y2, image_height)
    return x1, float(image_height) - y2, x2, float(image_height) - y1

def target_bbox_to_opencv(bbox, image_height: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox); _finite(x1, y1, x2, y2, image_height)
    return x1, float(image_height) - y2, x2, float(image_height) - y1

def assert_point_in_image(point, width: float, height: float, *, tolerance: float = 1.0) -> None:
    x, y = map(float, point); _finite(x, y, width, height)
    if not (-tolerance <= x <= width + tolerance and -tolerance <= y <= height + tolerance):
        raise ValueError(f"Point outside image {width}x{height}: {(x, y)}")

def assert_bbox_in_image(bbox, width: float, height: float, *, tolerance: float = 1.0) -> None:
    x1, y1, x2, y2 = map(float, bbox); _finite(x1, y1, x2, y2, width, height)
    if not (x1 < x2 and y1 < y2): raise ValueError(f"Invalid bbox ordering: {bbox}")
    assert_point_in_image((x1, y1), width, height, tolerance=tolerance)
    assert_point_in_image((x2, y2), width, height, tolerance=tolerance)

