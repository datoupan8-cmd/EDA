"""Original recovery geometry, with unchanged arithmetic."""
from typing import Sequence

def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = (
        max(float(left[0]), float(right[0])),
        max(float(left[1]), float(right[1])),
        min(float(left[2]), float(right[2])),
        min(float(left[3]), float(right[3])),
    )
    intersection_area = bbox_area(intersection)
    union = bbox_area(left) + bbox_area(right) - intersection_area
    return intersection_area / union if union > 0 else 0.0
