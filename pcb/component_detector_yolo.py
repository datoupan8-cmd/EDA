"""Compatibility import; maintain pcb/component/detector_yolo.py instead."""
import sys
from pcb.component import detector_yolo as _implementation
sys.modules[__name__] = _implementation
