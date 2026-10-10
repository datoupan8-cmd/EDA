"""Compatibility import; maintain pcb/component/geometry.py instead."""
import sys
from pcb.component import geometry as _implementation
sys.modules[__name__] = _implementation
