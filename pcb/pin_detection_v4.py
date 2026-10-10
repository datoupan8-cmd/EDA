"""Compatibility import for peer Pin detector."""
import sys
from .pin.peer_v4 import detection as _implementation
sys.modules[__name__] = _implementation
