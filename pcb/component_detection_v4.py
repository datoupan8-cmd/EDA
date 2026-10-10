"""Compatibility import; edit pcb/component/latest instead."""
import sys
from .component.latest import detection as _implementation
sys.modules[__name__] = _implementation
