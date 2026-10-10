"""Compatibility import for component ownership."""
import sys
from .component.latest import text_assignment as _implementation
sys.modules[__name__] = _implementation
