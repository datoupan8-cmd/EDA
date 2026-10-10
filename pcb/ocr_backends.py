"""Compatibility import; maintain pcb/text/ocr_backends.py instead."""
import sys
from pcb.text import ocr_backends as _implementation
sys.modules[__name__] = _implementation
