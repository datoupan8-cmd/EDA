"""Compatibility import; maintain pcb/wire_stage/v3.py instead."""
import sys
from pcb.wire_stage import v3 as _implementation
sys.modules[__name__] = _implementation
