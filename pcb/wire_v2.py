"""Compatibility import; maintain pcb/wire_stage/v2.py instead."""
import sys
from pcb.wire_stage import v2 as _implementation
sys.modules[__name__] = _implementation
