"""Compatibility import; maintain pcb/topology_stage/v2.py instead."""
import sys
from pcb.topology_stage import v2 as _implementation
sys.modules[__name__] = _implementation
