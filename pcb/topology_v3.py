"""Compatibility import; maintain pcb/topology_stage/v3.py instead."""
import sys
from pcb.topology_stage import v3 as _implementation
sys.modules[__name__] = _implementation
