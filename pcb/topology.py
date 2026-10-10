"""Compatibility import; maintain pcb/topology_stage/v1.py instead."""
import sys
from pcb.topology_stage import v1 as _implementation
sys.modules[__name__] = _implementation
