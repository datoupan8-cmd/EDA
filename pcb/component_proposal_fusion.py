"""Compatibility import; maintain pcb/component/proposal_fusion.py instead."""
import sys
from pcb.component import proposal_fusion as _implementation
sys.modules[__name__] = _implementation
