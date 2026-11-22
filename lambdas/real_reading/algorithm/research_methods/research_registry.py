"""Methods for evaluation runs. Production reads registry.METHODS only.

METHODS = the six production methods (same modules) + research-only entries:
- local_ink_graph: appearance-gated local ink graph (was in the analyser registry).
- peaks34_reference: historical Peaks34 control (see peaks34_reference.py).
The eight submitted scripts are in submitted_adapter.SUBMITTED_METHODS.
"""
from .registry import METHODS as PRODUCTION_METHODS
from . import local_ink_activation, peaks34_reference

METHODS = dict(PRODUCTION_METHODS)
METHODS['local_ink_graph'] = local_ink_activation
METHODS['peaks34_reference'] = peaks34_reference
