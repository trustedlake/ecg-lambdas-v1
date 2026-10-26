"""Experiment only: the v12 extraction core with its activation gate disabled.

Used to measure extraction quality separately from the gate. Not in the registry.
"""
from . import neutral_line_grid

VERSION = neutral_line_grid.VERSION + '-ungated'


def extract(context):
    return neutral_line_grid.extract(context, gated=False)
