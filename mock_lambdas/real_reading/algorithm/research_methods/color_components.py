"""Colored-grid specialist, isolated from the six previous methods."""
from .component_specialists import component_trace
VERSION = 'color-components-v6'

def extract(context):
    return component_trace(context, monochrome=False)
