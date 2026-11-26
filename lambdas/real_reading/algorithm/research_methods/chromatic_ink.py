"""Neutral dark ink on a chromatically distinct grid; historical technique."""
from .frozen_peaks34.ecg.signal_neutral import ink_evidence
from .common import trace_evidence

VERSION = 'neutral-color-v1'


def extract(context):
    if 'reference_box_px' in context.geometry:
        from ecg.signal_neutral import reference_ink_evidence
        raw, contrast = reference_ink_evidence(context.image)
    else:
        raw, contrast = ink_evidence(context.image)
    return trace_evidence(context, raw, contrast)
