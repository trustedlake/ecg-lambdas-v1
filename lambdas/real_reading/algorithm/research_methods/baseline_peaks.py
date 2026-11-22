"""Exact standard extraction order from approved peaks34, including its gate."""
from .contracts import Extraction
from .frozen_peaks34.ecg import signal, peak_recovery, contrast_gate

VERSION = 'baseline-full-photo-anchors-1'


def extract(context):
    raw, contrast = signal.ink_evidence(context.image)
    if context.geometry.get('extraction_domain') == 'full_photo':
        from .common import trace_full_photo
        return trace_full_photo(context,raw,contrast)
    leads = signal.locate_leads(raw, context.config, context.geometry)
    traces = [signal.trace_lead(raw, contrast, lead) for lead in leads]
    gate = contrast_gate.assess_color_separation(context.image, leads, traces, context.grid)
    diagnostics = {'color_gate': gate}
    diagnostics['peak_recovery'] = peak_recovery.recover(context.image, leads, traces)
    return Extraction(leads, traces, diagnostics)
