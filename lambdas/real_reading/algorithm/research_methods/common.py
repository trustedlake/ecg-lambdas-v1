"""Shared measured tracing for candidates, with explicit uncertainty."""
import numpy as np
from .contracts import Extraction
from .frozen_peaks34.ecg import signal_tinted, trace_ownership


def trace_evidence(context, raw, contrast):
    raw = np.asarray(raw, dtype=np.uint8)
    leads = signal_tinted.locate_leads(raw, context.config, context.geometry)
    traces = [signal_tinted.trace_lead(raw, contrast, lead) for lead in leads]
    ownership = trace_ownership.separate_traces(traces, leads, context.image.shape)
    return Extraction(leads, traces, {'ownership': ownership})
