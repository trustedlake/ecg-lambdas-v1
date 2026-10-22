"""Faint-ink experiment: measured grid crossings and uncertain lead ownership."""
from .frozen_peaks34.ecg import signal_tinted
from . import faint_ink_ownership as trace_ownership
from .contracts import Extraction
import cv2
import numpy as np

VERSION = 'slanted-grid-crossings'


def extract(context):
    raw, contrast = signal_tinted.ink_evidence(context.image)
    leads = signal_tinted.locate_leads(raw, context.config, context.geometry)
    extra, _ = signal_tinted.stroke_evidence(context.image)
    colors = context.image.astype(np.float32)
    scale = max(9, round(min(raw.shape) * .016) | 1)
    paper = cv2.morphologyEx(colors, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale)))
    absorption = np.maximum(0, paper - colors).min(axis=2)
    from scipy.ndimage import median_filter
    red=colors[...,0]
    length=max(31,int(np.median([l['row_spacing_px'] for l in leads])*2)|1)
    # Robust directional background includes sustained grid but not short waves.
    v=median_filter(red,size=(length,1),mode='nearest')
    h=median_filter(red,size=(1,length),mode='nearest')
    background=np.minimum(v,h)
    detail=background-red
    neutral = np.ptp(colors,axis=2)<12
    raw=(((detail>18)&(absorption>8)) | ((raw>0)&neutral)).astype(np.uint8)
    contrast=np.maximum(0,detail)
    extra=raw>0
    crossing = (colors.max(axis=2) < 210) & (np.ptp(colors, axis=2) < 90) & (absorption > 12)
    traces = []
    for lead in leads:
        trace = signal_tinted.trace_lead(raw, contrast, lead)
        # Recover only measured narrow strokes touching the selected trace.
        # Resolve ownership AFTER recovery, including new overlapping claims.
        trace = signal_tinted.recover_stroke_pixels(extra, trace, lead)
        x, y = trace['mask_origin']; h, w = trace['mask'].shape
        seed = trace['mask'].astype(np.uint8)
        bracketed = np.zeros_like(seed, dtype=bool)
        for kernel in (np.ones((21, 1), np.uint8), np.ones((1, 7), np.uint8),
                       np.eye(9, dtype=np.uint8), np.fliplr(np.eye(9, dtype=np.uint8))):
            bracketed |= cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel).astype(bool)
        # Steep strokes are rarely axis-aligned in photographs. Bridge only
        # bracketed measured pixels along a small bank of steep orientations.
        for dx in (-4, -2, 2, 4):
            kernel = np.zeros((21, 9), np.uint8)
            cv2.line(kernel, (4 - dx // 2, 0), (4 + dx // 2, 20), 1, 1)
            bracketed |= cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel).astype(bool)
        # Restore a bracketed gap only where all RGB channels absorb light.
        trace['mask'] |= bracketed & crossing[y:y+h, x:x+w]
        traces.append(trace)
    ownership = trace_ownership.separate_traces(traces, leads, context.image.shape)
    return Extraction(leads, traces, {'ownership': ownership})
