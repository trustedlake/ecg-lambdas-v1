"""Measured periodic-grid exclusion; abstains at removed intersections."""
import cv2
import numpy as np
from .common import trace_evidence
from .contracts import MethodAbstention
from .frozen_peaks34.ecg.signal_dark import grid_mask

VERSION = 'periodic-grid-v1'


def extract(context):
    if not context.grid['valid']:
        raise MethodAbstention('Consistent grid measurement required')
    gray = cv2.cvtColor(context.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), 7)
    contrast = np.maximum(0, background-gray)
    ink = ((contrast > 6) & (gray < .90*np.maximum(background, 1))).astype(np.uint8)
    try:
        excluded = grid_mask(ink, context.grid)
    except ValueError as error:
        if str(error) in ('Dark-grid profile needs a resolved major-grid period',
                          'Insufficient repeated grid lines for dark-grid profile'):
            raise MethodAbstention(str(error)) from error
        raise
    result = trace_evidence(context, ink & ~excluded, contrast)
    result.diagnostics['excluded_grid_pixels'] = int(excluded.sum())
    result.diagnostics['limitation'] = 'Grid intersections withheld; lost peaks are possible.'
    return result
