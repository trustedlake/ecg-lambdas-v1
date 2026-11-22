"""Explicit focal local-contrast recovery, preserving all unselected leads."""
from copy import deepcopy
import numpy as np
from . import signal_affine as affine
from .trace_ownership import separate_traces
from .signal_neutral import summarize


def ink_evidence(image):
    raw, contrast = affine.ink_evidence(image)
    return (raw & (contrast > 12)).astype('uint8'), contrast


def refine(image, leads, traces, names):
    raw, contrast = ink_evidence(image)
    proposed = deepcopy(traces)
    occupied = np.zeros(image.shape[:2], bool)
    for i, lead in enumerate(leads):
        if lead['name'] in names:
            proposed[i] = affine.recover_stroke_pixels(
                raw, affine.trace_lead(raw, contrast, lead), lead)
        else:
            t = traces[i]; x, y = t['mask_origin']; h, w = t['mask'].shape
            occupied[y:y+h, x:x+w] |= t['mask']
    separate_traces(proposed, leads, image.shape)
    report = {}
    for i, lead in enumerate(leads):
        if lead['name'] not in names:
            continue
        t = proposed[i]; x, y = t['mask_origin']; h, w = t['mask'].shape
        t['mask'] &= ~occupied[y:y+h, x:x+w]
        old = np.zeros(image.shape[:2], bool)
        previous = traces[i]; ox, oy = previous['mask_origin']; oh, ow = previous['mask'].shape
        old[oy:oy+oh, ox:ox+ow] = previous['mask']
        changed = (old[y:y+h, x:x+w] != t['mask']).any(0)
        summarize(t, changed)
        t['method'] = 'focal_local_contrast_exclusive_measured'
        traces[i] = t
        report[lead['name']] = {'method': t['method'],
            'limitation': 'Overlapping peaks remain partial; changed columns uncertain; no interpolated pixels.'}
    return report
