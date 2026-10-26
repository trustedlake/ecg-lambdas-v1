"""Recover measured narrow strokes attached to a trace, with joint ownership."""
import cv2
import numpy as np
from .signal_tinted import stroke_evidence
from .signal_neutral import summarize


def recover(image, leads, traces):
    measured, _ = stroke_evidence(image)
    # Detect components on the whole page: cropping must not turn a sustained
    # grid line into a short admissible peak.
    support = cv2.morphologyEx(measured.astype('uint8'), cv2.MORPH_OPEN,
                               np.ones((7, 1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    occupied = np.zeros(measured.shape, np.uint16)
    claims = np.zeros(measured.shape, np.uint16)
    candidates = []
    for t in traces:
        x, y = t['mask_origin']; h, w = t['mask'].shape
        occupied[y:y+h, x:x+w] += t['mask']
    for lead, t in zip(leads, traces):
        x, y = t['mask_origin']; h, w = t['mask'].shape
        local_labels = labels[y:y+h, x:x+w]
        touching = cv2.dilate(t['mask'].astype('uint8'), np.ones((3, 3), np.uint8)) > 0
        keep = np.zeros(n, bool)
        step = lead['row_spacing_px']
        for k in np.unique(local_labels[touching]):
            if not k: continue
            xx, yy, ww, hh, area = stats[k]
            keep[k] = (hh >= max(7, 1.2*ww) and hh <= .65*step
                       and ww <= .2*step and yy >= y and yy+hh <= y+h)
        candidate = cv2.dilate(keep[local_labels].astype('uint8'),
                               np.ones((3, 3), np.uint8)) > 0
        candidate &= measured[y:y+h, x:x+w]
        candidate &= occupied[y:y+h, x:x+w] == 0
        candidates.append(candidate)
        claims[y:y+h, x:x+w] += candidate
    report = []
    for t, candidate in zip(traces, candidates):
        x, y = t['mask_origin']; h, w = t['mask'].shape
        candidate &= claims[y:y+h, x:x+w] == 1
        added = int(candidate.sum())
        if added:
            t['mask'] |= candidate
            summarize(t, candidate.any(0))
            t['method'] = 'contrast_continuity_with_measured_narrow_strokes'
        report.append({'added_pixels': added})
    return report
