"""Opt-in measured dark/cool ink recovery for photographed red-grid reports.

Small closing joins fragments for connectivity only. It never supplies exported
pixels. Layout, grid calibration and the mandatory color gate remain unchanged.
"""
import cv2
import numpy as np
from .signal_neutral import extract_lead as neutral_extract, summarize


def ink_evidence(image):
    f = image.astype(np.float32)
    light = f.max(2)
    # Black ink may acquire a blue cast in a photograph. Absolute chroma would
    # discard it; red-dominant grid ink is excluded unless genuinely dark.
    raw = (light < 100) | ((f[..., 0] < 110) & (light < 180) &
                           (f[..., 0] - f[..., 2] < 8))
    return raw.astype('uint8'), 255 - light


def extract_lead(raw, contrast, lead):
    measured = raw.copy()
    x0, x1 = lead['trace_x']
    base, step = lead['baseline_y_px'], lead['row_spacing_px']
    y0 = max(0, int(base - step * .8))
    y1 = min(len(raw), int(base + step * .65))
    roi = measured[y0:y1, x0:x1]
    excluded = np.zeros(x1 - x0, bool)
    # Segment separators sit at the right boundary. Do not apply this to the
    # left boundary: a genuine first QRS can be close to it.
    vertical = cv2.morphologyEx(roi, cv2.MORPH_OPEN,
                               np.ones((max(9, int(step * .12)), 1), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if (x + w > roi.shape[1] - step * .18 and w >= 3 and
                step * .12 <= h <= step * .35 and area / (w * h) > .8):
            excluded[max(0, x - 1):min(len(excluded), x + w + 1)] = True
    roi[:, excluded] = 0
    # Printed next-lead labels below the separator are not waveform pixels.
    edge = max(0, int(roi.shape[1] - step * .18))
    below = max(0, int(base + step * .08) - y0)
    roi[below:, edge:] = 0
    connected = cv2.morphologyEx(measured, cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8))
    # Keep excluded columns disconnected even if closing could bridge them.
    connected[y0:y1, x0:x1][:, excluded] = 0
    trace = neutral_extract(connected, contrast, lead)
    x, y = trace['mask_origin']
    h, w = trace['mask'].shape
    trace['mask'] &= measured[y:y+h, x:x+w] > 0
    bridged = (connected[y:y+h, x:x+w] > measured[y:y+h, x:x+w]).any(0)
    summarize(trace, excluded | bridged)
    trace['method'] = 'printed_cool_dark_measured_components'
    return trace


def refine(image, leads, traces, names):
    raw, contrast = ink_evidence(image)
    occupied = np.zeros(raw.shape, bool)
    counts = np.zeros(raw.shape, np.uint16)
    candidates = {}
    for i, (lead, trace) in enumerate(zip(leads, traces)):
        if lead['name'] in names:
            candidate = extract_lead(raw, contrast, lead)
            x, y = candidate['mask_origin']; h, w = candidate['mask'].shape
            counts[y:y+h, x:x+w] += candidate['mask']
            candidates[i] = candidate
        else:
            x, y = trace['mask_origin']; h, w = trace['mask'].shape
            occupied[y:y+h, x:x+w] |= trace['mask']
    report = {}
    for i, trace in candidates.items():
        x, y = trace['mask_origin']; h, w = trace['mask'].shape
        conflict = trace['mask'] & (occupied[y:y+h, x:x+w] |
                                    (counts[y:y+h, x:x+w] > 1))
        trace['mask'] &= ~conflict
        old = np.zeros(raw.shape, bool)
        ox, oy = traces[i]['mask_origin']; oh, ow = traces[i]['mask'].shape
        old[oy:oy+oh, ox:ox+ow] = traces[i]['mask']
        changed = (old[y:y+h, x:x+w] != trace['mask']).any(0)
        summarize(trace, changed | conflict.any(0))
        trace['method'] = 'printed_cool_dark_measured_components'
        traces[i] = trace
        report[leads[i]['name']] = {
            'method': trace['method'], 'conflicting_pixels_withheld': int(conflict.sum()),
            'limitation': 'Changed/steep columns remain uncertain; measured pixels only, no interpolation.'}
    return report
