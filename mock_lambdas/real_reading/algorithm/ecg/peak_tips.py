"""Restore the printed tip of sharp QRS spikes.

A per-column run centre stops short of the true tip of a pointed stroke. At each
local peak or valley of the exported curve, move the sample to the most extreme
pixel of the contiguous mask run that continues from it. Only mask pixels are
used; nothing is interpolated or invented.
"""
import numpy as np


def extend_tips(trace, window=6, prominence=3):
    mask = np.asarray(trace['mask'], bool)
    mx0, my0 = trace['mask_origin']
    h, w = mask.shape
    y = trace['y_px']
    x = np.asarray(trace['x_px'], float)
    finite = np.flatnonzero(np.isfinite(y))
    if finite.size < 5:
        return 0
    yf = y[finite].astype(float)
    moved = 0
    new = {}
    for k, i in enumerate(finite):
        lo = max(0, k - window); hi = min(finite.size, k + window + 1)
        ring = yf[lo:hi]
        if hi - lo < 5:
            continue
        if yf[k] >= ring.max() and yf[k] - min(ring[0], ring[-1]) >= prominence:
            step = 1
        elif yf[k] <= ring.min() and max(ring[0], ring[-1]) - yf[k] >= prominence:
            step = -1
        else:
            continue
        start = int(round(yf[k] - my0)); c0 = int(round(x[i] - mx0))
        best = start; found = False
        for col in range(max(0, c0 - 1), min(w, c0 + 2)):
            if not (0 <= start < h and mask[start, col]):
                continue
            r = start
            while 0 <= r + step < h and mask[r + step, col]:
                r += step
            if (r - best) * step > 0 or not found:
                if (r - best) * step >= 0:
                    best = r; found = True
        if found and best != start:
            new[i] = best + my0
    for i, v in new.items():
        y[i] = v; moved += 1
    return moved
