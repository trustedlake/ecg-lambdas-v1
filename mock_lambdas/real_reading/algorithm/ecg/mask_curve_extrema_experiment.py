"""Experimental shared mask-to-curve reconstruction; no identity or lead rules.

Retain dominant centerline excursions instead of column medians. Multi-valued
columns remain uncertain. No resampling, interpolation or invented timing.
"""
import numpy as np
import cv2


def thin(mask):
    """Zhang-Suen thinning, with zero padding and no gap closing."""
    a = np.pad(np.asarray(mask, bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            n = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[k] & p[(k+1) % 8]).astype(np.uint8) for k in range(8))
            if phase == 0:
                keep = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                keep = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = a[1:-1, 1:-1] & (n >= 2) & (n <= 6) & (transitions == 1) & keep
            if remove.any():
                a[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return a[1:-1, 1:-1]


def reconstruct(trace, baseline_slope=0., rgb=None):
    """One sample per column, retaining the centerline's dominant excursion.

    A raster column can contain both sides of a steep QRS. Such columns remain
    ambiguous: choosing an observed extremum is a review candidate, not proof
    of the original waveform or an invented higher sampling frequency.
    """
    mask = np.asarray(trace['mask'], bool)
    x0, y0 = trace['mask_origin']
    h, w = mask.shape
    support = mask.copy()
    if rgb is not None:
        # Reject mask residues unsupported by local neutral darkness. Original
        # extraction is immutable; this is explicit numerical evidence filtering.
        raw = np.asarray(rgb, np.float32)
        strength = (cv2.GaussianBlur(raw,(0,0),7)-raw).min(axis=2)
        local = strength[y0:y0+h,x0:x0+w]
        positive = local[mask & (local>0)]
        cutoff = max(3., float(np.percentile(positive,50))*.15) if positive.size else 3.
        support &= local > cutoff
    skeleton = thin(support)
    x = np.asarray(trace['x_px'], np.float64).copy()
    baseline = trace['baseline_y_px'] + baseline_slope*(x-(x0+w/2))
    y = np.full(w, np.nan)
    ambiguous = np.asarray(trace['ambiguous'], bool).copy()
    for c in range(w):
        points = np.flatnonzero(skeleton[:,c])
        if not len(points):
            points = np.flatnonzero(support[:,c])
            if not len(points): continue
            points = points[len(points)//2:len(points)//2+1]
        values = points+y0
        y[c] = values[len(values)//2]
        ambiguous[c] |= len(points)>1
    # Preserve measured local centerline extrema at ONE image column per tip,
    # rather than drawing a rectangular envelope across the whole QRS width.
    top = np.full(w,np.nan); bottom=top.copy()
    for c in range(w):
        points=np.flatnonzero(skeleton[:,c])
        if len(points): top[c]=points[0]+y0; bottom[c]=points[-1]+y0
    proposals={}
    for curve,sign in ((top,-1),(bottom,1)):
        candidate=[]
        for c in range(1,w-1):
            if not np.isfinite(curve[c]): continue
            window=curve[max(0,c-2):min(w,c+3)]
            extreme=np.nanmin(window) if sign<0 else np.nanmax(window)
            excursion=sign*(curve[c]-baseline[c])
            if curve[c]==extreme and excursion>3:
                candidate.append(c)
        groups=np.split(np.asarray(candidate,dtype=int),np.flatnonzero(np.diff(candidate)>1)+1) if candidate else []
        for group in groups:
            center=group[len(group)//2]
            value=curve[center]
            if center not in proposals or abs(value-baseline[center])>abs(proposals[center]-baseline[center]):
                proposals[center]=value
    for c,value in proposals.items():
        y[c]=value
        ambiguous[c]=True
    observed = np.isfinite(y)
    return dict(x_px=x, y_px=y, y_min_px=np.asarray(trace['y_min_px']).copy(),
                y_max_px=np.asarray(trace['y_max_px']).copy(), baseline_y_px=baseline,
                observed=observed, ambiguous=ambiguous, valid=observed & ~ambiguous)
