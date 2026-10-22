"""Explicit dark-grid pilot: measured strokes, joint ownership, abstention.

The lattice is an appearance mask, not a change to grid calibration. Grid
crossings are withheld; connectivity closing never creates exported pixels.
No filenames, previous outputs or reference signals enter this module.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
from .signal_tinted import locate_leads
from .trace_ownership import _distances


def grid_mask(ink, grid):
    mask = np.zeros_like(ink, dtype=np.uint8)
    for transpose, key in [(False, 'y'), (True, 'x')]:
        im = ink.T.copy() if transpose else ink
        out = np.zeros_like(im); h, w = im.shape
        period = grid[key]['spacing_px'] * grid['major_interval_minor_squares']
        if not np.isfinite(period) or period < 8:
            raise ValueError('Dark-grid profile needs a resolved major-grid period')
        profile = cv2.GaussianBlur(im.mean(1).astype('float32').reshape(-1, 1), (1, 9), 1).ravel()
        peaks, _ = find_peaks(profile, distance=int(period * .7), prominence=.04)
        indices = np.rint(peaks / period).astype(int)
        good = abs(peaks - indices * period) < period * .22
        if good.sum() < 4:
            raise ValueError('Insufficient repeated grid lines for dark-grid profile')
        lattice = np.polyfit(indices[good], peaks[good], 1)
        for peak in np.polyval(lattice, np.arange(int(h / period) + 2)):
            if not 0 <= peak < h:
                continue
            xs, ys = [], []
            for x0 in range(0, w, 100):
                x1 = min(w, x0 + 100)
                lo, hi = max(0, int(peak - period * .3)), min(h, int(peak + period * .3) + 1)
                strength = im[lo:hi, x0:x1].mean(1)
                strength = cv2.GaussianBlur(strength.astype('float32').reshape(-1, 1), (1, 3), .6).ravel()
                yi = int(np.argmax(strength))
                if strength[yi] > .25:
                    xs.append((x0 + x1 - 1) / 2); ys.append(yi + lo)
            if len(xs) < 4:
                continue
            xs, ys = np.array(xs), np.array(ys)
            coef = np.polyfit(xs, ys, 1)
            for _ in range(3):
                keep = abs(ys - np.polyval(coef, xs)) < 2.5
                if keep.sum() >= 4:
                    coef = np.polyfit(xs[keep], ys[keep], 1)
            if keep.sum() < 4:
                continue
            # Interpolate only the nuisance grid position, never the ECG signal.
            line = np.interp(np.arange(w), xs[keep], ys[keep])
            for delta in range(-2, 3):
                yy = np.rint(line).astype(int) + delta
                ok = (yy >= 0) & (yy < h)
                out[yy[ok], np.arange(w)[ok]] = 1
        mask |= out.T if transpose else out
    return mask.astype(bool)


def ink_evidence(a, grid):
    if not grid['valid']:
        raise ValueError('Dark-grid profile requires consistent measured grid geometry')
    gray = a.max(2)
    background = float(np.percentile(gray, 90))
    detector = (gray < background * .71).astype('uint8')
    ink = (gray < background * .65).astype('uint8')
    excluded = grid_mask(detector, grid)
    # Remove remaining sustained straight runs; short steep waveform strokes
    # are shorter than the vertical kernel. Intersections remain unknown.
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 91), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((301, 1), np.uint8))
    excluded |= cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)) > 0
    return (ink > 0) & ~excluded, excluded


def trace_all(raw, excluded, leads):
    graph = cv2.morphologyEx(raw.astype('uint8'), cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    graph = cv2.morphologyEx(graph, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8)).astype(bool)
    best = np.full(raw.shape, np.inf, np.float32); second = best.copy()
    owner = np.full(raw.shape, -1, np.int16)
    claims = np.zeros(raw.shape, np.uint16)
    regions = []
    for index, lead in enumerate(leads):
        step, base = lead['row_spacing_px'], lead['baseline_y_px']
        x0, x1 = lead['trace_x']
        y0, y1 = max(0, int(base - step * 1.2)), min(raw.shape[0], int(base + step))
        region = np.s_[y0:y1, x0:x1]; regions.append((x0, y0, x1, y1))
        seed = raw[region] & (abs(np.arange(y0, y1)[:, None] - base) < step * .18)
        seed = cv2.morphologyEx(seed.astype('uint8'), cv2.MORPH_OPEN, np.ones((1, 7), np.uint8)) > 0
        distance = _distances(graph[region], seed)
        reachable = distance < step
        claims[region] += reachable & raw[region]
        distance[~reachable] = np.inf
        current, runner = best[region], second[region]
        better = distance < current
        runner[:] = np.where(better, current, np.minimum(runner, distance))
        owner[region][better] = index
        current[:] = np.minimum(current, distance)
    # Any two reachable owners cause abstention, not a nearest-row decision.
    disputed = cv2.dilate((claims > 1).astype('uint8'), np.ones((3, 3), np.uint8)) > 0
    owner[~np.isfinite(best) | disputed] = -1
    traces = []
    for index, (lead, (x0, y0, x1, y1)) in enumerate(zip(leads, regions)):
        region = np.s_[y0:y1, x0:x1]
        selected = raw[region] & (owner[region] == index)
        observed = selected.any(0); width = x1 - x0
        center = np.full(width, np.nan, np.float32); low = center.copy(); high = center.copy()
        near_grid = cv2.dilate(excluded[region].astype('uint8'), np.ones((3, 3), np.uint8)) > 0
        ambiguous = ~observed | (selected & near_grid).any(0) | (raw[region] & disputed[region]).any(0)
        for col in np.flatnonzero(observed):
            points = np.flatnonzero(selected[:, col]) + y0
            center[col] = np.median(points); low[col] = points[0]; high[col] = points[-1]
            if np.any(np.diff(points) > 1):
                ambiguous[col] = True
        valid = observed & ~ambiguous
        traces.append(dict(x_px=np.arange(x0, x1, dtype=np.float32), y_px=center,
            y_min_px=low, y_max_px=high, mask=selected, mask_origin=[x0, y0],
            observed=observed, ambiguous=ambiguous, valid=valid,
            baseline_y_px=float(np.nanmedian(center)) if observed.any() else float(lead['baseline_y_px']),
            baseline_method='median_observed_trace_not_clinical_isoelectric_reference',
            method='dark_grid_joint_measured_strokes_abstain', coverage=float(observed.mean()),
            unambiguous_fraction=float(valid.mean())))
    report = dict(method='dark_grid_joint_measured_strokes_abstain',
        shared_pixels_before=int((claims > 1).sum()), shared_pixels_after=0,
        disputed_measured_pixels=int((disputed & raw).sum()),
        limitation='Exclusive selection does not prove correct ownership; grid crossings are withheld.')
    return traces, report
