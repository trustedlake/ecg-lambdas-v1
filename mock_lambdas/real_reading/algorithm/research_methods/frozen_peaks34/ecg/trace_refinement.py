"""Opt-in measured recovery after conservative ownership, with grid-branch pruning.

Only explicitly configured leads change. A geodesic margin is a heuristic, not
proof of ownership: every changed column remains ambiguous. No old output,
filename, digital reference or interpolated waveform enters this module.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
from .trace_ownership import _anchors, _distances

def lattice(image, grid):
    """Locate repeated red grid ridges; calibration itself is never changed.

    Local interpolation tracks only nuisance-grid position and brightness.
    It never creates trace evidence. Dark strokes and short crossings are
    handled separately from the weak grid-coloured extensions.
    """
    redness = np.maximum(0, image[:, :, 0].astype(float) - image[:, :, 1].astype(float))
    excluded = np.zeros(image.shape[:2], bool)
    geometry = excluded.copy()
    for transpose, axis in [(False, 'x'), (True, 'y')]:
        score = redness.T if transpose else redness
        red = image[:, :, 0].T if transpose else image[:, :, 0]
        height, width = score.shape
        period = grid[axis]['spacing_px'] * grid.get('major_interval_minor_squares', 5)
        if not np.isfinite(period) or period < 8:
            continue  # An unresolved lattice cannot justify removing pixels.
        profile = cv2.GaussianBlur(score.mean(0).astype('float32')[None, :], (3, 1), .6).ravel()
        peaks, _ = find_peaks(profile, distance=int(.75*period), prominence=1)
        local = np.zeros((height, width), bool)
        local_geometry = local.copy()
        for index, peak in enumerate(peaks):
            neighbours = peaks[max(0, index-1):index+2]
            if not any(abs(abs(peak-other)-period) < .18*period for other in neighbours if other != peak):
                continue
            rows, columns, intensity = [], [], []
            for y in range(0, height, 80):
                low = max(0, int(peak-period*.2))
                high = min(width, int(peak+period*.2)+1)
                strength = score[y:min(y+80, height), low:high].mean(0)
                strength = cv2.GaussianBlur(strength.astype('float32')[None, :], (3, 1), .6).ravel()
                x = low + int(strength.argmax())
                rows.append(min(height-1, y+40)); columns.append(x)
                intensity.append(float(np.median(red[y:min(y+80, height), max(0, x-1):x+2])))
            path = np.rint(np.interp(np.arange(height), rows, columns)).astype(int)
            level = np.interp(np.arange(height), rows, intensity) - 40
            for offset in range(-2, 3):
                xx = np.clip(path+offset, 0, width-1); yy = np.arange(height)
                local[yy, xx] |= red[yy, xx] >= level
                local_geometry[yy, xx] = True
        excluded |= local.T if transpose else local
        geometry |= local_geometry.T if transpose else local_geometry
    return excluded, geometry


def _recompute(trace, selected):
    previous = trace['mask']
    changed = (previous ^ selected).any(0)
    observed = selected.any(0)
    width = selected.shape[1]
    center = np.full(width, np.nan, np.float32)
    low = center.copy(); high = center.copy()
    y0 = trace['mask_origin'][1]
    for x in np.flatnonzero(observed):
        points = np.flatnonzero(selected[:, x]) + y0
        center[x] = np.median(points); low[x] = points[0]; high[x] = points[-1]
    ambiguous = trace['ambiguous'] | changed
    valid = observed & ~ambiguous
    trace.update(mask=selected, y_px=center, y_min_px=low, y_max_px=high,
                 observed=observed, ambiguous=ambiguous, valid=valid,
                 baseline_y_px=float(np.nanmedian(center)) if observed.any() else trace['baseline_y_px'],
                 coverage=float(observed.mean()), unambiguous_fraction=float(valid.mean()))


def refine(image, raw, contrast, grid, leads, proposed, traces, names):
    """Recover unassigned supported pixels; never steal another final owner."""
    shape = raw.shape
    claims = np.zeros(shape, np.uint16)
    occupied = np.zeros(shape, np.int16)
    for i, (pre, trace) in enumerate(zip(proposed, traces), 1):
        x, y = trace['mask_origin']; h, w = trace['mask'].shape
        claims[y:y+h, x:x+w] += pre['mask']
        region = occupied[y:y+h, x:x+w]
        region[trace['mask']] = i
    graph = cv2.morphologyEx((claims > 0).astype('uint8'), cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    graph = cv2.morphologyEx(graph, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    distances = []
    for lead, pre in zip(leads, proposed):
        x, y = pre['mask_origin']; h, w = pre['mask'].shape
        distance = np.full(shape, np.inf, np.float32)
        distance[y:y+h, x:x+w] = _distances(graph[y:y+h, x:x+w], _anchors(pre, lead))
        distances.append(distance)
    distances = np.stack(distances)
    nuisance, grid_geometry = lattice(image, grid)
    candidates = {}
    for i, (lead, trace) in enumerate(zip(leads, traces)):
        if lead['name'] not in names: continue
        step = lead['row_spacing_px']
        runner = np.min(np.delete(distances, i, axis=0), axis=0) if len(leads)>1 else np.full(shape, np.inf)
        recover = ((claims > 0) & (raw > 0) & (contrast > 12)
                   & (distances[i] < .85*step) & (runner > distances[i] + .27*step)
                   & (occupied == 0))
        candidates[i] = recover
    # Reject any new tie simultaneously, so lead iteration order cannot decide.
    overlap = np.zeros(shape, np.uint8)
    for recover in candidates.values(): overlap += recover
    report = {'method': 'measured_geodesic_margin_and_periodic_grid_pruning', 'leads': {},
              'limitation': 'Heuristic ownership; all changed columns are ambiguous, not verified samples.'}
    for i, (lead, trace) in enumerate(zip(leads, traces)):
        if i not in candidates: continue
        x, y = trace['mask_origin']; h, w = trace['mask'].shape
        region = np.s_[y:y+h, x:x+w]
        original = trace['mask']
        selected = original | (candidates[i][region] & (overlap[region] == 1))
        # A short crossing supported on neighbouring non-grid pixels survives.
        off_grid = selected & ~grid_geometry[region]
        nearby = cv2.dilate(off_grid.astype('uint8'), np.ones((9, 9), np.uint8)) > 0
        selected &= ~(nuisance[region] & ~nearby)
        report['leads'][lead['name']] = {'added_pixels': int((selected & ~original).sum()),
            'removed_pixels': int((original & ~selected).sum()),
            'changed_columns': int((selected ^ original).any(0).sum())}
        _recompute(trace, selected)
    return report
