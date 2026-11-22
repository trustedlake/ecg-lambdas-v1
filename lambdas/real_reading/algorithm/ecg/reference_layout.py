"""Locate row origins, section markers and calibration pulses.

Everything found here is an origin. None of it bounds where a trace may go.
"""
import cv2
import numpy as np


def vertical_run_lengths(mask):
    """Length of the vertical run containing each measured pixel."""
    height, width = mask.shape
    padded = np.zeros((height+2, width), np.int8)
    padded[1:-1] = mask
    change = np.diff(padded, axis=0).T
    columns, starts = np.nonzero(change == 1)
    _, ends = np.nonzero(change == -1)
    result = np.zeros((width, height), np.int32)
    flat = result.ravel()
    lengths = ends-starts
    offsets = np.repeat(columns*height+starts, lengths)
    inside = np.arange(lengths.sum())-np.repeat(np.cumsum(lengths)-lengths, lengths)
    flat[offsets+inside] = np.repeat(lengths, lengths)
    return result.T


def locate_rows(mask, lengths, config, reference):
    """Row baselines measured in the reference; layout fractions are the fallback."""
    x0, y0, x1, y1 = reference
    expected = np.asarray(config['baseline_fractions'], float)*(y1-y0)
    count = len(expected)
    spacing = float(np.median(np.diff(expected))) if count > 1 else (y1-y0)*.5
    flat_limit = max(3, round(spacing*.035))
    flat = mask[y0:y1, x0:x1] & (lengths[y0:y1, x0:x1] <= flat_limit)
    profile = cv2.GaussianBlur(flat.mean(1).astype(np.float32).reshape(-1, 1),
                               (1, max(3, round(spacing*.03)) | 1), 0).ravel()
    remaining, centers = profile.copy(), []
    for _ in range(count):
        peak = int(remaining.argmax())
        if remaining[peak] <= 0:
            break
        centers.append(peak)
        radius = round(spacing*.5)
        remaining[max(0, peak-radius):peak+radius] = 0
    centers.sort()
    gaps = np.diff(centers)
    measured = len(centers) == count and (count < 2 or gaps.max() <= 1.6*gaps.min())
    if not measured:
        centers = []
        for prior in expected:
            low = max(0, int(prior-spacing*.3))
            high = max(low+1, min(len(profile), int(prior+spacing*.3)))
            part = profile[low:high]
            centers.append(low+int(part.argmax()) if part.max() > 0 else int(prior))
        centers.sort()
    step = float(np.median(np.diff(centers))) if count > 1 else spacing
    return np.asarray(centers, float)+y0, max(step, 20.), bool(measured)


def follow_baselines(mask, lengths, rows, step, reference):
    """A slowly varying baseline per row, continued across the whole photograph."""
    height, width = mask.shape
    x0, _, x1, _ = reference
    yy, xx = np.nonzero(mask & (lengths <= max(3, round(step*.035))))
    half = max(20, (x1-x0)//16)
    knots = np.arange(0, width+half, half)
    middle = int(np.abs(knots-(x0+x1)/2).argmin())
    paths = []
    for center in rows:
        values = np.full(len(knots), float(center))
        for indices in (range(middle, len(knots)), range(middle-1, -1, -1)):
            current = float(center)
            for k in indices:
                near = (np.abs(xx-knots[k]) <= half) & (np.abs(yy-current) < .22*step)
                if near.sum() >= half*.25:
                    current = float(np.median(yy[near]))
                # Curved paper moves gradually and must never capture the next row.
                current = float(np.clip(current, center-.3*step, center+.3*step))
                values[k] = current
        paths.append(np.interp(np.arange(width), knots, values))
    return np.asarray(paths, np.float32).clip(0, height-1)


def stroke_width(mask, lengths, bases, step, reference):
    x0, _, x1, _ = reference
    values = []
    for base in bases:
        for x in range(x0, x1, 3):
            top = max(0, int(base[x]-.1*step))
            part = lengths[top:int(base[x]+.1*step)+1, x]
            part = part[part > 0]
            if part.size:
                values.append(int(part.min()))
    return max(1., float(np.median(values))) if values else 1.


def remove_pulses(mask, bases, step, stroke, grid, calibration, reference, pitch):
    """Recognise a pulse by its three sides and its physical size."""
    cleaned, starts, found = mask.copy(), {}, []
    if not grid or not grid.get('valid') or not calibration:
        return cleaned, starts, found
    speed, gain = calibration.get('speed_mm_s'), calibration.get('gain_mm_mv')
    if not speed or not gain:
        return cleaned, starts, found
    tall = gain*grid['y']['pixels_per_mm']
    wide = .2*speed*grid['x']['pixels_per_mm']
    height, width = mask.shape
    left = max(0, int(reference[0]-.6*pitch))
    right = min(width, int(reference[0]+.3*pitch))
    if right-left < 4:
        return cleaned, starts, found
    for row, base in enumerate(bases):
        level = float(base[min(width-1, reference[0])])
        top, bottom = max(0, int(level-1.35*tall)), max(0, int(level-.65*tall))
        if bottom-top < 2:
            continue
        band = mask[top:bottom, left:right].astype(np.uint8)
        plateau = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((1, max(3, int(wide*.2))), np.uint8))
        joined = cv2.dilate(plateau, np.ones((max(3, round(stroke)), 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        for k in range(1, count):
            x, y, w, h, _ = stats[k]
            if not .65*wide <= w <= 1.45*wide or h > 4*stroke+4:
                continue
            first, last, upper = left+x, left+x+w-1, top+y
            reach = int(upper+.6*tall)
            margin = int(2*stroke+2)
            sides = [mask[upper:reach, max(0, side-margin):side+margin+1] for side in (first, last)]
            if any(not side.size or side.any(1).mean() < .85 for side in sides):
                continue
            end = min(width, last+margin)
            low = max(0, upper-margin)
            high = max(low, int(base[min(width-1, last)]-2*stroke))
            cleaned[low:high, max(0, first-2*margin):end] = False
            # The descending side ends on the baseline, where the signal starts.
            feet = min(height, int(base[min(width-1, last)]+3*stroke))
            cleaned[low:feet, max(0, first-2*margin):max(0, last-margin)] = False
            starts[row] = int(last)
            found.append([int(first), int(upper), int(w), int(high-upper)])
            break
    return cleaned, starts, found


def section_boundaries(mask, bases, step, stroke, counts, starts, grid, calibration, reference, pitch):
    """Measured markers divide origins; an agreed column is used where one is hidden."""
    height, width = mask.shape
    columns = max(counts)
    start = float(np.median(list(starts.values()))) if starts else float(reference[0])
    expected = [start+pitch*c for c in range(columns)]
    speed = (calibration or {}).get('speed_mm_s')
    if starts and speed and grid and grid.get('valid'):
        section = 10./columns*speed*grid['x']['pixels_per_mm']
        if abs(section/pitch-1) < .25:
            expected = [start+section*c for c in range(columns)]
    kernel = np.ones((max(7, int(step*.06)), max(3, round(stroke*1.6))), np.uint8)
    candidates = {}
    for row, base in enumerate(bases):
        for column in range(1, counts[row]):
            center = expected[column]
            left = max(0, int(center-.17*pitch))
            right = min(width, int(center+.17*pitch))
            if right-left < 4:
                continue
            level = base[int(np.clip(center, 0, width-1))]
            top = max(0, int(level-.3*step))
            local = mask[top:min(height, int(level+.3*step)), left:right].astype(np.uint8)
            thick = cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(thick, 8)
            for k in range(1, count):
                x, y, w, h, area = stats[k]
                if (.07*step <= h <= .3*step and w <= .055*step and area > .7*w*h
                        and y+top-.06*step <= base[left+x] <= y+top+h+.06*step):
                    yy, xx = np.nonzero(labels == k)
                    candidates.setdefault(column, []).append(
                        (row, left+x+w/2, yy+top, xx+left, [left+int(x), top+int(y), int(w), int(h)]))
    cleaned, markers = mask.copy(), []
    boundaries = np.tile(np.asarray(expected, float), (len(bases), 1))
    for row, value in starts.items():
        boundaries[row, 0] = value
    for column, options in candidates.items():
        # The same section starts at almost the same x in every row.
        def support(option):
            rows = {o[0] for o in options if abs(o[1]-option[1]) < .06*pitch}
            return len(rows), -abs(option[1]-expected[column])
        agreed = max(options, key=support)[1]
        near = [o for o in options if abs(o[1]-agreed) < .06*pitch]
        agreed = float(np.median([o[1] for o in near]))
        boundaries[:, column] = agreed
        for row in {o[0] for o in near}:
            chosen = min((o for o in near if o[0] == row), key=lambda o: abs(o[1]-agreed))
            boundaries[row, column] = chosen[1]
            cleaned[chosen[2], chosen[3]] = False
            markers.append(chosen[4])
    return cleaned, boundaries, markers
