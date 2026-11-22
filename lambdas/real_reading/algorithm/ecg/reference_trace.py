"""Trace the whole photograph; the reference gives only scale and origins.

There are no trace windows: not at the reference edges, between rows, or between
sections. Exported samples are always measured pixels, including stroke tips.
"""
import cv2
import numpy as np
from . import reference_layout as layout
from .config import COLORS
from .reference_ink import separate_ink, remove_spurs
from .reference_ownership import assign_rows, assign_sections, release_labels


def column_runs(column, bridge=3):
    edges = np.diff(np.concatenate(([0], column.astype(np.int8), [0])))
    runs = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1):
        # A stroke interrupted by a printed line is still one measured stroke.
        if runs and a-runs[-1][1]-1 <= bridge:
            runs[-1][1] = int(b)
        else:
            runs.append([int(a), int(b)])
    return runs


def continuous_path(columns, baseline):
    """The run per column that continues the trace; long measured strokes win."""
    if not columns:
        return []
    costs, back, previous = [], [], None
    for runs in columns:
        runs = np.asarray(runs, float)
        reward = .05*(runs[:, 1]-runs[:, 0])
        if previous is None:
            cost = .1*np.abs(runs.mean(1)-baseline)
            chosen = np.zeros(len(runs), int)
        else:
            above = previous[:, 0][None, :]-runs[:, 1][:, None]
            below = runs[:, 0][:, None]-previous[:, 1][None, :]
            jump = costs[-1][None, :]+np.maximum(0, np.maximum(above, below)-1)
            chosen = jump.argmin(1)
            cost = jump[np.arange(len(runs)), chosen]
        costs.append(cost-reward)
        back.append(chosen)
        previous = runs
    at = int(costs[-1].argmin())
    path = [at]
    for chosen in back[:0:-1]:
        at = int(chosen[at])
        path.append(at)
    return path[::-1]


def lead_baseline(mask, base, step, stroke):
    yy, xx = np.nonzero(mask)
    near = np.abs(yy-base[xx]) < .2*step
    if not near.any():
        return float(np.median(base[xx])) if len(xx) else float(np.median(base))
    ys, counts = np.unique(yy[near], return_counts=True)
    profile = np.bincount(ys-ys.min(), weights=counts)
    size = max(1, round(stroke))
    # The isoelectric line is the most populated level, not the wave median.
    return float(ys.min()+np.argmax(np.convolve(profile, np.ones(size), 'same')))


def sample_columns(mask, disputed, baseline, step, stroke):
    """One measured value per column that reaches the tip of a steep stroke."""
    width = mask.shape[1]
    y = np.full(width, np.nan, np.float32)
    low, high = y.copy(), y.copy()
    ambiguous = np.zeros(width, bool)
    selected = np.zeros_like(mask)
    present = np.flatnonzero(mask.any(0))
    columns = [column_runs(mask[:, x]) for x in present]
    for x, runs, pick in zip(present, columns, continuous_path(columns, baseline)):
        a, b = runs[pick]
        selected[a:b+1, x] = mask[a:b+1, x]
        low[x], high[x] = a, b
        if b-a+1 <= 2*stroke:
            y[x] = a+int(np.argmin(np.where(mask[a:b+1, x],
                                            np.abs(np.arange(a, b+1)-(a+b)//2), np.inf)))
        else:
            # The extreme measured pixel, never the middle of a steep stroke.
            y[x] = a if abs(a-baseline) >= abs(b-baseline) else b
    ambiguous |= disputed
    return selected, y, low, high, ambiguous


def trace_reference(context, raw, contrast):
    from research_methods.contracts import Extraction, MethodAbstention
    config, reference = context.config, context.geometry['reference_box_px']
    x0, _, x1, _ = reference
    ink, threshold = separate_ink(context.image, np.asarray(raw), reference)
    if not ink.any():
        raise MethodAbstention('No measured ink separable from the grid')
    matrix = config['leads']+[[name] for name in config.get('rhythm_leads', [])]
    counts = [len(names) for names in matrix]
    pitch = (x1-x0)/max(counts)
    lengths = layout.vertical_run_lengths(ink)
    rows, step, measured_rows = layout.locate_rows(ink, lengths, config, reference)
    bases = layout.follow_baselines(ink, lengths, rows, step, reference)
    stroke = layout.stroke_width(ink, lengths, bases, step, reference)
    calibration = config.get('calibration')
    ink, starts, pulses = layout.remove_pulses(
        ink, bases, step, stroke, context.grid, calibration, reference, pitch)
    ink, boundaries, markers = layout.section_boundaries(
        ink, bases, step, stroke, counts, starts, context.grid, calibration, reference, pitch)
    ink = remove_spurs(ink, layout.vertical_run_lengths(ink), step, stroke)
    lengths = layout.vertical_run_lengths(ink)
    owned, disputed, strokes = assign_rows(ink, lengths, bases, step, pitch, stroke, reference)
    labels_found = release_labels(ink, owned, markers, step, stroke)
    height, width = ink.shape
    claims = owned.sum(0)
    leads, traces = [], []
    for row, names in enumerate(matrix):
        base = bases[row]
        edges = boundaries[row, :len(names)]
        sections = assign_sections(owned[row], lengths, base, edges, step, stroke)
        # An undecidable piece only casts doubt on the trace that it touches.
        touching = cv2.dilate(owned[row].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        unclear = (disputed & touching & ~owned[row]).sum(0) > stroke
        for column, name in enumerate(names):
            mask = sections == column
            origin = float(edges[column])
            present = np.flatnonzero(mask.any(0))
            end = float(edges[column+1]) if column+1 < len(names) else float(x1)
            first = int(np.clip(origin, 0, width-1))
            if present.size:
                # Keep the section's time origin even when its first columns are gaps.
                left, right = min(first, int(present[0])), max(first+1, int(present[-1])+1)
            else:
                left, right = first, max(first+1, min(width, int(end)))
            part = mask[:, left:right]
            baseline = lead_baseline(part, base[left:right], step, stroke)
            selected, y, low, high, ambiguous = sample_columns(
                part, unclear[left:right], baseline, step, stroke)
            observed = selected.any(0)
            valid = observed & ~ambiguous
            found = np.flatnonzero(observed)
            lead = dict(name=name, row=row, column=column, baseline_y_px=baseline,
                        color=COLORS.get(name, '#344b5c'),
                        row_spacing_px=step, trace_x=[left, right], time_origin_x_px=origin,
                        box=[left, int(np.nanmin(low)) if found.size else int(baseline),
                             right, int(np.nanmax(high))+1 if found.size else int(baseline)+1])
            leads.append(lead)
            traces.append(dict(
                mask=selected, mask_origin=[left, 0],
                x_px=np.arange(left, right, dtype=np.float32), y_px=y,
                y_min_px=low, y_max_px=high, observed=observed, valid=valid,
                ambiguous=ambiguous, baseline_y_px=baseline, time_origin_x_px=origin,
                coverage=float(observed.mean()), unambiguous_fraction=float(valid.mean()),
                method='reference_origin_full_photo', baseline_method='measured_mode'))
    decided = [{k: (v if k == 'reach' else float(v) if v is not None else None)
                for k, v in s.items()} for s in strokes]
    return Extraction(leads, traces, {
        'reference_box_px': list(reference), 'ink_threshold': threshold,
        'rows_measured': measured_rows, 'baselines_px': rows.tolist(),
        'stroke_width_px': stroke, 'measured_markers': markers,
        'measured_section_labels': labels_found,
        'measured_calibration_pulses': pulses, 'crossing_strokes': decided,
        'undecidable_pixels': int(disputed.sum()),
        'crossing_pixels': int((claims > 1).sum())})
