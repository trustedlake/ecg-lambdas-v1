"""Observed grid/trace color separation, independently of waveform accuracy.

Uses the already estimated, *source-image* grid spacing and phase to sample
strokes, rather than looking for extraordinarily long continuous lines.
Uncertain measurement is an advisory, NEVER a reason to discard an ECG.
Only high-confidence evidence permits a veto: an actual Lab collision, or
fully achromatic measured grid and trace (same hue, even if brightness differs).
No model, reference image, clinical value or nominal config color is used.
"""
import cv2
import numpy as np

# Conservative, physically interpretable CIELAB distances. Distinguish brightness
# as well as hue: gray-grid/dark-waveform photos must not all be rejected.
COLLISION_DISTANCE = 8.0
COLLISION_FRACTION = .60
MAX_COLLISION_MEDIAN_DISTANCE = 8.0
MIN_COLLIDING_LEADS_FRACTION = .75
MIN_COLLISION_PER_LEAD = .50
MIN_TRACE_PIXELS = 100
MAX_SAMPLES_PER_LEAD = 1200
# User policy: a monochrome grid and monochrome trace are not chromatically
# distinguishable, regardless of their brightness difference. Strict neutral
# thresholds avoid confusing lightly tinted ECG paper with a gray-only source.
NEUTRAL_RGB_SPREAD = 2.5
MIN_NEUTRAL_TRACE_LEADS_FRACTION = .75
MIN_NEUTRAL_TRACE_PIXELS_PER_LEAD = 80
MIN_NEUTRAL_TRACE_PIXELS_FRACTION = .80


def _lab(rgb):
    a = np.clip(np.rint(np.asarray(rgb).reshape(-1, 1, 3)), 0, 255).astype('uint8')
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype('float32')
    lab[:, 0] *= 100 / 255
    return lab


def _observed_grid_colors(image, grid):
    """Sample narrow grid strokes at grid.py's measured period and phase.

    Scan both axes in regularly spaced samples, comparing each candidate with
    its flanks to avoid sampling bare paper. Line medians suppress isolated waveform crossings. Sampling must not depend
    on extracted masks: a mask contaminated by grid would censor the darkest
    grid pixels and manufacture apparent color separation.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype('float32')
    axis_reports = {}
    pools = {'minor': [], 'major': []}
    for axis, length in [('x', w), ('y', h)]:
        evidence = grid.get(axis, {}) if isinstance(grid, dict) else {}
        step = evidence.get('spacing_px')
        phase = evidence.get('phase_px')
        if (not evidence.get('valid') or step is None or phase is None or
                not np.isfinite(step) or not np.isfinite(phase) or step < 3.5):
            axis_reports[axis] = {'usable': False, 'reason': 'GRID_PERIOD_UNCERTAIN'}
            continue
        # grid.py calibrates the *major* phase, not a grid color. Repeated
        # minor lines are five divisions apart per major square.
        major = float(step) * grid.get('major_interval_minor_squares', 5)
        positions = []
        for k in range(-1, int((length + major) / major) + 2):
            for j in range(5):
                coordinate = phase + k * major + j * step
                if .05 * length <= coordinate <= .95 * length:
                    positions.append((coordinate, 'major' if j == 0 else 'minor'))
        # A 1px grid may be subpixel-shifted by interpolation; test +/-1px.
        flank = max(3, int(round(.38 * step)))
        colors = {'minor': [], 'major': []}
        n_valid = 0
        cross_size = h if axis == 'x' else w
        samples = np.arange(int(.09 * cross_size), int(.91 * cross_size),
                            max(3, int(cross_size / 550)))
        for position, kind in positions:
            pos = int(round(position))
            if pos - flank - 1 < 0 or pos + flank + 1 >= length:
                continue
            offsets = np.array([pos - 1, pos, pos + 1])
            if axis == 'x':
                darks = gray[samples[:, None], offsets[None, :]]
                pick = offsets[np.argmin(darks, axis=1)]
                left = gray[samples, pos - flank]
                right = gray[samples, pos + flank]
                valid = np.ones(len(samples), dtype=bool)
                values = image[samples, pick]
            else:
                darks = gray[offsets[:, None], samples[None, :]].T
                pick = offsets[np.argmin(darks, axis=1)]
                left = gray[pos - flank, samples]
                right = gray[pos + flank, samples]
                valid = np.ones(len(samples), dtype=bool)
                values = image[pick, samples]
            local_contrast = (left + right) / 2 - darks.min(axis=1)
            valid &= local_contrast >= 3.0
            n = int(np.count_nonzero(valid))
            if n >= 8:
                n_valid += 1
                # Representative of each physical line, prevents text/QRS
                # crossings or one dark line from taking over the grid pool.
                colors[kind].append(np.median(values[valid], axis=0))
        axis_reports[axis] = {'usable': n_valid >= 3, 'measured_lines': n_valid,
                              'candidate_lines': len(positions),
                              'spacing_px': round(float(step), 3),
                              'phase_px': round(float(phase), 3)}
        for kind in pools:
            pools[kind].extend(colors[kind])
    anchors = []
    for kind in ('minor', 'major'):
        arr = np.asarray(pools[kind], dtype='float32').reshape(-1, 3)
        if len(arr) >= 3:
            anchors.append(np.median(arr, axis=0))
    return np.asarray(anchors, dtype='float32').reshape(-1, 3), axis_reports, {
        kind: len(pools[kind]) for kind in pools}


def assess_color_separation(image, leads, traces, grid=None):
    """Return diagnostic and high-confidence collision decision.

    accepted=True on inconclusive measurements means *not vetoed*, not proof
    of ECG accuracy. `measurement_status` makes that distinction explicit.
    """
    report = {'method': 'grid_py_independent_sampling_rgb_lab_and_achromatic_v4',
              'accepted': True, 'block': False, 'reason': None,
              'measurement_status': 'INCONCLUSIVE', 'decision': 'WARN',
              'policy': {'collision_lab_distance': COLLISION_DISTANCE,
                         'collision_fraction': COLLISION_FRACTION,
                         'max_collision_median_distance': MAX_COLLISION_MEDIAN_DISTANCE,
                         'min_colliding_leads_fraction': MIN_COLLIDING_LEADS_FRACTION,
                         'min_trace_pixels': MIN_TRACE_PIXELS,
                         'neutral_rgb_spread_max': NEUTRAL_RGB_SPREAD,
                         'min_neutral_trace_leads_fraction': MIN_NEUTRAL_TRACE_LEADS_FRACTION,
                         'min_neutral_trace_pixels_fraction': MIN_NEUTRAL_TRACE_PIXELS_FRACTION,
                         'scope': 'Conservative color-collision screen; not accuracy'}}
    h, w = image.shape[:2]
    if not traces or len(leads) != len(traces):
        report['reason'] = 'INSUFFICIENT_TRACE_EVIDENCE'
        return report
    selected = []
    counts = {}
    for lead, trace in zip(leads, traces):
        mask = np.asarray(trace.get('mask'), dtype=bool)
        x, y = map(int, trace.get('mask_origin', [0, 0]))
        if mask.ndim != 2 or min(x, y) < 0 or x + mask.shape[1] > w or y + mask.shape[0] > h:
            report['reason'] = 'INVALID_TRACE_MASK_GEOMETRY'
            return report
        pixels = image[y:y+mask.shape[0], x:x+mask.shape[1]][mask]
        counts[lead['name']] = int(len(pixels))
        if len(pixels):
            interval = max(1, int(np.ceil(len(pixels) / MAX_SAMPLES_PER_LEAD)))
            selected.append((lead['name'], pixels[::interval][:MAX_SAMPLES_PER_LEAD]))
    report['trace_samples_per_lead'] = counts
    if sum(counts.values()) < MIN_TRACE_PIXELS:
        report['reason'] = 'INSUFFICIENT_TRACE_COLOR_EVIDENCE'
        return report
    anchors, axes, line_counts = _observed_grid_colors(image, grid or {})
    report['grid_evidence'] = {'axes': axes, 'lines_by_type': line_counts}
    if not len(anchors):
        report['reason'] = 'GRID_COLOR_NOT_MEASURABLE'
        return report
    pixels = np.concatenate([p for _, p in selected], axis=0)
    sample_lab = _lab(pixels)
    grid_lab = _lab(anchors)
    distance = np.linalg.norm(sample_lab[:, None, :] - grid_lab[None, :, :], axis=2).min(axis=1)
    collision = distance < COLLISION_DISTANCE
    per_lead = {}
    start = 0
    for name, sample in selected:
        segment = distance[start:start+len(sample)]
        per_lead[name] = {'median_lab_distance': round(float(np.median(segment)), 2),
                          'collision_fraction': round(float(np.mean(segment < COLLISION_DISTANCE)), 4),
                          'pixels': len(sample)}
        start += len(sample)
    report.update(measurement_status='MEASURED',
                  grid_anchors_rgb=[[round(float(c), 1) for c in a] for a in anchors],
                  trace_rgb_median=np.median(pixels, axis=0).round(1).tolist(),
                  trace_pixels_tested=int(len(pixels)),
                  lab_distance_median=round(float(np.median(distance)), 2),
                  color_collision_fraction=round(float(np.mean(collision)), 4),
                  per_lead=per_lead)
    # RESTRICTION: a decisive collision cannot be declared by one unmeasurable
    # lead. Per-lead information remains diagnostic; hard veto is global.
    robust_grid = all(axis.get('usable', False) and axis.get('measured_lines', 0) >= 6
                      for axis in axes.values()) and len(axes) == 2
    colliding_leads = (sum(v['collision_fraction'] >= MIN_COLLISION_PER_LEAD
                           for v in per_lead.values()) / len(per_lead))
    report['colliding_leads_fraction'] = round(float(colliding_leads), 4)
    report['grid_measurement_robust'] = bool(robust_grid)
    # Lab distance includes brightness and therefore missed source images
    # with R == G == B on both grid and trace (real cases 23 and 38).  A
    # distinct, conservative achromatic policy checks the measured colors,
    # not filenames, config classes or prior human labels. Strong grid phase
    # evidence and many neutral-colored traces are required to veto.
    anchor_rgb_spreads = np.ptp(anchors, axis=1)
    trace_color_spread = np.ptp(pixels, axis=1).astype('float32')
    neutral_leads = {}
    for name, sample in selected:
        spread = np.ptp(sample, axis=1)
        fraction = float(np.mean(spread <= NEUTRAL_RGB_SPREAD))
        neutral_leads[name] = {'neutral_fraction': round(fraction, 4),
                               'pixels': int(len(sample)),
                               'neutral': bool(len(sample) >= MIN_NEUTRAL_TRACE_PIXELS_PER_LEAD and
                                               fraction >= MIN_NEUTRAL_TRACE_PIXELS_FRACTION)}
    neutral_trace_leads_fraction = (sum(x['neutral'] for x in neutral_leads.values()) /
                                    max(1, len(neutral_leads)))
    chromatic_lab_distance = np.linalg.norm(sample_lab[:, None, 1:] -
                                            grid_lab[None, :, 1:], axis=2).min(axis=1)
    achromatic = (robust_grid and len(anchors) >= 2 and len(neutral_leads) >= 2 and
                  bool(np.all(anchor_rgb_spreads <= NEUTRAL_RGB_SPREAD)) and
                  neutral_trace_leads_fraction >= MIN_NEUTRAL_TRACE_LEADS_FRACTION and
                  float(np.mean(trace_color_spread <= NEUTRAL_RGB_SPREAD)) >=
                  MIN_NEUTRAL_TRACE_PIXELS_FRACTION)
    report['achromatic_separation'] = {
        'grid_anchor_rgb_spreads': [round(float(v), 2) for v in anchor_rgb_spreads],
        'trace_neutral_pixel_fraction': round(float(np.mean(
            trace_color_spread <= NEUTRAL_RGB_SPREAD)), 4),
        'neutral_trace_leads_fraction': round(neutral_trace_leads_fraction, 4),
        'per_lead': neutral_leads,
        'chromatic_lab_distance_median': round(float(np.median(chromatic_lab_distance)), 2),
        'confirmed_no_chromatic_separation': bool(achromatic)}
    confirmed = (robust_grid and len(per_lead) >= 2 and
                 np.median(distance) < MAX_COLLISION_MEDIAN_DISTANCE and
                 float(np.mean(collision)) >= COLLISION_FRACTION and
                 colliding_leads >= MIN_COLLIDING_LEADS_FRACTION)
    if confirmed:
        report.update(accepted=False, block=True, reason='CONFIRMED_TRACE_GRID_COLOR_COLLISION',
                      decision='REJECT_HIGH_CONFIDENCE')
    elif achromatic:
        report.update(accepted=False, block=True,
                      reason='ACHROMATIC_TRACE_GRID_NO_CHROMATIC_SEPARATION',
                      decision='REJECT_ACHROMATIC_COLOR_POLICY')
    elif not robust_grid:
        report.update(reason='GRID_COLOR_PARTIALLY_MEASURABLE', decision='WARN')
    elif float(np.mean(collision)) > .25:
        report.update(reason='LOW_TRACE_GRID_CONTRAST', decision='WARN')
    else:
        report.update(reason='TRACE_GRID_COLOR_SEPARATED', decision='NO_COLOR_VETO')
    return report
