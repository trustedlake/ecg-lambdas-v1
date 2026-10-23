"""Rule-based ECG diagnoser (screening aid, not a medical report).

Works on the extracted signal (mV, native sampling rate, gaps as None) using published
interpretation criteria that can be checked on an extracted trace: QRS duration, voltage
criteria for hypertrophy, axis and the rate/rhythm thresholds. ST, Q-wave and bundle
branch findings use rules not yet validated on photographed traces, so they are reported
as "possible" with a review alert. Every result carries a three-level alert (ok / review /
urgent) biased towards false positives: values close to a limit and partial readings also
ask for review. Anything that
cannot be measured reliably from a photographed trace (P wave, PR, QT) is returned as not
measurable instead of being estimated. Stdlib only. No tuning per image.
"""

import math
import statistics

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# Contiguous lead pairs (Cabrera frontal order and precordial order; aVR excluded).
CONTIGUOUS = {frozenset(p) for p in (("aVL", "I"), ("II", "aVF"), ("aVF", "III"), ("I", "II"),
                                     ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6"))}

ST_ELEVATION_MV = 0.1
ST_ELEVATION_V2V3_MV = {"F": 0.15, "M": 0.2}   # men >= 40 y; age is not provided
ST_DEPRESSION_MV = 0.05
Q_MIN_MS, Q_MIN_MV = 30, 0.1
WIDE_QRS_MS, BORDERLINE_QRS_MS = 120, 110
SOKOLOW_MV = 3.5
CORNELL_MV = {"F": 2.0, "M": 2.8}
RVH_MV = 1.05
HR_LOW, HR_HIGH, PAUSE_S, PAUSE_RELEVANT_S = 50, 100, 1.5, 2.0
NAN = float("nan")
MAX_BRIDGED_GAP_S = 0.02   # gaps up to 20 ms are bridged for QRS measurement only
# Alert levels: "urgent" = clear, time-sensitive abnormality; "review" = any sign a person should
# look (including values close to a limit); "ok" = nothing found. False positives are preferred.
ALERT_ORDER = ("ok", "review", "urgent")
HR_URGENT_LOW, HR_URGENT_HIGH = 40, 150
HR_NEAR_BPM = 5            # rates within 5 bpm of a limit ask for review
AXIS_NEAR_DEG = 10         # axis within 10 degrees of a limit asks for review
VOLTAGE_NEAR = 0.9         # voltage sums above 90% of a limit ask for review
RR_CV_NEAR, RR_CV_IRREGULAR = 0.10, 0.15
UNVALIDATED = "rule not yet validated on photographed ECGs; confirm by eye"
MARK_PAD_S = 0.25
MARK_MAX_S = 4.0


class NoQrs(Exception):
    pass


class TooNoisy(Exception):
    pass


def _median(values):
    return statistics.median(values) if values else None


def _clean(samples):
    """None -> NaN so gaps are never bridged."""
    return [NAN if v is None else float(v) for v in samples]


def _bridge_short_gaps(x, max_len):
    """Linear fill of interior gaps up to max_len samples, for QRS measurement only.

    The extractor often loses a few pixels on the steep R stroke; without this, a 5 ms hole drops the whole beat.
    Coverage is still computed on the raw samples, so the gaps stay visible in the quality figures."""
    x, n, i = list(x), len(x), 0
    while i < n:
        if not math.isnan(x[i]):
            i += 1
            continue
        j = i
        while j < n and math.isnan(x[j]):
            j += 1
        if i > 0 and j < n and j - i <= max_len:
            a, b = x[i - 1], x[j]
            for t in range(i, j):
                x[t] = a + (b - a) * (t - i + 1) / (j - i + 1)
        i = j
    return x


def _has_gap(x, a, b):
    return any(math.isnan(v) for v in x[max(a, 0):b + 1])


def _baseline(x, fs):
    """Slow baseline (0.6 s median), used for beat detection only, never for ST."""
    n, half = len(x), max(int(0.3 * fs), 1)
    step = max(int(0.05 * fs), 1)
    anchors = []
    for i in range(0, n, step):
        seg = [v for v in x[max(i - half, 0):i + half + 1] if not math.isnan(v)]
        anchors.append((i, _median(seg) if seg else NAN))
    base = [NAN] * n
    for k, (i, v) in enumerate(anchors):
        j, nv = anchors[k + 1] if k + 1 < len(anchors) else (n, v)
        for t in range(i, min(j, n)):
            base[t] = v if (math.isnan(nv) or math.isnan(v)) else v + (nv - v) * (t - i) / max(j - i, 1)
    return base


def detect_beats(x, fs):
    """R peak indices (largest absolute deflection of each QRS) on one lead."""
    n = len(x)
    if n < int(0.8 * fs):
        return []
    base = _baseline(x, fs)
    xb = [v - b for v, b in zip(x, base)]
    d = [0.0] * n
    for i in range(1, n - 1):
        a, b = xb[i + 1], xb[i - 1]
        d[i] = 0.0 if (math.isnan(a) or math.isnan(b)) else abs(a - b)
    w = max(int(0.05 * fs), 1)
    env, acc = [0.0] * n, 0.0
    for i in range(n):
        acc += d[i] - (d[i - w] if i >= w else 0.0)
        env[i] = acc / w
    top = sorted(env)[int(0.98 * (n - 1))]
    if top <= 0:
        return []
    thr, refractory = 0.4 * top, int(0.25 * fs)
    peaks, i = [], 0
    while i < n:
        if env[i] < thr:
            i += 1
            continue
        j = i
        while j < n and env[j] >= thr:
            j += 1
        seg = [(abs(xb[t]), t) for t in range(max(i - w, 0), min(j + w, n)) if not math.isnan(xb[t])]
        if seg:
            p = max(seg)[1]
            if not peaks or p - peaks[-1] >= refractory:
                peaks.append(p)
            elif abs(xb[p]) > abs(xb[peaks[-1]]):
                peaks[-1] = p
        i = j
    if peaks:
        med = _median([abs(xb[p]) for p in peaks])
        peaks = [p for p in peaks if abs(xb[p]) >= 0.35 * med]
    # A beat closer than half the typical interval to its neighbour is a double detection
    # (typically a wide T wave next to the QRS): keep the steeper of the two, which is the QRS.
    if len(peaks) >= 4:
        typical = _median([b - a for a, b in zip(peaks, peaks[1:])])
        merged = [peaks[0]]
        for p in peaks[1:]:
            if p - merged[-1] < 0.5 * typical:
                if env[p] > env[merged[-1]]:
                    merged[-1] = p
            else:
                merged.append(p)
        peaks = merged
    return peaks


def _qrs_bounds(x, p, fs):
    """Onset/offset around R peak p: slope stays under 12% of the beat's max slope."""
    n, reach = len(x), int(0.14 * fs)
    lo, hi = max(p - reach, 1), min(p + reach, n - 2)
    if hi - lo < int(0.12 * fs) or _has_gap(x, lo - 1, hi + 1):
        return None
    slope = {i: abs(x[i + 1] - x[i - 1]) / 2 for i in range(lo, hi + 1)}
    core_lo, core_hi = max(p - int(0.06 * fs), lo), min(p + int(0.06 * fs), hi)
    core = max(slope[i] for i in range(core_lo, core_hi + 1))
    if core <= 0:
        return None
    thr, need = 0.12 * core, max(int(0.012 * fs), 2)

    def walk(step, bound):
        i, quiet = p, 0
        while lo <= i + step <= hi:
            i += step
            quiet = quiet + 1 if slope[i] < thr else 0
            if quiet >= need:
                return i - step * (need - 1)
        return None

    onset, offset = walk(-1, lo), walk(1, hi)
    if onset is None or offset is None or offset <= onset:
        return None
    return onset, offset


def _count_peaks(seg, prominence):
    """Positive local maxima of a QRS segment with enough prominence (rSR' counting)."""
    keep = []
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]:
            low = min(seg[keep[-1]:i + 1]) if keep else min(seg[:i + 1])
            if seg[i] - low >= prominence:
                keep.append(i)
    return len(keep)


def _longest_gap(x):
    run = best = 0
    for v in x:
        run = run + 1 if math.isnan(v) else 0
        best = max(best, run)
    return best


def _low_reason(i):
    """Plain-language cause of a low lead: what the extraction produced and what happened to each heartbeat."""
    cov = round(i["coverage"] * 100)
    trace = f"{cov}% of the trace extracted (needs 80%), longest hole {round(i['longest_gap_ms'])} ms"
    if i["coverage"] < 0.8:
        return trace + (f"; {i['found']} heartbeats found, {len(i['beats'])} usable" if i["found"] else "")
    if i["found"] == 0:
        return f"{trace}; no heartbeat detected"
    return (f"{trace}; {i['found']} heartbeats found, none usable "
            f"({i['dropped_gap']} cut by holes in the trace, {i['dropped_shape']} with no clear QRS shape)")


def analyse_lead(name, samples, fs, px=None):
    x = _clean(samples)
    valid = [v for v in x if not math.isnan(v)]
    info = {"name": name, "coverage": len(valid) / len(x) if x else 0, "beats": [], "raw": x, "px": px,
            "longest_gap_ms": 1000 * _longest_gap(x) / fs, "found": 0, "dropped_gap": 0, "dropped_shape": 0}
    if len(valid) < fs:
        return info
    x = _bridge_short_gaps(x, max(int(MAX_BRIDGED_GAP_S * fs), 1))
    info["x"] = x
    for p in detect_beats(x, fs):
        info["found"] += 1
        bounds = _qrs_bounds(x, p, fs)
        if not bounds:
            reach = int(0.14 * fs)
            lo, hi = max(p - reach, 1), min(p + reach, len(x) - 2)
            info["dropped_gap" if _has_gap(x, lo - 1, hi + 1) else "dropped_shape"] += 1
            continue
        on, off = bounds
        pre = [v for v in x[max(on - int(0.06 * fs), 0):max(on - int(0.02 * fs), 0) + 1] if not math.isnan(v)]
        seg = x[on:off + 1]
        if len(pre) < 2 or any(math.isnan(v) for v in seg):
            info["dropped_gap"] += 1
            continue
        ref = sum(pre) / len(pre)
        j60 = off + int(0.06 * fs)
        stv = [v for v in x[max(j60 - int(0.01 * fs), 0):j60 + int(0.01 * fs) + 1] if not math.isnan(v)]
        r, s = max(max(seg) - ref, 0.0), max(ref - min(seg), 0.0)
        rel = [v - ref for v in x[on:p + 1]]
        q_ms = q_mv = 0.0
        if x[p] >= ref and rel and min(rel) < -0.02:      # negative start before an upright R
            neg = 0
            for v in rel:
                if v < -0.02:
                    neg += 1
                elif neg:
                    break
            q_ms, q_mv = 1000 * neg / fs, -min(rel)
        info["beats"].append({
            "peak": p, "on": on, "off": off,
            "bridged": any(math.isnan(info["raw"][t]) for t in range(on, off + 1)),
            "dur_ms": 1000 * (off - on) / fs, "r": r, "s": s, "area": sum(v - ref for v in seg) / fs,
            "st": (sum(stv) / len(stv) - ref) if stv else None, "q_ms": q_ms, "q_mv": q_mv,
            "notches": _count_peaks([v - ref for v in seg], 0.2 * max(r, s, 0.05))})
    return info


def _beat_near_median(info, key, absolute=False):
    """The beat whose value is closest to the lead's median: the most representative one to show."""
    beats = [b for b in info["beats"] if b.get(key) is not None]
    if not beats:
        return None
    val = (lambda b: abs(b[key])) if absolute else (lambda b: b[key])
    med = _median([val(b) for b in beats])
    return min(beats, key=lambda b: abs(val(b) - med))


def _lead_px(lead):
    xs, ys = lead.get("x_px"), lead.get("y_px")
    if not xs or not ys or len(xs) != len(lead["samples"]) or len(ys) != len(xs):
        return None
    return xs, ys


def _photo_box(info, lo, hi):
    """Bounding box [x0, y0, x1, y1] in photo pixels of samples lo..hi, or None when positions are unknown."""
    px = info.get("px")
    if not px:
        return None
    xs = [v for v in px[0][lo:hi + 1] if v is not None]
    ys = [v for v in px[1][lo:hi + 1] if v is not None]
    if not xs or not ys:
        return None
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]


def _mark(info, fs, a, b, label):
    """Evidence mark: one window of the lead around samples a..b (gaps as None, bridged ranges listed)."""
    raw, x = info["raw"], info.get("x", info["raw"])
    pad = int(MARK_PAD_S * fs)
    lo, hi = max(a - pad, 0), min(b + pad, len(raw) - 1)
    if hi - lo > int(MARK_MAX_S * fs):
        mid = (a + b) // 2
        lo, hi = max(mid - int(MARK_MAX_S * fs / 2), 0), min(mid + int(MARK_MAX_S * fs / 2), len(raw) - 1)
    bridged, start = [], None
    for t in range(lo, hi + 1):
        filled = math.isnan(raw[t]) and not math.isnan(x[t])
        if filled and start is None:
            start = t
        if not filled and start is not None:
            bridged.append([start - lo, t - 1 - lo])
            start = None
    if start is not None:
        bridged.append([start - lo, hi - lo])
    return {"lead": info["name"], "sampling_rate_hz": fs, "label": label,
            "from_index": a - lo, "to_index": b - lo,
            "samples": [None if math.isnan(v) else round(v, 3) for v in raw[lo:hi + 1]],
            "bridged": bridged, "photo_box": _photo_box(info, lo, hi),
            "photo_box_focus": _photo_box(info, a, b)}


def _beat_mark(infos, fs, lead, key, label, absolute=False, window=None):
    info = infos.get(lead)
    beat = _beat_near_median(info, key, absolute) if info else None
    if not beat:
        return None
    on, off = (beat["on"], beat["off"]) if window is None else (beat["on"], beat["off"] + window)
    return _mark(info, fs, on, off, label)


def _beat_mark_by_key(infos, fs, lead, key, label):
    return _beat_mark(infos, fs, lead, key, label)


def _best_lead(leads, score):
    """Lead (among those scored, not None) with the highest score."""
    scored = [(score(n), n) for n in leads]
    scored = [(v, n) for v, n in scored if v is not None]
    return max(scored)[1] if scored else None


def _stat(info, key):
    values = [b[key] for b in info["beats"] if b.get(key) is not None]
    return _median(values) if values else None


def _contiguous(flagged):
    return sorted({n for pair in CONTIGUOUS if pair <= flagged for n in pair})


def _criterion(name, value, unit, limit, sex_specific, reference=None):
    return {"name": name, "value": value, "unit": unit, "reference": reference or f"≤ {limit} {unit}",
            "sex_specific": sex_specific,
            "status": "not_evaluated" if value is None else ("abnormal" if value > limit else "normal")}


def diagnose(sex, fs, leads):
    """leads: [{"name", "samples"}] -> (summary, details, warnings). Raises NoQrs / TooNoisy."""
    warnings, findings, criteria = [], [], []

    def find(code, label, severity, evidence, mark=None, marks=None, alert=None):
        # `mark` is the primary evidence window; `marks` lists every window the finding is based on.
        alert = alert or {"info": "ok", "warning": "review", "critical": "urgent"}[severity]
        findings.append({"code": code, "label": label, "severity": severity, "alert_level": alert, "evidence": evidence,
                         **({"mark": mark} if mark else {}), **({"marks": marks} if marks else {})})

    infos = {}
    for lead in leads:
        cand = analyse_lead(lead["name"], lead["samples"], fs, _lead_px(lead))
        if lead["name"] not in infos or len(cand["beats"]) > len(infos[lead["name"]]["beats"]):
            infos[lead["name"]] = cand

    # Rate and rhythm: longest strip (lead II preferred on ties).
    longest = max(leads, key=lambda l: (sum(v is not None for v in l["samples"]), l["name"] == "II"))
    raw = _clean(longest["samples"])
    x = _bridge_short_gaps(raw, max(int(MAX_BRIDGED_GAP_S * fs), 1))
    peaks = detect_beats(x, fs)
    spans = [(a, b) for a, b in zip(peaks, peaks[1:]) if not _has_gap(x, a, b)]
    rr = [(b - a) / fs for a, b in spans]
    rinfo = {"name": longest["name"], "raw": raw, "x": x, "px": _lead_px(longest)}
    if not rr:
        raise NoQrs()
    hr = round(60 / _median(rr))
    if hr > 300 or hr < 15:
        raise TooNoisy()
    cv = statistics.pstdev(rr) / (sum(rr) / len(rr)) if len(rr) > 1 else 0.0
    rhythm_conf = "high" if len(rr) >= 4 else "medium"
    if len(rr) >= 4 and cv > RR_CV_IRREGULAR:
        rhythm = "Irregular rhythm"
        find("IRREGULAR_RHYTHM", "Irregular rhythm (P waves not assessed)", "warning",
             f"RR variation {cv * 100:.0f}% over {len(rr)} intervals; possible atrial fibrillation or ectopy",
             _mark(rinfo, fs, *spans[max(range(len(rr)), key=lambda k: abs(rr[k] - _median(rr)))], "Most irregular interval"))
    else:
        rhythm = "Regular rhythm"
        if len(rr) >= 4 and cv > RR_CV_NEAR:
            find("RR_VARIATION_BORDERLINE", "Borderline RR variation", "info",
                 f"RR variation {cv * 100:.0f}% over {len(rr)} intervals (irregular above {RR_CV_IRREGULAR * 100:.0f}%)",
                 _mark(rinfo, fs, *spans[max(range(len(rr)), key=lambda k: abs(rr[k] - _median(rr)))], "Most irregular interval"),
                 alert="review")
    k = min(range(len(rr)), key=lambda k: abs(rr[k] - _median(rr)))
    typical = lambda: _mark(rinfo, fs, *spans[k], "Typical beat interval")
    if hr < HR_LOW:
        marked = hr < HR_URGENT_LOW
        find("BRADYCARDIA", "Low heart rate" + (" (marked)" if marked else ""), "critical" if marked else "warning",
             f"{hr} bpm (< {HR_LOW})", typical())
    elif hr >= HR_HIGH:
        marked = hr >= HR_URGENT_HIGH
        find("TACHYCARDIA", "High heart rate" + (" (marked)" if marked else ""), "critical" if marked else "warning",
             f"{hr} bpm (≥ {HR_HIGH})", typical())
    elif hr < HR_LOW + HR_NEAR_BPM:
        find("HEART_RATE_NEAR_LOW", "Heart rate close to the bradycardia limit", "info",
             f"{hr} bpm (bradycardia < {HR_LOW})", typical(), alert="review")
    elif hr >= HR_HIGH - HR_NEAR_BPM:
        find("HEART_RATE_NEAR_HIGH", "Heart rate close to the tachycardia limit", "info",
             f"{hr} bpm (tachycardia ≥ {HR_HIGH})", typical(), alert="review")
    if max(rr) > PAUSE_S:
        relevant = max(rr) > PAUSE_RELEVANT_S
        find("PAUSE", "Pause", "critical" if relevant else "warning",
             f"longest RR {max(rr):.1f} s (> {PAUSE_S} s)" + ("; clinically relevant (> 2 s)" if relevant else ""),
             _mark(rinfo, fs, *spans[rr.index(max(rr))], "Longest pause"))

    # QRS duration and bundle branch block .
    durations = sorted(d for d in (_stat(i, "dur_ms") for i in infos.values()) if d is not None)
    qrs = round(durations[min(int(0.75 * len(durations)), len(durations) - 1)]) if durations else None
    qrs_conf = "medium" if fs >= 200 and len(durations) >= 6 else "low"
    net = {n: _stat(i, "area") for n, i in infos.items()}
    r_amp = {n: _stat(i, "r") for n, i in infos.items()}
    s_amp = {n: _stat(i, "s") for n, i in infos.items()}

    def notched(lead):
        return lead in infos and max((b["notches"] for b in infos[lead]["beats"]), default=0) >= 2

    def septal_q(lead):
        return lead in infos and (_stat(infos[lead], "q_ms") or 0) >= Q_MIN_MS

    qrs_lead = _best_lead(infos, lambda n: (-abs(_stat(infos[n], "dur_ms") - qrs) if qrs is not None and _stat(infos[n], "dur_ms") is not None else None))
    qrs_mark = lambda: _beat_mark(infos, fs, qrs_lead, "dur_ms", "QRS complex")
    if qrs is not None:
        criteria.append(_criterion("QRS duration", qrs, "ms", WIDE_QRS_MS - 1, False, f"< {WIDE_QRS_MS} ms"))
        if qrs >= WIDE_QRS_MS:
            wide_s = lambda l: (s_amp.get(l) or 0) > 0.5 * (r_amp.get(l) or 1e9)
            rbbb = (notched("V1") or notched("V2")) and (wide_s("I") or wide_s("V6"))
            lbbb = (net.get("V1") is not None and net["V1"] < 0
                    and any((net.get(l) or 0) > 0 for l in ("I", "V6")) and not (septal_q("I") or septal_q("V6")))
            if rbbb and not lbbb:
                find("RBBB_PATTERN", "Possible right bundle branch block pattern", "warning",
                     f"QRS {qrs} ms (≥ 120) with rSR'-type pattern in V1/V2 and wide S in I/V6; {UNVALIDATED}", qrs_mark())
            elif lbbb and not rbbb:
                find("LBBB_PATTERN", "Possible left bundle branch block pattern", "warning",
                     f"QRS {qrs} ms (≥ 120), negative V1, positive I/V6 without septal q; {UNVALIDATED}", qrs_mark())
            else:
                find("WIDE_QRS", "Wide QRS (intraventricular conduction delay)", "warning", f"QRS {qrs} ms (≥ 120)", qrs_mark())
            if hr >= HR_HIGH:
                find("WIDE_QRS_TACHYCARDIA", "Wide-complex tachycardia", "critical",
                     f"QRS {qrs} ms (≥ {WIDE_QRS_MS}) at {hr} bpm (≥ {HR_HIGH})", qrs_mark())
        elif qrs >= BORDERLINE_QRS_MS:
            find("QRS_BORDERLINE", "Borderline QRS duration", "info", f"QRS {qrs} ms (110–119 ms range)", qrs_mark(), alert="review")

    # Axis from the net QRS area in I and aVF.
    axis = None
    if net.get("I") is not None and net.get("aVF") is not None and (net["I"] or net["aVF"]):
        axis = round(math.degrees(math.atan2(net["aVF"], net["I"])))
        axis_lead = "I" if abs(net["I"]) >= abs(net["aVF"]) else "aVF"
        axis_mark = lambda: _beat_mark(infos, fs, axis_lead, "area", f"QRS in lead {axis_lead}", True)
        axis_detail = f"net QRS area I {net['I'] * 1000:+.0f} mV·ms, aVF {net['aVF'] * 1000:+.0f} mV·ms"
        axis_marks = lambda: [m for m in (
            _beat_mark(infos, fs, n, "area", f"QRS in lead {n}: net area {net[n] * 1000:+.0f} mV·ms", True)
            for n in ("I", "aVF")) if m] or None
        if -30 <= axis <= 90:
            near = axis < -30 + AXIS_NEAR_DEG or axis > 90 - AXIS_NEAR_DEG
            find("NORMAL_AXIS", "Normal electrical axis" + (", close to the limit" if near else ""), "info",
                 f"axis {axis}° (normal −30° to +90°); {axis_detail}", axis_mark(), axis_marks(), alert="review" if near else None)
        elif -90 <= axis < -30:
            find("LEFT_AXIS_DEVIATION", "Left axis deviation", "warning", f"axis {axis}° (< −30°); {axis_detail}", axis_mark(), axis_marks())
        elif 90 < axis <= 180:
            find("RIGHT_AXIS_DEVIATION", "Right axis deviation", "warning", f"axis {axis}° (> +90°); {axis_detail}", axis_mark(), axis_marks())
        else:
            find("EXTREME_AXIS", "Extreme axis deviation", "warning", f"axis {axis}° (−90° to −180°); {axis_detail}", axis_mark(), axis_marks())

    # Voltage criteria .
    sok = cor = None
    if s_amp.get("V1") is not None and (r_amp.get("V5") is not None or r_amp.get("V6") is not None):
        sok = round(s_amp["V1"] + max(r_amp.get("V5") or 0, r_amp.get("V6") or 0), 2)
    if r_amp.get("aVL") is not None and s_amp.get("V3") is not None:
        cor = round(r_amp["aVL"] + s_amp["V3"], 2)
    criteria.append(_criterion("Sokolow-Lyon (S V1 + R V5/V6)", sok, "mV", SOKOLOW_MV, False))
    criteria.append(_criterion("Cornell (R aVL + S V3)", cor, "mV", CORNELL_MV[sex], True,
                               f"≤ {CORNELL_MV[sex]} mV ({'female' if sex == 'F' else 'male'})"))
    lvh = [f"{n} {v} mV" for n, v, lim in (("Sokolow-Lyon", sok, SOKOLOW_MV), ("Cornell", cor, CORNELL_MV[sex]))
           if v is not None and v > lim]
    if lvh:
        cands = ([("V1", "s"), ("V5", "r"), ("V6", "r")] if sok is not None and sok > SOKOLOW_MV
                 else [("aVL", "r"), ("V3", "s")])
        amp = {"r": r_amp, "s": s_amp}
        lvh_lead, lvh_key = max(((n, k) for n, k in cands if amp[k].get(n) is not None), key=lambda c: amp[c[1]][c[0]])
        find("LVH_VOLTAGE", "Voltage criteria for left ventricular hypertrophy", "warning", "; ".join(lvh),
             _beat_mark_by_key(infos, fs, lvh_lead, lvh_key, f"Tallest {lvh_key.upper()} wave in {lvh_lead}"))
    else:
        near = [f"{n} {v} mV (limit {lim} mV)" for n, v, lim in (("Sokolow-Lyon", sok, SOKOLOW_MV), ("Cornell", cor, CORNELL_MV[sex]))
                if v is not None and v > VOLTAGE_NEAR * lim]
        if near:
            find("LVH_VOLTAGE_BORDERLINE", "Voltage close to the left ventricular hypertrophy limit", "info",
                 "; ".join(near), alert="review")
    if r_amp.get("V1") is not None and (s_amp.get("V5") is not None or s_amp.get("V6") is not None):
        rv = round(r_amp["V1"] + max(s_amp.get("V5") or 0, s_amp.get("V6") or 0), 2)
        if rv > RVH_MV and axis is not None and axis > 90:
            find("RVH_PATTERN", "Pattern suggestive of right ventricular hypertrophy", "warning",
                 f"R V1 + S V5/V6 {rv} mV (> {RVH_MV}) with right axis",
                 _beat_mark_by_key(infos, fs, "V1", "r", "R wave in V1"))

    # ST segment and Q waves : two contiguous leads, aVR excluded.
    st = {n: _stat(i, "st") for n, i in infos.items() if i["beats"] and n != "aVR"}
    elev = {n for n, v in st.items() if v is not None and v >= (ST_ELEVATION_V2V3_MV[sex] if n in ("V2", "V3") else ST_ELEVATION_MV)}
    dep = {n for n, v in st.items() if v is not None and v <= -ST_DEPRESSION_MV}
    q = {n for n, i in infos.items() if n != "aVR" and (_stat(i, "q_ms") or 0) >= Q_MIN_MS and (_stat(i, "q_mv") or 0) >= Q_MIN_MV}
    # ST/Q rules are not validated on photographed traces yet: reported as "possible" with a review alert.
    if _contiguous(elev):
        lead = _best_lead(_contiguous(elev), lambda n: st.get(n))
        find("ST_ELEVATION", "Possible ST elevation in contiguous leads", "warning",
             "leads " + ", ".join(_contiguous(elev)) + f"; {UNVALIDATED}",
             _beat_mark(infos, fs, lead, "st", f"ST segment in {lead}", window=int(0.2 * fs)))
    if _contiguous(dep):
        lead = _best_lead(_contiguous(dep), lambda n: -st[n] if n in st else None)
        find("ST_DEPRESSION", "Possible ST depression in contiguous leads", "warning",
             "leads " + ", ".join(_contiguous(dep)) + f"; {UNVALIDATED}",
             _beat_mark(infos, fs, lead, "st", f"ST segment in {lead}", window=int(0.2 * fs)))
    if _contiguous(q):
        lead = _best_lead(_contiguous(q), lambda n: _stat(infos[n], "q_mv"))
        find("PATHOLOGICAL_Q", "Possible pathological Q waves in contiguous leads", "warning",
             "leads " + ", ".join(_contiguous(q)) + f"; {UNVALIDATED}",
             _beat_mark(infos, fs, lead, "q_mv", f"Q wave in {lead}"))

    per_lead = []
    for name in LEADS:
        if name not in infos:
            continue
        i = infos[name]
        ok = i["coverage"] >= 0.8 and len(i["beats"]) >= 1
        reason = None if ok else _low_reason(i)
        if not ok:
            warnings.append({"code": "LEAD_LOW_QUALITY", "lead": name})
        per_lead.append({"name": name, "r_amplitude_mv": None if r_amp.get(name) is None else round(r_amp[name], 3),
                         "s_amplitude_mv": None if s_amp.get(name) is None else round(s_amp[name], 3),
                         "quality": "good" if ok else "low", "quality_reason": reason,
                         "estimated": any(b["bridged"] for b in i["beats"])})

    abnormal = [f for f in findings if f["severity"] != "info"]
    partial = qrs_conf == "low" or len(per_lead) < 12
    classification = "abnormal" if abnormal else ("borderline" if partial else "normal")
    headline = {"normal": "No abnormal pattern found in the criteria that could be checked",
                "borderline": "No abnormal pattern found in the parts of the ECG that could be read",
                "abnormal": "Possible abnormalities found; clinical review needed"}[classification]
    reasons = [f["label"] for f in findings if f["alert_level"] != "ok"]
    if partial:
        reasons.append("Part of the ECG could not be read reliably")
    alert = max([f["alert_level"] for f in findings] + ["review" if partial else "ok"], key=ALERT_ORDER.index)
    alert_text = {"ok": "No sign of a problem in the criteria that could be checked",
                  "review": "Possible problem: a person should review this ECG",
                  "urgent": "Clear abnormality: review this ECG urgently"}[alert]
    summary = {"classification": classification, "headline": headline, "rhythm": rhythm, "heart_rate_bpm": hr,
               "alert_level": alert, "alert_text": alert_text, "alert_reasons": reasons,
               "key_findings": [f["label"] for f in (abnormal or findings[:2])],
               "text": f"{rhythm}, heart rate {hr} bpm" + (f", QRS {qrs} ms" if qrs else "") + f". {headline}."}
    measurements = {
        "heart_rate_bpm": {"value": hr, "unit": "bpm", "confidence": rhythm_conf, "lead": longest["name"]},
        "rr_ms": {"value": round(_median(rr) * 1000), "unit": "ms", "confidence": rhythm_conf},
        "pr_ms": {"value": None, "unit": "ms", "confidence": "low"},
        "qrs_ms": {"value": qrs, "unit": "ms", "confidence": qrs_conf},
        "qt_ms": {"value": None, "unit": "ms", "confidence": "low"},
        "qtc_ms": {"value": None, "unit": "ms", "confidence": "low", "formula": "Bazett"},
        "qrs_axis_deg": {"value": axis, "unit": "deg", "confidence": "medium" if axis is not None else "low"},
    }
    return summary, {"measurements": measurements, "findings": findings, "criteria": criteria, "per_lead": per_lead}, warnings
