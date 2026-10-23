"""Rule-based ECG diagnoser (screening aid, not a medical report).

Works on the extracted signal (mV, native sampling rate, gaps as None) using published
interpretation criteria that can be checked on an extracted trace: QRS duration, voltage
criteria for hypertrophy, axis and the rate/rhythm thresholds. ST, Q-wave and bundle
branch findings are computed but only reported when EXPERIMENTAL is set. Anything that
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
HR_LOW, HR_HIGH, PAUSE_S = 50, 100, 2.0
NAN = float("nan")
EXPERIMENTAL = False


class NoQrs(Exception):
    pass


class TooNoisy(Exception):
    pass


def _median(values):
    return statistics.median(values) if values else None


def _clean(samples):
    """None -> NaN so gaps are never bridged."""
    return [NAN if v is None else float(v) for v in samples]


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


def analyse_lead(name, samples, fs):
    x = _clean(samples)
    valid = [v for v in x if not math.isnan(v)]
    info = {"name": name, "coverage": len(valid) / len(x) if x else 0, "beats": []}
    if len(valid) < fs:
        return info
    for p in detect_beats(x, fs):
        bounds = _qrs_bounds(x, p, fs)
        if not bounds:
            continue
        on, off = bounds
        pre = [v for v in x[max(on - int(0.06 * fs), 0):max(on - int(0.02 * fs), 0) + 1] if not math.isnan(v)]
        seg = x[on:off + 1]
        if len(pre) < 2 or any(math.isnan(v) for v in seg):
            continue
        ref = sum(pre) / len(pre)
        j60 = off + int(0.06 * fs)
        stv = [v for v in x[max(j60 - int(0.01 * fs), 0):j60 + int(0.01 * fs) + 1] if not math.isnan(v)]
        r, s = max(seg) - ref, ref - min(seg)
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
            "dur_ms": 1000 * (off - on) / fs, "r": r, "s": s, "area": sum(v - ref for v in seg) / fs,
            "st": (sum(stv) / len(stv) - ref) if stv else None, "q_ms": q_ms, "q_mv": q_mv,
            "notches": _count_peaks([v - ref for v in seg], 0.2 * max(r, s, 0.05))})
    return info


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

    def find(code, label, severity, evidence):
        findings.append({"code": code, "label": label, "severity": severity, "evidence": evidence})

    infos = {}
    for lead in leads:
        cand = analyse_lead(lead["name"], lead["samples"], fs)
        if lead["name"] not in infos or len(cand["beats"]) > len(infos[lead["name"]]["beats"]):
            infos[lead["name"]] = cand

    # Rate and rhythm: longest strip (lead II preferred on ties).
    longest = max(leads, key=lambda l: (sum(v is not None for v in l["samples"]), l["name"] == "II"))
    x = _clean(longest["samples"])
    peaks = detect_beats(x, fs)
    rr = [(b - a) / fs for a, b in zip(peaks, peaks[1:]) if not _has_gap(x, a, b)]
    if not rr:
        raise NoQrs()
    hr = round(60 / _median(rr))
    if hr > 300 or hr < 15:
        raise TooNoisy()
    cv = statistics.pstdev(rr) / (sum(rr) / len(rr)) if len(rr) > 1 else 0.0
    rhythm_conf = "high" if len(rr) >= 4 else "medium"
    if len(rr) >= 4 and cv > 0.15:
        rhythm = "Irregular rhythm"
        find("IRREGULAR_RHYTHM", "Irregular rhythm (P waves not assessed)", "warning",
             f"RR variation {cv * 100:.0f}% over {len(rr)} intervals; possible atrial fibrillation or ectopy")
    else:
        rhythm = "Regular rhythm"
    if hr < HR_LOW:
        find("BRADYCARDIA", "Low heart rate", "warning", f"{hr} bpm (< {HR_LOW})")
    elif hr >= HR_HIGH:
        find("TACHYCARDIA", "High heart rate", "warning", f"{hr} bpm (≥ {HR_HIGH})")
    if max(rr) > PAUSE_S:
        find("PAUSE", "Pause", "warning", f"longest RR {max(rr):.1f} s (> {PAUSE_S:.0f} s)")

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

    if qrs is not None:
        criteria.append(_criterion("QRS duration", qrs, "ms", WIDE_QRS_MS - 1, False, f"< {WIDE_QRS_MS} ms"))
        if qrs >= WIDE_QRS_MS:
            wide_s = lambda l: (s_amp.get(l) or 0) > 0.5 * (r_amp.get(l) or 1e9)
            rbbb = (notched("V1") or notched("V2")) and (wide_s("I") or wide_s("V6"))
            lbbb = (net.get("V1") is not None and net["V1"] < 0
                    and any((net.get(l) or 0) > 0 for l in ("I", "V6")) and not (septal_q("I") or septal_q("V6")))
            if EXPERIMENTAL and rbbb and not lbbb:
                find("RBBB_PATTERN", "Pattern suggestive of right bundle branch block", "warning",
                     f"QRS {qrs} ms (≥ 120) with rSR'-type pattern in V1/V2 and wide S in I/V6")
            elif EXPERIMENTAL and lbbb and not rbbb:
                find("LBBB_PATTERN", "Pattern suggestive of left bundle branch block", "warning",
                     f"QRS {qrs} ms (≥ 120), negative V1, positive I/V6 without septal q")
            else:
                find("WIDE_QRS", "Wide QRS (intraventricular conduction delay)", "warning", f"QRS {qrs} ms (≥ 120)")
        elif qrs >= BORDERLINE_QRS_MS:
            find("QRS_BORDERLINE", "Borderline QRS duration", "info", f"QRS {qrs} ms (110–119 ms range)")

    # Axis from the net QRS area in I and aVF.
    axis = None
    if net.get("I") is not None and net.get("aVF") is not None and (net["I"] or net["aVF"]):
        axis = round(math.degrees(math.atan2(net["aVF"], net["I"])))
        if -30 <= axis <= 90:
            find("NORMAL_AXIS", "Normal electrical axis", "info", f"axis {axis}° (−30° to +90°)")
        elif -90 <= axis < -30:
            find("LEFT_AXIS_DEVIATION", "Left axis deviation", "warning", f"axis {axis}° (< −30°)")
        elif 90 < axis <= 180:
            find("RIGHT_AXIS_DEVIATION", "Right axis deviation", "warning", f"axis {axis}° (> +90°)")
        else:
            find("EXTREME_AXIS", "Extreme axis deviation", "warning", f"axis {axis}° (−90° to −180°)")

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
        find("LVH_VOLTAGE", "Voltage criteria for left ventricular hypertrophy", "warning", "; ".join(lvh))
    if r_amp.get("V1") is not None and (s_amp.get("V5") is not None or s_amp.get("V6") is not None):
        rv = round(r_amp["V1"] + max(s_amp.get("V5") or 0, s_amp.get("V6") or 0), 2)
        if rv > RVH_MV and axis is not None and axis > 90:
            find("RVH_PATTERN", "Pattern suggestive of right ventricular hypertrophy", "warning",
                 f"R V1 + S V5/V6 {rv} mV (> {RVH_MV}) with right axis")

    # ST segment and Q waves : two contiguous leads, aVR excluded.
    st = {n: _stat(i, "st") for n, i in infos.items() if i["beats"] and n != "aVR"}
    elev = {n for n, v in st.items() if v is not None and v >= (ST_ELEVATION_V2V3_MV[sex] if n in ("V2", "V3") else ST_ELEVATION_MV)}
    dep = {n for n, v in st.items() if v is not None and v <= -ST_DEPRESSION_MV}
    q = {n for n, i in infos.items() if n != "aVR" and (_stat(i, "q_ms") or 0) >= Q_MIN_MS and (_stat(i, "q_mv") or 0) >= Q_MIN_MV}
    # ST/Q/BBB findings stay off in production until validated (EXPERIMENTAL is only set by the evaluation scripts).
    if EXPERIMENTAL:
        if _contiguous(elev):
            find("ST_ELEVATION", "Possible ST elevation in contiguous leads", "critical",
                 "leads " + ", ".join(_contiguous(elev)))
        if _contiguous(dep):
            find("ST_DEPRESSION", "Possible ST depression in contiguous leads", "warning",
                 "leads " + ", ".join(_contiguous(dep)))
        if _contiguous(q):
            find("PATHOLOGICAL_Q", "Possible pathological Q waves in contiguous leads", "warning",
                 "leads " + ", ".join(_contiguous(q)))

    warnings.append({"code": "MEASUREMENT_UNRELIABLE",
                     "message": "PR, QT, ST and Q-wave findings are not reported: they cannot be located reliably on an extracted trace."})
    per_lead = []
    for name in LEADS:
        if name not in infos:
            continue
        i = infos[name]
        ok = i["coverage"] >= 0.8 and len(i["beats"]) >= 1
        if not ok:
            warnings.append({"code": "LEAD_LOW_QUALITY", "lead": name})
        per_lead.append({"name": name, "r_amplitude_mv": None if r_amp.get(name) is None else round(r_amp[name], 3),
                         "s_amplitude_mv": None if s_amp.get(name) is None else round(s_amp[name], 3),
                         "quality": "good" if ok else "low"})

    abnormal = [f for f in findings if f["severity"] != "info"]
    partial = qrs_conf == "low" or len(per_lead) < 12
    classification = "abnormal" if abnormal else ("borderline" if partial else "normal")
    headline = {"normal": "No abnormal pattern found in the criteria that could be checked",
                "borderline": "No abnormal pattern found, but only part of the ECG could be checked",
                "abnormal": "Possible abnormalities found; clinical review needed"}[classification]
    summary = {"classification": classification, "headline": headline, "rhythm": rhythm, "heart_rate_bpm": hr,
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
