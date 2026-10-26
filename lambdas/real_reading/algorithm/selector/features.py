"""Signal-quality features for extracted ECG leads (input: mV array at FS Hz, NaN = missing)."""
import numpy as np
from scipy.signal import find_peaks, butter, sosfiltfilt

FS = 150.0
FEATURE_NAMES = [
    "coverage", "n_gaps", "longest_gap_s", "std", "range", "roughness", "max_jump", "flat_frac",
    "spike_frac", "clip_frac", "n_peaks", "hr", "rr_cv", "peak_cv", "beat_corr", "lowf_frac",
    "midf_frac", "highf_frac", "acf_peak", "slope", "absmean", "skew", "kurt", "stair_frac",
    "max_corr_other", "einthoven_res", "aVR_res", "med_std_ratio",
]


def _interp(x):
    ok = np.isfinite(x)
    if ok.sum() < 4:
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def lead_features(x):
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    ok = np.isfinite(x)
    f["coverage"] = ok.mean()
    d = np.diff(ok.astype(int))
    f["n_gaps"] = int((d == -1).sum())
    run = best = 0
    for v in ok:
        run = 0 if v else run + 1
        best = max(best, run)
    f["longest_gap_s"] = best / FS
    y = _interp(x)
    if y is None or len(y) < 40:
        return f
    y0 = y - np.median(y)
    s = y0.std() + 1e-9
    f["std"] = s
    f["range"] = np.ptp(y0)
    dy = np.diff(y0)
    f["roughness"] = np.abs(dy).mean() / s
    f["max_jump"] = np.abs(dy).max() / s
    f["flat_frac"] = (np.abs(dy) < 1e-4).mean()
    mad = np.median(np.abs(dy - np.median(dy))) + 1e-6
    f["spike_frac"] = (np.abs(dy) > 12 * mad).mean()
    f["clip_frac"] = max((y0 >= y0.max() - 1e-6).mean(), (y0 <= y0.min() + 1e-6).mean())
    sos = butter(2, [0.5, 30], "bp", fs=FS, output="sos")
    z = sosfiltfilt(sos, y0)
    zs = z.std() + 1e-9
    pk, pr = find_peaks(np.abs(z), height=2.0 * zs, distance=int(0.25 * FS))
    f["n_peaks"] = len(pk)
    if len(pk) >= 3:
        rr = np.diff(pk) / FS
        f["hr"] = 60.0 / rr.mean()
        f["rr_cv"] = rr.std() / rr.mean()
        h = pr["peak_heights"]
        f["peak_cv"] = h.std() / h.mean()
        w = int(0.25 * FS)
        beats = [z[p - w:p + w] for p in pk if p - w >= 0 and p + w <= len(z)]
        if len(beats) >= 3:
            B = np.array(beats)
            B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-9)
            C = B @ B.T / B.shape[1]
            f["beat_corr"] = np.median(C[np.triu_indices(len(B), 1)])
    F = np.abs(np.fft.rfft(y0 * np.hanning(len(y0)))) ** 2
    fr = np.fft.rfftfreq(len(y0), 1 / FS)
    tot = F.sum() + 1e-12
    f["lowf_frac"] = F[fr < 0.5].sum() / tot
    f["midf_frac"] = F[(fr >= 0.5) & (fr < 40)].sum() / tot
    f["highf_frac"] = F[fr >= 40].sum() / tot
    a = np.correlate(z, z, "full")[len(z) - 1:]
    a = a / (a[0] + 1e-12)
    lo, hi = int(0.3 * FS), min(int(2.0 * FS), len(a) - 1)
    f["acf_peak"] = a[lo:hi].max() if hi > lo else 0.0
    f["slope"] = abs(np.polyfit(np.arange(len(y0)) / FS, y0, 1)[0]) / s
    f["absmean"] = np.abs(y0).mean()
    f["skew"] = float(((y0 / s) ** 3).mean())
    f["kurt"] = float(((y0 / s) ** 4).mean())
    # staircase: share of steps with exactly repeated value after quantisation
    q = np.round(dy / (0.002)).astype(int)
    f["stair_frac"] = (q == 0).mean()
    return f


def record_features(leads):
    """leads: dict name -> array. Returns dict name -> feature vector (list)."""
    base = {k: lead_features(v) for k, v in leads.items()}
    ys = {k: _interp(v) for k, v in leads.items()}
    names = list(leads)
    n = min(len(v) for v in leads.values())

    def get(k):
        v = ys.get(k)
        return None if v is None else v[:n] - np.median(v[:n])

    I, II, III, aVR, aVL, aVF = (get(k) for k in ["I", "II", "III", "aVR", "aVL", "aVF"])
    eint = avr = 0.0
    if I is not None and II is not None and III is not None:
        eint = np.sqrt(np.mean((II - I - III) ** 2)) / (np.sqrt(np.mean(II ** 2)) + 1e-9)
    if I is not None and II is not None and aVR is not None:
        avr = np.sqrt(np.mean((aVR + (I + II) / 2) ** 2)) / (np.sqrt(np.mean(aVR ** 2)) + 1e-9)
    stds = np.array([base[k]["std"] for k in names])
    med = np.median(stds) + 1e-9
    out = {}
    for k in names:
        b = base[k]
        yk = get(k)
        mc = 0.0
        if yk is not None:
            cs = []
            for o in names:
                yo = get(o)
                if o != k and yo is not None and yk.std() > 1e-6 and yo.std() > 1e-6:
                    cs.append(abs(np.corrcoef(yk, yo)[0, 1]))
            mc = max(cs) if cs else 0.0
        b = dict(b, max_corr_other=mc, einthoven_res=eint, aVR_res=avr, med_std_ratio=b["std"] / med)
        out[k] = [b[n_] for n_ in FEATURE_NAMES]
    return out
