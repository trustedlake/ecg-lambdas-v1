"""Ground-truth loading and synthetic extraction-error simulation."""
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from features import FS

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _header(path):
    lines = Path(path).read_text().splitlines()
    first = lines[0].split()
    fs, n = float(first[2]), int(first[3])
    gains, names = [], []
    for ln in lines[1:13]:
        p = ln.split()
        gains.append(float(re.match(r"[\d.]+", p[2]).group()))
        names.append(p[-1])
    return fs, n, gains, names


def _to_fs(sig, fs):
    up, down = int(round(FS * 2)), int(round(fs * 2))
    return resample_poly(sig, up, down, axis=1)


def load_record(hea):
    hea = Path(hea)
    fs, n, gains, names = _header(hea)
    dat = hea.with_suffix(".dat")
    if dat.exists():
        raw = np.fromfile(dat, dtype="<i2").reshape(n, 12).T.astype(float)
    else:
        raw = np.asarray(loadmat(hea.with_suffix(".mat"))["val"], float)[:12]
    sig = raw / np.array(gains)[:, None]
    sig = _to_fs(sig, fs)
    return {nm: sig[i] for i, nm in enumerate(names) if nm in LEADS}


def load_all(root):
    recs = []
    for hea in sorted(Path(root).rglob("*.hea")):
        try:
            r = load_record(hea)
            if len(r) == 12 and np.isfinite(list(r.values())).all():
                recs.append((hea.stem, r))
        except Exception as e:  # skip unreadable records
            print("skip", hea, e)
    return recs


def _gap(x, rng, sev):
    n = len(x)
    for _ in range(rng.integers(1, 4)):
        L = int(n * sev * rng.uniform(0.05, 0.4))
        s = rng.integers(0, max(1, n - L))
        x[s:s + L] = np.nan
    return x


def _flat(x, rng, sev):
    n = len(x)
    L = int(n * sev * rng.uniform(0.1, 0.6)) + 5
    s = rng.integers(0, max(1, n - L))
    x[s:s + L] = x[s]
    return x


def _spikes(x, rng, sev):
    k = max(1, int(sev * 25))
    idx = rng.integers(0, len(x), k)
    amp = np.nanstd(x) * rng.uniform(2, 8, k) * rng.choice([-1, 1], k)
    x[idx] += amp
    return x


def _clip(x, rng, sev):
    c = np.nanmedian(x)
    lim = (1 - 0.8 * sev) * max(np.nanmax(x) - c, c - np.nanmin(x))
    return np.clip(x, c - lim, c + lim)


def _gain(x, rng, sev):
    g = np.exp(rng.choice([-1, 1]) * sev * rng.uniform(0.2, 0.9))
    return x * g


def _drift(x, rng, sev):
    t = np.linspace(0, 1, len(x))
    return x + sev * np.nanstd(x) * 3 * np.sin(2 * np.pi * t * rng.uniform(0.2, 1.5) + rng.uniform(0, 6))


def _noise(x, rng, sev):
    return x + rng.normal(0, sev * np.nanstd(x) * 0.8, len(x))


def _warp(x, rng, sev):
    n = len(x)
    t = np.arange(n) + sev * 30 * np.sin(np.arange(n) / n * 2 * np.pi * rng.uniform(1, 4))
    ok = np.isfinite(x)
    return np.interp(t, np.arange(n)[ok], x[ok])


def _grid(x, rng, sev):
    n = len(x)
    p = rng.uniform(5, 20)
    sq = np.sign(np.sin(2 * np.pi * np.arange(n) / p))
    return x + sev * np.nanstd(x) * 1.5 * sq


def _smooth(x, rng, sev):
    k = int(3 + sev * 25)
    ok = np.isfinite(x)
    y = np.interp(np.arange(len(x)), np.arange(len(x))[ok], x[ok])
    return np.convolve(y, np.ones(k) / k, mode="same")


CORRUPTIONS = [_gap, _flat, _spikes, _clip, _gain, _drift, _noise, _warp, _grid, _smooth]


def _fake_lead(x, rng, other):
    n = len(x)
    kind = rng.integers(0, 6)
    sd = np.nanstd(x)
    if kind == 0:
        return np.full(n, np.nanmedian(x)) + rng.normal(0, 1e-4, n)
    if kind == 1:
        return np.cumsum(rng.normal(0, sd * 0.1, n))
    if kind == 2:
        return sd * np.sin(2 * np.pi * np.arange(n) / FS * rng.uniform(0.5, 3))
    if kind == 3:
        return rng.normal(0, sd, n)
    if kind == 4:
        return np.roll(other, rng.integers(0, n))
    return -x


def simulate_extraction(truth, rng):
    """Mild extractor-like degradation of a truth lead (quantisation, jitter, tiny gaps)."""
    q = rng.uniform(0.004, 0.04)
    x = np.round(truth / q) * q + rng.normal(0, 0.004, len(truth))
    return x


def make_variant(rec, rng):
    """Returns (dict lead->corrupted array, dict lead->quality 0..100)."""
    truth = {k: v.copy() for k, v in rec.items()}
    mode = rng.random()
    out = {k: simulate_extraction(v, rng) for k, v in truth.items()}
    n_bad = 0 if mode < 0.3 else int(rng.integers(1, 13))
    bad = rng.choice(list(truth), n_bad, replace=False) if n_bad else []
    for k in bad:
        x = out[k].copy()
        if rng.random() < 0.2:
            x = _fake_lead(x, rng, truth[rng.choice(list(truth))])
        else:
            sev = rng.uniform(0.1, 1.0)
            for fn in rng.choice(CORRUPTIONS, int(rng.integers(1, 3)), replace=False):
                x = fn(x, rng, sev)
        out[k] = x
    # mild natural gaps on clean leads, like the real extractor
    for k in out:
        if k not in bad and rng.random() < 0.4:
            out[k] = _gap(out[k].copy(), rng, rng.uniform(0.02, 0.15))
    qual = {}
    for k in truth:
        t, c = truth[k], out[k]
        ok = np.isfinite(c)
        cov = ok.mean()
        if ok.sum() < 10:
            qual[k] = 0.0
            continue
        tt, cc = t[ok] - t[ok].mean(), c[ok] - c[ok].mean()
        e = np.sqrt(np.mean((tt - cc) ** 2)) / (t.std() + 1e-9)
        qual[k] = 100.0 * max(0.0, 1 - min(e, 1.0)) * np.sqrt(cov)
    return out, qual
