"""Score an extractor signals.npz: python selector/score.py <signals.npz> [...]"""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from features import record_features, FS
from data import LEADS

_MODEL = None


def load_leads(npz):
    s = np.load(npz)
    out = {}
    for k in LEADS:
        if f"{k}__amplitude_mv" not in s.files:
            continue
        t = s[f"{k}__time_s"].astype(float)
        a = s[f"{k}__amplitude_mv"].astype(float)
        good = np.isfinite(a) & np.isfinite(t)  # the whole extracted curve; no extractor flag filters it
        fin = np.isfinite(t)
        if good.sum() < 4 or fin.sum() < 4:
            continue
        grid = np.arange(t[fin][0], t[fin][-1], 1 / FS)
        y = np.interp(grid, t[good], a[good])
        gap = np.interp(grid, t[fin], (~good[fin]).astype(float)) > 0.5  # keep missing spans missing
        y[gap] = np.nan
        out[k] = y
    return out


def score_leads(leads):
    """Per-lead score 0-100 = P(correct) from the selector classifier (own-lead features only)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = pickle.load(open(Path(__file__).parent / "model_selector.pkl", "rb"))
    from features import lead_features
    names = list(leads)
    X = np.nan_to_num(np.array([[lead_features(leads[k])[n] for n in _MODEL["features"]] for k in names]), nan=0.0, posinf=0.0, neginf=0.0)
    p = _MODEL["model"].predict_proba(X)[:, 1] * 100
    return dict(zip(names, p))


def record_score(per_lead):
    v = np.array(list(per_lead.values()))
    return float(0.6 * v.mean() + 0.4 * v.min())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        pl = score_leads(load_leads(p))
        print(p, f"score={record_score(pl):.1f}", {k: round(v) for k, v in pl.items()})
