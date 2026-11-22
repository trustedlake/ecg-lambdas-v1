"""Experimental, fixed OpenCV routing rules. No votes/IDs/reference at runtime.

Routes to a candidate, never certifies extraction or bypasses its blocking gate.
RGB and explicit layout regions are required. Thresholds are provisional physical
heuristics, not fitted to labels. No production entrypoint calls this module.
"""
import cv2
import numpy as np

VERSION = 'appearance-router-v1'


def describe(image, leads):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    region = np.zeros(image.shape[:2], np.uint8)
    for lead in leads:
        x0, y0, x1, y1 = map(int, lead['box'])
        region[max(0,y0):min(image.shape[0],y1), max(0,x0):min(image.shape[1],x1)] = 1
    if not region.any():
        raise ValueError('Empty explicit layout')
    scale = min(1., 1200 / max(image.shape[:2]))
    rgb = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(region, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = np.ptp(rgb.astype(np.float32), axis=2)
    # Local bright envelope and background variation, distinct from ink chroma.
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((31,31), np.uint8))
    contrast = background.astype(float) - gray
    ink = mask & (contrast >= 25)
    return {
        'background_spread': float(np.diff(np.percentile(background[mask], [10,90]))[0]),
        'ink_fraction': float(ink[mask].mean()),
        'colored_ink_fraction': float((chroma[ink] >= 15).mean()) if ink.any() else 0.,
        'page_chroma_p90': float(np.percentile(chroma[mask], 90)),
        'contrast_p99': float(np.percentile(contrast[mask], 99)),
    }


def choose(features):
    if not all(np.isfinite(v) for v in features.values()):
        raise ValueError('Non-finite descriptor')
    if features['ink_fraction'] < .001 or features['contrast_p99'] < 15:
        method, reason = None, 'insufficient_visible_ink'
    elif features['background_spread'] >= 30:
        method, reason = 'uneven_photo', 'uneven_background'
    elif features['colored_ink_fraction'] >= .35:
        method, reason = 'faint_ink', 'chromatic_strokes'
    elif features['page_chroma_p90'] >= 15:
        method, reason = 'chromatic_ink', 'colored_page_neutral_strokes'
    elif features['contrast_p99'] >= 100:
        method, reason = 'grayscale_contrast', 'neutral_high_contrast'
    else:
        method, reason = 'baseline_peaks', 'ambiguous_appearance_fallback'
    return {'version': VERSION, 'method': method, 'reason': reason,
            'status': 'EXPERIMENTAL' if method else 'ABSTAINED',
            'quality_verified': False}


def select(image, leads):
    features = describe(image, leads)
    return {**choose(features), 'features': features}
