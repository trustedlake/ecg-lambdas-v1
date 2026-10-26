"""Conservative appearance gate, calibrated on the 50-image development batch.

Receives prepared RGB pixels only. This is applicability, not clinical quality.
The unchanged extractor remains callable directly for gate-off experiments.
"""
import cv2
import numpy as np
from .contracts import MethodAbstention

VERSION = 'local-ink-graph-appearance-gate'


def assess(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB image')
    # Fixed normalized interior excludes page borders; never a per-case ROI.
    pixels = cv2.resize(rgb, (800, 600))[60:540, 80:720].astype(np.float32)
    light = pixels.mean(axis=2)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    median = float(np.median(light))
    upper = float(np.percentile(light, 90))
    features = dict(luminance_median=median, luminance_p90=upper,
                    chroma_median=float(np.median(chroma)),
                    luminance_spread=upper-median)
    checks = dict(gray_paper=185 <= median <= 230,
                  no_white_background=upper <= 240,
                  low_color_cast=features['chroma_median'] <= 12,
                  visible_contrast=features['luminance_spread'] >= 8)
    return dict(active=all(checks.values()), features=features, checks=checks,
                production_blocked=True, calibration='development_batch_only')


def extract(context):
    decision = assess(context.image)
    if not decision['active']:
        raise MethodAbstention('Tinta local / conectividade: appearance outside scope')
    from .local_ink_graph import extract as ungated_extract
    result = ungated_extract(context)
    result.diagnostics['activation'] = decision
    return result
