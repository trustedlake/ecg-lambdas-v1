"""Local illumination normalization retains RGB; no geometric warping."""
import cv2
import numpy as np
from .frozen_peaks34.ecg import signal
from .common import trace_evidence

VERSION = 'local-rgb-v1'


def evidence(image):
    pixels = image.astype(np.float32)
    # A common gain for the three channels preserves their ratios.
    light = pixels.max(2)
    background = cv2.GaussianBlur(light, (0, 0), max(9., min(light.shape)*.035))
    corrected = np.clip(pixels * (245./np.maximum(background, 30.))[..., None], 0, 255)
    return signal.ink_evidence(corrected.astype(np.uint8))


def extract(context):
    raw, contrast = evidence(context.image)
    return trace_evidence(context, raw, contrast)
