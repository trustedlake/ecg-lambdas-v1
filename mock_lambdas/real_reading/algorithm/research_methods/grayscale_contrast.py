"""Luminance contrast, independently of whether pixels carry chroma."""
import cv2
import numpy as np
from .common import trace_evidence

VERSION = 'local-luminance-v1'


def evidence(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), 7)
    contrast = np.maximum(0, background-gray)
    # Compare darkness to nearby paper, not a global fixed grayscale threshold.
    raw = (contrast > 10) & (gray < .78*np.maximum(background, 1))
    return raw.astype(np.uint8), contrast


def extract(context):
    raw, contrast = evidence(context.image)
    return trace_evidence(context, raw, contrast)
