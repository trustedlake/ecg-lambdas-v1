"""Appearance descriptors, not diagnoses or a trained method selector."""
import cv2
import numpy as np


def measure(image, leads):
    region = np.zeros(image.shape[:2], bool)
    for lead in leads:
        x0, y0, x1, y1 = lead['box']
        region[max(0,y0):y1, max(0,x0):x1] = True
    pixels = image[region].astype(np.float32)
    if not len(pixels):
        raise ValueError('Empty ECG region')
    chroma = pixels.max(1)-pixels.min(1)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    light = gray[region]
    return {'near_neutral_fraction': float(np.mean(chroma <= 3)),
            'chroma_p90': float(np.percentile(chroma, 90)),
            'luminance_p10': float(np.percentile(light, 10)),
            'luminance_p90': float(np.percentile(light, 90)),
            'laplacian_variance': float(cv2.Laplacian(gray, cv2.CV_32F)[region].var()),
            'scope': 'Layout ECG boxes; may contain labels/grid. Not a trace/grid contrast classifier.'}
