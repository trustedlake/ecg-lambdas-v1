"""Separate trace ink from printed grid over the whole photograph.

The reference rectangle only supplies the population used to measure the
threshold. It is never used as a mask: the same rule is applied to every pixel.
"""
import cv2
import numpy as np


def ink_score(image, reference):
    """Darkness relative to local paper, discounted by warm (grid) chroma.

    Chromaticity and luminosity are measured separately on the preserved RGB:
    a shaded red grid is dark but coloured, black ink is dark in all channels.
    """
    x0, _, x1, _ = reference
    size = max(9, round((x1-x0)*.011)) | 1
    paper = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    paper = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), size*.3)
    relative = np.minimum(1.2, image.astype(np.float32)/np.maximum(paper, 1))
    darkness = 1-relative.max(2)
    warmth = np.clip(relative[..., 0]-relative[..., 1:].min(2), 0, 1)
    return darkness-warmth


def separate_ink(image, raw, reference):
    """Keep only pixels that the calling method measured and that are not grid."""
    x0, y0, x1, y1 = reference
    score = ink_score(image, reference)
    local = score[y0:y1, x0:x1]
    population = local[local > max(.10, float(np.median(local))+.08)]
    if population.size < 50:
        return np.zeros(raw.shape, bool), None
    levels = (np.clip(population, 0, 1)*255).astype(np.uint8).reshape(-1, 1)
    threshold = cv2.threshold(levels, 0, 255, cv2.THRESH_OTSU)[0]/255
    measured = raw > 0
    strong = measured & (score > threshold)
    weak = (measured & (score > threshold-.07)).astype(np.uint8)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.bincount(labels[strong], minlength=count) >= 3
    keep[0] = False
    return keep[labels], float(threshold)


def remove_spurs(mask, lengths, step, stroke):
    """Drop grid remnants hanging from one side of a steep stroke."""
    thin = mask & (lengths <= max(2, round(.8*stroke)))
    steep = mask & (lengths >= max(8, 4*stroke))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
    cleaned = mask.copy()
    height, width = mask.shape
    for k in range(1, count):
        x, y, w, h, _ = stats[k]
        if not 2 <= w <= .06*step or h > max(3, stroke):
            continue
        top, bottom = max(0, y-1), min(height, y+h+1)
        before = (max(0, x-2), x)
        after = (x+w, min(width, x+w+2))
        for attached, free in ((before, after), (after, before)):
            if (steep[top:bottom, attached[0]:attached[1]].any()
                    and not mask[top:bottom, free[0]:free[1]].any()):
                cleaned[y:y+h, x:x+w][labels[y:y+h, x:x+w] == k] = False
                break
    return cleaned
