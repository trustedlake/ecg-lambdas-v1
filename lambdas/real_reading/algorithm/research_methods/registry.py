"""Explicit research registry. No automatic selection is trained yet."""
from . import baseline_peaks, chromatic_ink, faint_ink, uneven_photo, grayscale_contrast, same_appearance

METHODS = {module.__name__.rsplit('.', 1)[-1]: module for module in (
    baseline_peaks, chromatic_ink, faint_ink, uneven_photo, grayscale_contrast, same_appearance)}
