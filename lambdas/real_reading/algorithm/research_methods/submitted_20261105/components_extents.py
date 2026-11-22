#!/usr/bin/env python3
"""
ECG 12-lead printout digitization by dark-ink connected components.

Designed for the standard 3 x 4 lead layout + a long rhythm strip.
Uses only NumPy and OpenCV APIs supported by OpenCV 5.

Install:
    python -m pip install "opencv-python>=5,<6" numpy

Run:
    python extract_ecg.py ECG.jpg --overlay ecg_overlay.png --mask ecg_mask.png --csv ecg_pixels.csv

Outputs:
    - color overlay of the original ECG ink
    - binary 0/255 mask for downstream image processing
    - CSV per-column center and vertical extent of the detected ink

Limits:
    Requires a relatively clean, high-contrast, dark ECG line on a lighter grid.
    The CSV column median alone is NOT equivalent to full digitization in very
    steep QRS strokes. The binary mask preserves those strokes for later analysis.
    Automatic layout detection assumes a 3x4+rhythm printed ECG.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


LEADS = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
# BGR vivid colors to distinguish adjacent leads
COLORS = [
    (30, 205, 255), (255, 150, 45), (100, 240, 75), (210, 75, 255),
    (255, 85, 80), (0, 230, 180), (255, 70, 190), (20, 160, 255),
    (220, 205, 35), (155, 80, 255), (50, 245, 120), (255, 160, 140),
    (10, 60, 255),
]


def find_vertical_separators(ink: np.ndarray) -> list[tuple[int, int]]:
    """Find full-height lead dividers without Hough transform or OCR."""
    h, w = ink.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(25, round(h * 0.19)))
    )
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    num, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)

    bars = []
    for x, y, bw, bh, area in stats[1:]:
        if (bh > h * .30 and bw < w * .015
                and w * .08 < x < w * .91):
            bars.append((int(x), int(x + bw - 1), int(bh)))
    bars.sort(key=lambda item: -item[2])

    # Keep separated candidates; multiple adjacent bar pixels count as one.
    selected = []
    for left, right, height in bars:
        if all(abs((left+right)/2-(a+b)/2) > w*.08 for a, b in selected):
            selected.append((left, right))
        if len(selected) == 3:
            break
    return sorted(selected)


def find_baselines(ink: np.ndarray, separators: list[tuple[int,int]]) -> list[int]:
    """Find four horizontal ECG baseline bands by dark-pixel row projection."""
    h, w = ink.shape
    valid_columns = np.ones(w, dtype=bool)
    valid_columns[:round(w*.034)] = False
    for left, right in separators:
        valid_columns[max(0,left-7):min(w,right+8)] = False
    projection = ink[:, valid_columns].sum(axis=1).astype(np.float32)

    k = max(5, round(h * .009))
    if k % 2 == 0:
        k += 1
    smooth = np.convolve(projection, np.ones(k)/k, mode="same")
    search = smooth.copy()
    search[:round(h*.06)] = 0
    search[round(h*.95):] = 0

    peaks = []
    for _ in range(4):
        y = int(np.argmax(search))
        if search[y] < .23 * np.max(smooth):
            break
        peaks.append(y)
        delta = round(h*.16)
        search[max(0,y-delta):min(h,y+delta+1)] = 0
    peaks.sort()
    if len(peaks) != 4 or np.min(np.diff(peaks)) < h*.16:
        # Standard four equally spaced printout rows.
        return [round((row + .585)*h/4) for row in range(4)]
    return peaks


def extract_component(roi_ink: np.ndarray) -> np.ndarray:
    """Keep the wide connected waveform; discard grid dots, labels and noise."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(roi_ink, 8)
    width = roi_ink.shape[1]
    selected_label = 0
    best_score = -1.
    for label_idx in range(1, n):
        x, y, cw, ch, area = stats[label_idx]
        if cw < width * .43:
            continue
        # Long horizontally connected ECG line; area breaks width ties.
        score = (cw / width) * 100000 + area
        if score > best_score:
            best_score = score
            selected_label = label_idx
    return (labels == selected_label).astype(np.uint8) if selected_label else np.zeros_like(roi_ink)


def extract_ecg(image: np.ndarray, threshold: int=130, thickness: int=2):
    """Return color overlay, exact binary ink mask, and per-column wave samples."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = (gray < threshold).astype(np.uint8)

    separators = find_vertical_separators(ink)
    if len(separators) == 3:
        x_regions = [
            (round(w*.033), separators[0][0] - 2),
            (separators[0][1] + 2, separators[1][0] - 2),
            (separators[1][1] + 2, separators[2][0] - 2),
            (separators[2][1] + 2, w - 1),
        ]
    else:
        # Fallback when separator lines have been removed from the source.
        cuts = [round(i*w/4) for i in (1,2,3)]
        x_regions = [
            (round(w*.033), cuts[0] - 2), (cuts[0]+2, cuts[1]-2),
            (cuts[1]+2, cuts[2]-2), (cuts[2]+2, w-1)
        ]

    baseline = find_baselines(ink, separators)
    y_cuts = [max(0, baseline[0] - (baseline[1]-baseline[0])//2)]
    y_cuts.extend((baseline[r] + baseline[r+1])//2 for r in range(3))
    y_cuts.append(h)

    binary = np.zeros((h,w), np.uint8)
    color_idx = np.full((h,w), -1, dtype=np.int16)
    rows = []
    count = 0
    for r in range(4):
        y0, y1 = y_cuts[r:r+2]
        region_columns = range(4) if r < 3 else range(1)
        for c in region_columns:
            if r < 3:
                x0, x1 = x_regions[c]
                name = LEADS[r][c]
            else:
                x0, x1 = round(w*.033), w-1
                name = "rhythm"
            if x1 <= x0 or y1 <= y0:
                continue
            crop = ink[y0:y1, x0:x1+1]
            waveform = extract_component(crop)
            if not np.any(waveform):
                print(f"WARNING: no waveform in {name}", flush=True)
                continue

            binary[y0:y1, x0:x1+1][waveform.astype(bool)] = 255
            # Preserve each pixel in the mask. CSV provides an optional
            # per-column summary (not a replacement for the full mask).
            for col in range(waveform.shape[1]):
                yr = np.flatnonzero(waveform[:, col])
                if len(yr):
                    rows.append((name, x0+col, int(np.median(yr))+y0,
                                 int(yr.min())+y0, int(yr.max())+y0, len(yr)))

            # Thickness applies only to visual overlay; raw mask unchanged.
            paint = waveform if thickness <= 1 else cv2.dilate(
                waveform,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2*thickness-1, 2*thickness-1))
            )
            region = color_idx[y0:y1,x0:x1+1]
            region[paint.astype(bool)] = count
            print(f"{name:>6}: x={x0:4d}..{x1:4d}, "
                  f"waveform pixels={int(waveform.sum())}", flush=True)
            count += 1

    overlay = image.copy()
    for idx in range(count):
        overlay[color_idx == idx] = COLORS[idx % len(COLORS)]
    return overlay, binary*1, rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="Input printed 12-lead ECG image")
    p.add_argument("--overlay", default="ecg_overlay.png", help="Colored overlay PNG")
    p.add_argument("--mask", default="ecg_mask.png", help="Binary ink mask PNG")
    p.add_argument("--csv", default="ecg_pixels.csv", help="Columns / extrema CSV")
    p.add_argument("--threshold", type=int, default=130, help="Ink grayscale threshold (default: 130)")
    p.add_argument("--thickness", type=int, default=2, help="Colored overlay radius in pixels (default: 2)")
    args = p.parse_args()

    source = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source is None:
        p.error(f"Cannot read input: {args.image}")
    overlay, mask, rows = extract_ecg(source, args.threshold, args.thickness)

    for path, data in ((args.overlay, overlay), (args.mask, mask)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(path, data):
            raise RuntimeError(f"Cannot save {path}")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(("lead", "x_px", "y_median_px", "y_min_px",
                         "y_max_px", "dark_pixels_in_column"))
        writer.writerows(rows)
    print(f"OpenCV {cv2.__version__}: saved {args.overlay}, {args.mask}, {args.csv}")


if __name__ == "__main__":
    main()
