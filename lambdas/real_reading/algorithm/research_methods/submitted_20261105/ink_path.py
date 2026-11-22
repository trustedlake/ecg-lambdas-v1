#!/usr/bin/env python3
"""Detect and highlight ECG ink in a conventional 3x4 + rhythm-strip ECG.

Uses OpenCV morphology/components and shortest-path tracing; no neural network.
Outputs an annotated PNG, a binary signal mask PNG, and trace point CSV/JSON.
Coordinates are in IMAGE PIXELS, not calibrated mV or seconds.

Example:
    pip install opencv-python==5.0.0.93 numpy
    python ecg_trace_opencv5.py 57.jpg --output resultado_ecg.png

Public API:
    results, overlay, signal_mask = extract_ecg(image_bgr, threshold=None)
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
from pathlib import Path

import cv2
import numpy as np


# Lead positions in the conventional 12-lead + extended lead III format.
LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
# Bright colors in BGR format, one per row; rhythm strip uses yellow.
ROW_COLORS = [(30, 195, 0), (230, 60, 0), (0, 60, 255)]
RHYTHM_COLOR = (0, 200, 230)


def _pick_spaced_peaks(values: np.ndarray, count: int, min_distance: int,
                       left: int, right: int) -> list[int]:
    """Select strongest separated projection peaks, with sub-pixel-noise smoothing."""
    smooth = np.convolve(values.astype(float), [0.2, 0.6, 0.2], mode="same")
    order = np.argsort(smooth[left:right])[::-1] + left
    out = []
    for i in order:
        if all(abs(int(i) - p) >= min_distance for p in out):
            out.append(int(i))
            if len(out) == count:
                break
    return sorted(out)


def detect_layout(ink: np.ndarray) -> tuple[list[int], list[int]]:
    """Find the four wave baselines and three long vertical column dividers."""
    h, w = ink.shape
    # Continuous ECG baselines produce a strong horizontal ink projection.
    row_projection = ink[:, max(0, int(w * .035)):max(1, int(w * .995))].sum(axis=1)
    baselines = _pick_spaced_peaks(row_projection, 4, int(h * .16),
                                   int(h * .08), int(h * .96))
    if len(baselines) != 4:
        raise ValueError("Could not locate the four ECG rows")

    # Long, dark dividers separate the conventional 4 columns.
    top = max(0, baselines[0] - int(h * .07))
    bottom = min(h, baselines[2] + int(h * .17))
    column_projection = ink[top:bottom, :].sum(axis=0)
    dividers = _pick_spaced_peaks(column_projection, 3, int(w * .15),
                                  int(w * .12), int(w * .91))
    if len(dividers) != 3:
        raise ValueError("Could not locate the three ECG column dividers")
    return baselines, dividers


def _main_signal_component(crop: np.ndarray, baseline_y: int) -> tuple[np.ndarray, float]:
    """Keep the long connected waveform, excluding loose grid marks/letters."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(crop, 8)
    ch, cw = crop.shape
    choices = []
    for i in range(1, count):
        xx, yy, ww, hh, area = (int(v) for v in stats[i])
        if ww < .42 * cw or area < .004 * cw * ch:
            continue
        # Reward sustained horizontal coverage and proximity to wave baseline.
        center_yy = float(centroids[i, 1])
        score = (ww / cw) * 3 + np.log1p(area) / 10 - abs(center_yy - baseline_y) / max(ch, 1)
        choices.append((score, i, ww))
    if not choices:
        return np.zeros_like(crop, np.uint8), 0.0
    _, best_id, ww = max(choices)
    mask = (labels == best_id).astype(np.uint8)
    return mask, min(1., ww / cw)


def _walk_ink(mask: np.ndarray, start: tuple[int, int],
              goals: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Dijkstra over 8-connected ECG pixels; node/goal tuples use (y,x)."""
    height, width = mask.shape
    source_key = start[0] * width + start[1]
    distances = {source_key: 0.0}
    parents = {source_key: -1}
    heap = [(0.0, source_key)]
    directions = ((1,0,1.),(-1,0,1.),(0,1,1.),(0,-1,1.),
                  (1,1,1.4142),(1,-1,1.4142),(-1,1,1.4142),(-1,-1,1.4142))
    end_key = None
    while heap:
        dist, k = heapq.heappop(heap)
        if dist > distances.get(k, float("inf")):
            continue
        y, x = divmod(k, width)
        if (y, x) in goals:
            end_key = k
            break
        for dy, dx, cost in directions:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width and mask[ny, nx]):
                continue
            nk = ny * width + nx
            nd = dist + cost
            if nd < distances.get(nk, float("inf")):
                distances[nk] = nd
                parents[nk] = k
                heapq.heappush(heap, (nd, nk))
    if end_key is None:
        return []
    path = []
    k = end_key
    while k != -1:
        y, x = divmod(k, width)
        path.append((x, y))
        k = parents[k]
    path.reverse()
    return path


def _shortest_stroke_path(mask: np.ndarray, baseline_y: int) -> list[tuple[int, int]]:
    """Trace the connected waveform without cutting the tips off sharp QRS spikes.

    First find a continuous ink path. Then detect strong extrema missed by its
    shortest-path shortcut and force the route to visit those ink extrema.
    Coordinates are (x,y), relative to this crop. Sharp vertical segments can
    contain multiple points with the same x value.
    """
    yy, xx = np.where(mask > 0)
    if len(xx) == 0:
        return []
    minx, maxx = int(xx.min()), int(xx.max())
    sy = np.flatnonzero(mask[:, minx])
    ey = np.flatnonzero(mask[:, maxx])
    start = (int(sy[np.argmin(abs(sy - baseline_y))]), minx)
    goal_pixels = {(int(y), maxx) for y in ey}
    original = _walk_ink(mask, start, goal_pixels)
    if len(original) < 2:
        return original

    # Find black ECG ink endpoints that shortest-path routing bypassed.
    height, width = mask.shape
    ink_top = np.full(width, height, np.int32)
    ink_bottom = np.full(width, -1, np.int32)
    np.minimum.at(ink_top, xx, yy)
    np.maximum.at(ink_bottom, xx, yy)
    curve_top = np.full(width, height, np.int32)
    curve_bottom = np.full(width, -1, np.int32)
    path_arr = np.array(original, dtype=np.int32)
    np.minimum.at(curve_top, path_arr[:, 0], path_arr[:, 1])
    np.maximum.at(curve_bottom, path_arr[:, 0], path_arr[:, 1])
    valid = curve_bottom >= 0
    # Shortest path is within the mask; typically 0-2 px of difference is just
    # stroke thickness. Differences >6 px are missing peaks, not line width.
    miss_up = np.where(valid, curve_top - ink_top, 0)
    miss_down = np.where(valid, ink_bottom - curve_bottom, 0)
    cut = 7
    minimum_excursion = max(12, int(height * .06))
    candidates = []
    for kind, miss, extrema in (
        ("up", miss_up, ink_top), ("down", miss_down, ink_bottom)
    ):
        hits = (miss >= cut) & (abs(extrema - baseline_y) >= minimum_excursion)
        hits[:3] = False
        hits[-3:] = False
        indices = np.flatnonzero(hits)
        if len(indices) == 0:
            continue
        # Merge nearby missed-tip pixels from the same QRS into one landmark.
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1)
        for group in groups:
            chosen = int(group[np.argmax(miss[group])])
            candidates.append((int(miss[chosen]), chosen, int(extrema[chosen])))
    # Suppress almost coincident landmarks and limit noisy scans.
    candidates.sort(reverse=True)
    landmarks = []
    for magnitude, x, y in candidates:
        if all(abs(x - px) > 5 for px, py in landmarks):
            landmarks.append((x, y))
        if len(landmarks) >= 30:
            break
    landmarks.sort()
    if not landmarks:
        return original

    points = original[:1]
    cursor = (original[0][1], original[0][0])
    for x, y in landmarks:
        segment = _walk_ink(mask, cursor, {(y, x)})
        if segment:
            points.extend(segment[1:])
            cursor = (y, x)
    finish = (original[-1][1], original[-1][0])
    segment = _walk_ink(mask, cursor, {finish})
    if segment:
        points.extend(segment[1:])
    return points

def extract_ecg(image_bgr: np.ndarray, threshold: int | None = None):
    """Return detected lead point arrays, annotated BGR image, and pure ink mask.

    Expected input: standard 3 rows x 4 leads + full-width bottom rhythm strip.
    The detected grid/layout is automatic, but page geometry must be similar.
    """
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("Expected a color ECG image (BGR)")
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if threshold is None:
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        threshold = int(min(175, max(65, otsu)))
    ink = (gray < threshold).astype(np.uint8)
    baselines, dividers = detect_layout(ink)
    pitch = float(np.median(np.diff(baselines)))
    if pitch <= 0:
        raise ValueError("Invalid ECG row spacing")

    # Keep away from vertical dividers and the left calibration pulses.
    xm = max(3, int(w * .003))
    x_regions = [(max(0, int(w * .032)), dividers[0] - xm),
                 (dividers[0] + xm, dividers[1] - xm),
                 (dividers[1] + xm, dividers[2] - xm),
                 (dividers[2] + xm, w - 2)]
    y_regions = [
        (int(baselines[0] - .22*pitch), int(baselines[0] + .70*pitch)),
        (int(baselines[1] - .20*pitch), int(baselines[1] + .75*pitch)),
        (int(baselines[2] - .47*pitch), int(baselines[2] + .78*pitch)),
    ]

    overlay = image_bgr.copy()
    detected_mask = np.zeros((h, w), np.uint8)
    results = []

    def process(lead: str, x0: int, x1: int, y0: int, y1: int,
                baseline: int, color: tuple[int, int, int]):
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        crop = ink[y0:y1, x0:x1]
        if crop.size == 0:
            return
        component, coverage = _main_signal_component(crop, baseline - y0)
        if not np.any(component):
            results.append({"lead": lead, "status": "not_found", "coverage": 0., "points": []})
            return
        pixels = component.astype(bool)
        detected_mask[y0:y1, x0:x1][pixels] = 255
        # Exact pixel-level trace fill is important: even if the vector route
        # misses a branch, the visual overlay still highlights the detected ink.
        overlay[y0:y1, x0:x1][pixels] = color
        local_points = _shortest_stroke_path(component, baseline - y0)
        points = [[int(x + x0), int(y + y0)] for x, y in local_points]
        if len(points) >= 2:
            arr = np.asarray(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [arr], isClosed=False, color=color, thickness=2,
                          lineType=cv2.LINE_AA)
        results.append({"lead": lead, "status": "ok" if points else "path_failed",
                        "coverage": round(float(coverage), 3),
                        "roi": [x0, y0, x1, y1], "baseline_px": int(baseline),
                        "points": points})

    for r in range(3):
        for c in range(4):
            x0, x1 = x_regions[c]
            y0, y1 = y_regions[r]
            process(LEAD_LAYOUT[r][c], x0, x1, y0, y1,
                    baselines[r], ROW_COLORS[r])

    # A long-lead III rhythm strip occupying the bottom row.
    process("III_rhythm", int(w*.032), w-2,
            int(baselines[3]-.19*pitch), int(baselines[3]+.43*pitch),
            baselines[3], RHYTHM_COLOR)
    return results, overlay, detected_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ECG JPG/PNG input")
    parser.add_argument("--output", type=Path, default=Path("ecg_overlay.png"),
                        help="Annotated PNG output")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Optional grayscale darkness cutoff, e.g., 120")
    args = parser.parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Cannot read image: {args.input}")
    results, overlay, mask = extract_ecg(image, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = args.output.with_suffix("")
    cv2.imwrite(str(args.output), overlay)
    cv2.imwrite(str(root) + "_mask.png", mask)
    with (root.parent / (root.name + "_points.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lead", "point_index", "x_px", "y_px", "baseline_px"])
        for entry in results:
            for i, (x, y) in enumerate(entry["points"]):
                writer.writerow([entry["lead"], i, x, y, entry.get("baseline_px", "")])
    with (root.parent / (root.name + "_points.json")).open("w", encoding="utf-8") as f:
        json.dump({"image_width_px": image.shape[1],
                   "image_height_px": image.shape[0],
                   "opencv_version": cv2.__version__,
                   "units": "raw_image_pixels_not_calibrated",
                   "leads": results}, f, ensure_ascii=False)
    print("OpenCV:", cv2.__version__)
    for entry in results:
        print(f'{entry["lead"]:>10}  {entry["status"]:11}  '
              f'coverage={entry["coverage"]:.0%}  points={len(entry["points"])}')
    print("Files:", args.output, root.parent/(root.name+"_mask.png"),
          root.parent/(root.name+"_points.csv"),
          root.parent/(root.name+"_points.json"))


if __name__ == "__main__":
    main()
