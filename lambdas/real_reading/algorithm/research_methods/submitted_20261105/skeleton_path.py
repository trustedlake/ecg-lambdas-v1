#!/usr/bin/env python3
"""Detect and draw ECG traces using OpenCV 5 + NumPy.

Designed for the common 12-lead, 3-row x 4-column ECG printout with a
full-width rhythm strip below it (same layout as the example image).
No AI/LLM, pretrained model, or clinical interpretation is involved.

Install: python -m pip install opencv-python-headless==5.0.0.93 numpy
Run:     python ecg_trace_opencv5.py 59.jpg --output overlay.png --csv points.csv

`extract_ecg(image)` is also importable; it returns pixel paths for each lead.
Coordinates are in the ORIGINAL image; y increases downward. These are image
coordinates, NOT calibrated voltage/time measurements.
"""

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np

LEADS = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
    ("III_rhythm",),
)
# OpenCV uses BGR, not RGB.
COLORS = ((0, 0, 255), (255, 0, 0), (0, 185, 30), (0, 140, 255),
          (210, 0, 210), (0, 190, 200), (220, 30, 30), (0, 75, 240),
          (190, 0, 190), (100, 120, 0), (0, 160, 70), (200, 0, 100),
          (0, 150, 255))


def thin_zhang_suen(binary: np.ndarray) -> np.ndarray:
    """One-pixel centerline by Zhang-Suen thinning, implemented in NumPy."""
    a = np.pad((binary > 0).astype(np.uint8), 1)
    for _ in range(150):
        removed = 0
        for step in (0, 1):
            p2, p3, p4 = a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:]
            p5, p6, p7 = a[2:, 2:], a[2:, 1:-1], a[2:, :-2]
            p8, p9 = a[1:-1, :-2], a[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = sum((u == 0) & (v == 1) for u, v in
                              ((p2, p3), (p3, p4), (p4, p5), (p5, p6),
                               (p6, p7), (p7, p8), (p8, p9), (p9, p2)))
            if step == 0:
                corner = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                corner = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            kill = ((a[1:-1, 1:-1] == 1) & (neighbors >= 2) &
                    (neighbors <= 6) & (transitions == 1) & corner)
            removed += int(kill.sum())
            a[1:-1, 1:-1][kill] = 0
        if not removed:
            break
    return a[1:-1, 1:-1]


def shortest_ink_path(skeleton, start, ends):
    """Follow actual connected ink pixels, never jump across the background."""
    h, w = skeleton.shape
    ends = set(ends)
    parents = {start: None}
    visited = np.zeros_like(skeleton)
    visited[start] = 1
    queue = deque([start])
    final = None
    while queue:
        y, x = queue.popleft()
        if (y, x) in ends:
            final = (y, x)
            break
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1),
                       (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)):
            if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx] and not visited[yy, xx]:
                visited[yy, xx] = 1
                parents[(yy, xx)] = (y, x)
                queue.append((yy, xx))
    path = []
    while final is not None:
        path.append(final)
        final = parents[final]
    return path[::-1]


def find_qrs_extrema(ink, skeleton, baseline_y, amplitude_px=19):
    """Force path through deep QRS tips; otherwise graph shortcuts may miss them."""
    h, w = ink.shape
    candidates = []
    ys = np.arange(h)[:, None]
    for polarity in (+1, -1):
        if polarity == 1:
            edge = np.max(np.where(ink, ys, -1000), axis=0)
            magnitude = edge - baseline_y
        else:
            edge = np.min(np.where(ink, ys, 1000), axis=0)
            magnitude = baseline_y - edge
        columns = np.where((magnitude > amplitude_px) &
                           (np.arange(w) > 12) & (np.arange(w) < w-12))[0]
        if not len(columns):
            continue
        # Nearby candidate columns belong to the same peak.
        for group in np.split(columns, np.where(np.diff(columns) > 7)[0] + 1):
            x = int(group[np.argmax(magnitude[group])])
            candidates.append((x, int(edge[x]), int(magnitude[x]), polarity))
    candidates.sort(key=lambda item: item[0])
    merged = []
    for p in candidates:
        if merged and p[0] - merged[-1][0] < 12 and p[3] == merged[-1][3]:
            if p[2] > merged[-1][2]:
                merged[-1] = p
        else:
            merged.append(p)
    nodes = np.column_stack(np.where(skeleton != 0))  # [y, x]
    if not len(nodes):
        return []
    result = []
    for x, y, _, _ in merged:
        dist = (nodes[:, 1] - x)**2 + (nodes[:, 0] - y)**2
        at = int(np.argmin(dist))
        if dist[at] < 12**2:
            result.append(tuple(map(int, nodes[at])))
    return result


def detect_rows_and_columns(ink):
    """Find the 4 baseline bands and the 3 long vertical lead separators."""
    h, w = ink.shape
    row_score = ink[:, int(w*.045):int(w*.98)].sum(axis=1).astype(float)
    k = max(5, int(h*.013) | 1)
    smoothed = np.convolve(row_score, np.ones(k)/k, mode="same")
    centers = []
    for y in np.argsort(smoothed)[::-1]:
        if int(h*.08) < y < int(h*.97) and all(abs(y-r) > h*.13 for r in centers):
            centers.append(int(y))
            if len(centers) == 4:
                break
    if len(centers) != 4:
        raise ValueError("Could not identify the four ECG rows")
    centers.sort()
    if min(np.diff(centers)) < h*.15:
        raise ValueError("Unexpected row layout; expected 3 rows + rhythm strip")

    strip = ink[int(h*.10):int(h*.70)].astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h*.24))))
    long_lines = cv2.morphologyEx(strip, cv2.MORPH_OPEN, kernel)
    col_score = (long_lines > 0).sum(axis=0)
    columns = np.where((col_score > h*.30) &
                       (np.arange(w) > w*.15) & (np.arange(w) < w*.90))[0]
    groups = np.split(columns, np.where(np.diff(columns) > 2)[0]+1)
    separators = [int(np.median(group)) for group in groups if len(group)]
    if len(separators) != 3:
        raise ValueError(f"Found {len(separators)} lead separators, expected 3")
    return centers, separators


def extract_lead(ink, bounds, baseline, amplitude_px=19):
    """Extract an ordered (N, 2) array of x,y pixels from a lead box."""
    x0, y0, x1, y1 = bounds
    region = ink[y0:y1, x0:x1].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    # The ECG line normally spans almost the full box. Exclude isolated text/grid.
    matches = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] > region.shape[1]*.65]
    if not matches:
        raise ValueError("No connected, full-width ECG curve was found")
    main = max(matches, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    signal = np.uint8(labels == main)
    skel = thin_zhang_suen(signal)
    nodes = np.column_stack(np.where(skel))
    if not len(nodes):
        raise ValueError("Empty skeleton")
    xmin, xmax = int(nodes[:, 1].min()), int(nodes[:, 1].max())
    starters = nodes[nodes[:, 1] <= xmin+1]
    finishers = nodes[nodes[:, 1] >= xmax-1]
    y_start = np.median(starters[:, 0])
    start = tuple(map(int, starters[np.argmin(abs(starters[:, 0]-y_start))]))
    targets = [tuple(map(int, p)) for p in finishers]
    forced = find_qrs_extrema(signal, skel, baseline - y0, amplitude_px)
    current = start
    joined = []
    for mark in forced:
        segment = shortest_ink_path(skel, current, [mark])
        if segment:
            joined.extend(segment if not joined else segment[1:])
            current = segment[-1]
    tail = shortest_ink_path(skel, current, targets)
    if not tail:
        raise ValueError("The connected curve does not reach the right edge")
    joined.extend(tail if not joined else tail[1:])
    pixels = np.asarray([(x+x0, y+y0) for y,x in joined], dtype=np.int32)
    if pixels[:, 0].max() - pixels[:, 0].min() < region.shape[1]*.60:
        raise ValueError("Extracted path is too short")
    return pixels


def extract_ecg(image_bgr, threshold=140):
    """Return {lead_name: Nx2 pixel path} from a standard 12-lead ECG image."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink = gray < threshold
    row_centers, dividers = detect_rows_and_columns(ink)
    limits = [int(w*.033), *dividers, int(w*.988)]
    traces = {}
    for r, row in enumerate(LEADS):
        spacing_up = row_centers[r]-row_centers[r-1] if r else row_centers[1]-row_centers[0]
        spacing_down = row_centers[r+1]-row_centers[r] if r<3 else spacing_up
        top_factors = (.15, .22, .23, .10)
        bottom_factors = (.75, .70, .86, .43)
        top = max(0, int(row_centers[r]-top_factors[r]*spacing_up))
        bottom = min(h, int(row_centers[r]+bottom_factors[r]*spacing_down))
        for c, name in enumerate(row):
            if r == 3:
                x0, x1 = int(w*.046), int(w*.978)
            else:
                x0, x1 = limits[c]+int(w*.013), limits[c+1]-int(w*.006)
            if not (x0 < x1 and top < bottom):
                raise ValueError(f"Invalid region for {name}")
            try:
                traces[name] = extract_lead(ink, (x0, top, x1, bottom), row_centers[r],
                                            amplitude_px=max(8, int(h*.023)))
            except ValueError as exc:
                print(f"WARNING: {name} skipped: {exc}")
    return traces


def draw_overlay(image_bgr, traces, thickness=2):
    output = image_bgr.copy()
    names = [name for row in LEADS for name in row]
    for name, color in zip(names, COLORS):
        if name in traces:
            cv2.polylines(output, [traces[name].reshape(-1,1,2)],
                          isClosed=False, color=color, thickness=thickness,
                          lineType=cv2.LINE_AA)
    return output


def write_csv(filename, traces):
    with open(filename, "w", newline="", encoding="utf-8") as fd:
        writer = csv.writer(fd)
        writer.writerow(["lead", "point_index", "x_px", "y_px"])
        for lead, path in traces.items():
            writer.writerows((lead, i, int(x), int(y)) for i,(x,y) in enumerate(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="input JPG/PNG of a 12-lead ECG")
    ap.add_argument("--output", type=Path, default=Path("ecg_overlay.png"))
    ap.add_argument("--csv", type=Path, default=Path("ecg_points.csv"))
    ap.add_argument("--threshold", type=int, default=140, help="black-ink threshold (0-255)")
    args = ap.parse_args()
    if not 1 <= args.threshold <= 255:
        ap.error("--threshold must be between 1 and 255")
    image = cv2.imread(str(args.image))
    if image is None:
        ap.error(f"Cannot open image: {args.image}")
    print(f"OpenCV version: {cv2.__version__}")
    if cv2.__version__.split('.')[0] != '5':
        print("NOTE: designed for OpenCV 5; this execution uses an available compatible version")
    traces = extract_ecg(image, args.threshold)
    if not traces:
        raise SystemExit("No ECG curves extracted")
    overlay = draw_overlay(image, traces)
    if not cv2.imwrite(str(args.output), overlay):
        raise SystemExit(f"Cannot write: {args.output}")
    write_csv(args.csv, traces)
    print(f"Extracted {len(traces)}/13 strips: {', '.join(traces)}")
    print(f"Overlay: {args.output}\nPixel paths: {args.csv}")


if __name__ == "__main__":
    main()
