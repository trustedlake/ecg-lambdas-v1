"""Measured paths between anchors of one lead, with joint conflict rejection.

Dark ink on the grid may participate in the internal route but is not exported.
No interpolated points, reference waveform or previous result is used.
"""
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .signal_dark import ink_evidence, locate_leads, trace_all


def pixel_graph(ink, costly):
    yy, xx = np.nonzero(ink)
    ids = np.full(ink.shape, -1, np.int32)
    ids[yy, xx] = np.arange(len(xx))
    rows, cols, costs = [], [], []
    h, w = ink.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            ny, nx = yy + dy, xx + dx
            ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            src = np.flatnonzero(ok); dst = ids[ny[ok], nx[ok]]
            keep = dst >= 0; src, dst = src[keep], dst[keep]
            penalty = 1 + 250 * (costly[yy[src], xx[src]] & costly[yy[dst], xx[dst]])
            rows.append(src); cols.append(dst); costs.append(np.hypot(dx, dy) * penalty)
    graph = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(xx), len(xx))).tocsr()
    return graph, ids, yy, xx


def select_paths(ink, costly, excluded, leads):
    graph, ids, yy, xx = pixel_graph(ink, costly)
    candidates, routes, details = [], [], []
    for lead in leads:
        base, step = lead['baseline_y_px'], lead['row_spacing_px']
        x0, x1 = lead['trace_x']
        seed = ink & ~costly & (abs(np.arange(ink.shape[0])[:, None] - base) < step * .18)
        seed = cv2.morphologyEx(seed.astype('uint8'), cv2.MORPH_OPEN, np.ones((1, 11), np.uint8)) > 0
        anchors = []
        for start in range(x0, x1, 25):
            sy, sx = np.nonzero(seed[:, start:min(start + 25, x1)])
            if len(sx):
                k = np.argmin(abs(sx - 12) * .5 + abs(sy - base))
                anchors.append((int(sy[k]), int(sx[k] + start)))
        route = np.zeros(ink.shape, bool); successes = 0
        for (ay, ax), (by, bx) in zip(anchors, anchors[1:]):
            if bx - ax > 110:
                continue
            nodes = np.flatnonzero((xx >= ax - 3) & (xx <= bx + 3) &
                                  (yy >= base - step * 1.65) & (yy <= base + step * 1.4))
            source, target = np.searchsorted(nodes, [ids[ay, ax], ids[by, bx]])
            distance, predecessor = dijkstra(graph[nodes][:, nodes], indices=source, return_predecessors=True)
            if not np.isfinite(distance[target]):
                continue
            index = target
            while index != source:
                node = nodes[index]; route[yy[node], xx[node]] = True
                index = predecessor[index]
            route[ay, ax] = True; successes += 1
        expanded = (cv2.dilate(route.astype('uint8'), np.ones((3, 3), np.uint8)) > 0) & ink
        expanded[:, :x0] = False; expanded[:, x1:] = False
        routes.append(expanded)
        candidates.append(expanded & ~excluded)
        details.append(dict(anchors=len(anchors), connected_intervals=successes))
    counts = np.sum(routes, axis=0)
    disputed = cv2.dilate((counts > 1).astype('uint8'), np.ones((3, 3), np.uint8)) > 0
    traces = []
    for lead, candidate, route in zip(leads, candidates, routes):
        x0, x1 = lead['trace_x']; step = lead['row_spacing_px']; base = lead['baseline_y_px']
        y0, y1 = max(0, int(base - step * 1.65) - 1), min(ink.shape[0], int(base + step * 1.4) + 2)
        region = np.s_[y0:y1, x0:x1]
        mask = (candidate & ~disputed)[region]
        observed = mask.any(0); width = x1 - x0
        center = np.full(width, np.nan, np.float32); low = center.copy(); high = center.copy()
        withheld = route[region] & (excluded[region] | disputed[region])
        ambiguous = ~observed | withheld.any(0)
        for col in np.flatnonzero(observed):
            points = np.flatnonzero(mask[:, col]) + y0
            center[col] = np.median(points); low[col] = points[0]; high[col] = points[-1]
            if np.any(np.diff(points) > 1):
                ambiguous[col] = True
        valid = observed & ~ambiguous
        traces.append(dict(x_px=np.arange(x0, x1, dtype=np.float32), y_px=center,
            y_min_px=low, y_max_px=high, mask=mask, mask_origin=[x0, y0],
            observed=observed, ambiguous=ambiguous, valid=valid,
            baseline_y_px=float(np.nanmedian(center)) if observed.any() else float(base),
            baseline_method='median_observed_trace_not_clinical_isoelectric_reference',
            method='same_lead_anchor_paths_grid_abstention',
            coverage=float(observed.mean()), unambiguous_fraction=float(valid.mean())))
    report = dict(method='same_lead_anchor_paths_grid_abstention',
        shared_pixels_before=int((counts > 1).sum()), shared_pixels_after=0,
        disputed_measured_pixels=int((disputed & ink).sum()),
        leads={l['name']:d for l,d in zip(leads,details)},
        limitation='Grid pixels withheld even when used for connectivity. Paths can omit branches; exclusivity is not proof of accuracy.')
    return traces, report


def merge_traces(previous, paths, shape):
    """Keep measured branches from both selectors; abstain on joint conflicts."""
    masks = []
    for old, new in zip(previous, paths):
        full = np.zeros(shape, bool)
        for t in (old, new):
            x, y = t['mask_origin']; h, w = t['mask'].shape
            full[y:y+h, x:x+w] |= t['mask']
        masks.append(full)
    shared = np.sum(masks, axis=0) > 1
    disputed = cv2.dilate(shared.astype('uint8'), np.ones((3, 3), np.uint8)) > 0
    result = []
    for old, new, full in zip(previous, paths, masks):
        x0 = min(old['mask_origin'][0], new['mask_origin'][0])
        y0 = min(old['mask_origin'][1], new['mask_origin'][1])
        x1 = max(t['mask_origin'][0]+t['mask'].shape[1] for t in (old,new))
        y1 = max(t['mask_origin'][1]+t['mask'].shape[0] for t in (old,new))
        mask = (full & ~disputed)[y0:y1,x0:x1]
        observed = mask.any(0)
        ambiguous = ~observed | (full & disputed)[y0:y1,x0:x1].any(0)
        for t in (old,new):
            start = t['mask_origin'][0]-x0
            ambiguous[start:start+len(t['ambiguous'])] |= t['ambiguous'] & t['observed']
        center = np.full(x1-x0,np.nan,np.float32); low=center.copy(); high=center.copy()
        for col in np.flatnonzero(observed):
            points=np.flatnonzero(mask[:,col])+y0
            center[col]=np.median(points);low[col]=points[0];high[col]=points[-1]
            ambiguous[col] |= bool(np.any(np.diff(points)>1))
        valid=observed & ~ambiguous
        t=dict(new)
        t.update(mask=mask,mask_origin=[x0,y0],x_px=np.arange(x0,x1,dtype=np.float32),
            y_px=center,y_min_px=low,y_max_px=high,observed=observed,ambiguous=ambiguous,
            valid=valid,coverage=float(observed.mean()),unambiguous_fraction=float(valid.mean()),
            baseline_y_px=float(np.nanmedian(center)) if observed.any() else new['baseline_y_px'],
            method='measured_branches_and_anchor_paths_joint_abstention')
        result.append(t)
    return result, int(shared.sum())


def extract(a, grid, config, geometry):
    raw, excluded = ink_evidence(a, grid)
    gray = a.max(2); background = float(np.percentile(gray, 90))
    ink = (gray < background * .65).astype('uint8')
    core = (cv2.distanceTransform(ink, cv2.DIST_L2, 5) > 2.1).astype('uint8')
    thick = (cv2.dilate(core, np.ones((3, 3), np.uint8)) > 0) & (ink > 0)
    leads = locate_leads((raw | thick).astype('uint8'), config, geometry)
    traces, report = select_paths(ink > 0, excluded & ~thick, excluded, leads)
    old_leads = locate_leads(raw.astype('uint8'), config, geometry)
    previous, _ = trace_all(raw, excluded, old_leads)
    assert [l['name'] for l in old_leads] == [l['name'] for l in leads]
    traces, shared = merge_traces(previous, traces, raw.shape)
    report.update(method='measured_branches_and_anchor_paths_joint_abstention',
                  merge_shared_pixels_before=shared, shared_pixels_after=0)
    return leads, traces, report
