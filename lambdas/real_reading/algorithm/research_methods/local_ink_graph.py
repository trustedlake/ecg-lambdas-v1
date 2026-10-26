"""Independent local-contrast/component graph experiment. No ancestor extractor.

Only the frozen layout locator and the data contract are shared infrastructure.
Input is RGB and explicit layout, never image identity. Missing ink stays missing.
"""
import cv2
import numpy as np
from .contracts import Extraction
from .frozen_peaks34.ecg.signal_tinted import locate_leads

VERSION = 'local-ink-graph-connected-neutral-strokes'


def evidence(rgb):
    light = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    scale = max(9, round(min(light.shape) * .016) | 1)
    # Estimate the local paper in each channel: red grid absorbs green/blue,
    # whereas neutral trace ink absorbs all three. Keep these separate from light.
    smooth = cv2.GaussianBlur(rgb.astype(np.float32), (9, 3), 2.0, sigmaY=.65)
    paper_rgb = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale)))
    absorption = np.maximum(0, paper_rgb - smooth)
    neutral = absorption.min(axis=2)
    colored = absorption.max(axis=2) - neutral
    residual = np.maximum(0, neutral - .65 * colored)
    darkness = cv2.cvtColor(absorption, cv2.COLOR_RGB2GRAY)
    grid = colored
    red_excess = smooth[:, :, 0] - (smooth[:, :, 1] + smooth[:, :, 2]) / 2
    noise = cv2.GaussianBlur(np.minimum(neutral, 20), (0, 0), scale)
    ink = ((neutral > np.maximum(6, noise * 2.5)) & (red_excess < 6)) | ((neutral > 45) & (red_excess < 6))
    frame = np.zeros(light.shape, bool)
    # Long page-border segments may be tilted. Detect them geometrically.
    border = np.zeros(light.shape, np.uint8)
    lines = cv2.HoughLinesP(((darkness > 20) * 255).astype(np.uint8), 1, np.pi / 1800,
                            threshold=200, minLineLength=int(light.shape[1] * .65), maxLineGap=12)
    if lines is not None:
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            edge_x = max(x0, x1) < light.shape[1]*.06 or min(x0, x1) > light.shape[1]*.94
            edge_y = max(y0, y1) < light.shape[0]*.06 or min(y0, y1) > light.shape[0]*.97
            if edge_x or edge_y:
                cv2.line(border, (x0, y0), (x1, y1), 1, 11)
    frame |= border > 0
    ink &= ~frame
    # Reject isolated moire speckles before they can form artificial graph edges.
    _, cc, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    stable = (stats[:,4] >= 8) & (np.maximum(stats[:,2],stats[:,3]) >= 5)
    stable[0] = False
    ink &= stable[cc]
    return ink, residual, grid, frame


def extract(context):
    ink, strength, grid, frame = evidence(context.image)
    joined = cv2.dilate(ink.astype(np.uint8), np.ones((3, 7), np.uint8))
    _, components, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    keep = (stats[:, 2] > ink.shape[1] * .045) & (stats[:, 3] < ink.shape[0] * .35)
    keep[0] = False
    layout_ink = ink & keep[components]
    leads = locate_leads(layout_ink.astype(np.uint8), context.config, context.geometry)
    # Locate the sloping baseline by support across the page, not a single
    # horizontal histogram peak (which can be a grid line or footer).
    priors = context.config.get('paper_baseline_fractions', context.config['baseline_fractions'])
    step = float(np.median(np.diff(priors))) * ink.shape[0]
    rgbf = cv2.GaussianBlur(context.image.astype(np.float32), (5,5), 1)
    redness = rgbf[:,:,0] - (rgbf[:,:,1]+rgbf[:,:,2])/2
    support = (ink & (redness < 2)).astype(np.float32)
    support = cv2.boxFilter(support, -1, (1, 9), normalize=False)
    xs = np.arange(int(ink.shape[1]*.08), int(ink.shape[1]*.9), 3)
    lines = []
    for row, prior in enumerate(priors):
        offset = float(np.median([line[1]-priors[j]*ink.shape[0] for j,line in enumerate(lines)])) if lines else 0.
        expected = prior*ink.shape[0] + offset
        best = (-1, prior*ink.shape[0], 0.)
        for slope in np.linspace(-.12, .12, 49):
            centers = np.arange(int(expected-step*.4), int(expected+step*.35), 2)
            yy = np.rint(centers[:,None] + slope*(xs[None,:]-ink.shape[1]/2)).astype(int)
            yy = np.clip(yy, 0, ink.shape[0]-1)
            score = np.minimum(support[yy,xs], 3).mean(1)
            k = int(np.argmax(score))
            if score[k] > best[0]: best = (float(score[k]), float(centers[k]), float(slope))
        lines.append(best)
    for lead in leads:
        _, center, slope = lines[lead['row']]
        lead['baseline_y_px'] = center + slope*((sum(lead['trace_x'])/2)-ink.shape[1]/2)
        lead['baseline_slope'] = slope
    # Text has several vertical runs per column in a compact band. Preserve
    # tall ink strokes crossing that band so a neighbouring QRS is not erased.
    text_h = max(9, int(step*.1) | 1); text_w = max(15, int(step*.15) | 1)
    starts = ink & ~np.vstack((np.zeros((1,ink.shape[1]),bool), ink[:-1]))
    runs = cv2.boxFilter(starts.astype(np.float32), -1, (text_w,text_h), normalize=False)
    density = cv2.boxFilter(ink.astype(np.float32), -1, (text_w,text_h), normalize=True)
    yy, xx = np.indices(ink.shape)
    # Recover thin, dark strokes which horizontal smoothing diluted into pink
    # paper. Require raw neutral darkness close to a measured trace baseline.
    raw = context.image.astype(np.float32)
    dark_neutral = (raw.max(2) < 130) & ((raw.max(2)-raw.min(2)) < 30) & ~frame
    corridor = np.zeros_like(ink)
    recovered = np.zeros_like(ink)
    for _, center, slope in lines:
        band = np.abs(yy-(center+slope*(xx-ink.shape[1]/2))) < step*.06
        # Well-observed rows keep their approved evidence unchanged.
        if (ink & band).any(0).mean() < 2/3:
            # Find the continuous dark baseline in a broad band. The path is
            # only a pixel-selection guide: missing ink is never synthesized.
            offsets = np.arange(-int(step*.16), int(step*.16)+1)
            columns = np.arange(ink.shape[1])
            rows = np.clip(np.rint(center+slope*(columns-ink.shape[1]/2))[None,:].astype(int)+offsets[:,None],0,ink.shape[0]-1)
            colors = raw[rows,columns]
            cost = colors.max(2) + 2*(colors.max(2)-colors.min(2))
            cost = np.minimum(cost,220)
            previous = cost[:,0].copy()
            parents = np.zeros(cost.shape,np.int16)
            for col in range(1,cost.shape[1]):
                choices = np.full((7,len(offsets)),np.inf)
                for j,shift in enumerate(range(-3,4)):
                    index = np.arange(len(offsets))+shift
                    good = (index>=0)&(index<len(offsets))
                    choices[j,good] = previous[index[good]] + abs(shift)*12
                best = choices.argmin(0)
                parents[:,col] = np.arange(len(offsets))+best-3
                previous = cost[:,col]+choices[best,np.arange(len(offsets))]
            indices = np.zeros(cost.shape[1],int);indices[-1]=previous.argmin()
            for col in range(cost.shape[1]-1,0,-1): indices[col-1]=parents[indices[col],col]
            path = rows[indices,columns]
            tube = np.abs(yy-path[None,:])<=2
            corridor |= band
            recovered |= dark_neutral & tube
            # Remove short grid branches beside the baseline, while retaining
            # observed tall strokes through it (QRS excursions).
            vertical = cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,1),np.uint8))
            vertical = cv2.morphologyEx(vertical,cv2.MORPH_OPEN,np.ones((max(9,int(step*.2)),1),np.uint8))
            tall_support = cv2.dilate(vertical,np.ones((1,3),np.uint8)).astype(bool)
            ink &= ~band | tube | tall_support
    ink |= recovered
    below = np.zeros_like(ink)
    for _, center, slope in lines:
        distance = yy - (center+slope*(xx-ink.shape[1]/2))
        below |= (distance > text_h) & (distance < step*.32)
    glyph_centers = (runs > text_w*1.25) & (density > .18) & below
    text_mask = cv2.dilate(glyph_centers.astype(np.uint8), np.ones((max(3,text_h//2),max(3,text_w//2)),np.uint8)) > 0
    tall = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9,1),np.uint8))
    tall = cv2.morphologyEx(tall, cv2.MORPH_OPEN, np.ones((text_h*2+1,1),np.uint8))
    text_mask &= ~cv2.dilate(tall,np.ones((1,3),np.uint8)).astype(bool)
    # Repeated compact labels anchor column transitions, independently of identity.
    label_support = np.zeros(ink.shape[1], np.float32)
    for row, (_, center, slope) in enumerate(lines):
        if sum(l['row'] == row for l in leads) <= 1: continue
        distances = yy - (center+slope*(xx-ink.shape[1]/2))
        label_support += (glyph_centers & (distances > text_h) & (distances < step*.32)).sum(0)
    label_support = cv2.GaussianBlur(label_support[None,:], (text_w|1,1), 0).ravel()
    cols = int(context.config['columns'])
    transition_fit = None
    if cols > 1 and label_support.max() > 0:
        left = min(l['box'][0] for l in leads); right = max(l['box'][2] for l in leads)
        nominal = (right-left)/cols
        best = (-1., None)
        for pitch in np.arange(nominal*.75, nominal*1.10, 2):
            for first in np.arange(left+nominal*.7, left+nominal*1.15, 2):
                points = np.rint(first+pitch*np.arange(cols-1)).astype(int)
                if points[-1] >= right: continue
                values = label_support[points]
                score = float(values.mean()+values.min())
                if score > best[0]: best = (score, points)
        if best[1] is not None:
            transition_fit = []
            for anchor in best[1]:
                candidates = np.arange(max(left,anchor-text_w), min(right,anchor+4))
                row_scores = []
                for row, (_, center, slope) in enumerate(lines):
                    if sum(l['row']==row for l in leads) != cols: continue
                    base_y = np.rint(center+slope*(candidates-ink.shape[1]/2)).astype(int)
                    values = np.zeros(len(candidates))
                    for dy in range(-int(step*.1),int(step*.12)):
                        values += ink[np.clip(base_y+dy,0,ink.shape[0]-1),candidates]
                    row_scores.append(values)
                common = np.min(row_scores,axis=0)
                transition_fit.append(int(candidates[np.argmax(common)]))
            boundaries = [left] + transition_fit + [right]
            for lead in leads:
                if sum(l['row']==lead['row'] for l in leads) != cols: continue
                col = lead['column']
                lead['box'][0], lead['box'][2] = boundaries[col], boundaries[col+1]
    # Compact print just below a column start is a label, not a waveform.
    # Use measured layout and component geometry, never a lead name or image ID.
    _, glyph_labels, glyph_stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    for k, (gx, gy, gw, gh, ga) in enumerate(glyph_stats[1:], 1):
        if not (text_h*.4 < gh < text_h*1.6 and gh*1.2 < gw < text_w*2):
            continue
        for lead in leads:
            _, center, slope = lines[lead['row']]
            distance = gy - (center+slope*(gx+gw/2-ink.shape[1]/2))
            if abs(gx-lead['box'][0]) < text_w and text_h*.65 < distance < step*.32:
                text_mask[glyph_labels == k] = True
                break
    ink &= ~text_mask
    traces = []
    for lead in leads:
        x0, _, x1, _ = lead['box']
        if lead['column'] == 0: x0 = lead['trace_x'][0]
        lead['trace_x'] = [x0, x1]
        _, center, slope = lines[lead['row']]
        lead['baseline_y_px'] = center+slope*((x0+x1)/2-ink.shape[1]/2)
        spacing = lead['row_spacing_px']
        y0 = max(0, int(lead['baseline_y_px'] - spacing * 1.65))
        y1 = min(ink.shape[0], int(lead['baseline_y_px'] + spacing * .8))
        base = lead['baseline_y_px'] - y0
        crop = ink[y0:y1, x0:x1]
        # Connectivity is used only for attribution; dilated pixels are never exported.
        graph = cv2.dilate(crop.astype(np.uint8), np.ones((13, 5), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(graph, 8)
        selected = np.zeros_like(crop)
        for index, (x, y, width, height, area) in enumerate(stats[1:], 1):
            if width < 2:
                continue
            ys, xs = np.nonzero((labels == index) & crop)
            if not len(xs):
                continue
            near = np.abs(ys - (base + lead["baseline_slope"] * (xs - crop.shape[1]/2))) < spacing * .18
            if near.sum() < 4 or (width < max(22, spacing * .12) and height < spacing * .3):
                continue
            selected[ys, xs] = True
        _, fragments, fragment_stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
        retain = (fragment_stats[:,4] >= 25) | (fragment_stats[:,3] >= spacing*.12) | (fragment_stats[:,2] >= spacing*.08)
        retain[0] = False
        for k, (fx,fy,fw,fh,fa) in enumerate(fragment_stats[1:],1):
            local_base = base+lead['baseline_slope']*(fx+fw/2-crop.shape[1]/2)
            if fh < spacing*.065 and fw < spacing*.08 and abs(fy+fh/2-local_base) > spacing*.07:
                retain[k] = False
        # Retain observed fragments aligned along a tall narrow stroke. Closing
        # supplies connectivity only; no closed/interpolated pixel is exported.
        vertical = cv2.morphologyEx(crop.astype(np.uint8), cv2.MORPH_CLOSE,
                                   np.ones((max(3,int(spacing*.10)), 1),np.uint8))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN,
                                   np.ones((max(5,int(spacing*.35)), 1),np.uint8))
        aligned = cv2.dilate(vertical, np.ones((3,3),np.uint8)).astype(bool)
        selected &= retain[fragments] | aligned
        # Weak narrow fragments away from the baseline can be neutral-looking
        # grid/moire. Compare their absorption to this row's observed ink.
        local_strength = strength[y0:y1, x0:x1]
        sy, sx = np.indices(crop.shape)
        baseline_distance = np.abs(sy-(base+lead['baseline_slope']*(sx-crop.shape[1]/2)))
        baseline_values = local_strength[selected & (baseline_distance < spacing*.04)]
        reference_strength = float(np.median(baseline_values)) if baseline_values.size else 0.
        _, weak_labels, weak_stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8),8)
        for k,(fx,fy,fw,fh,fa) in enumerate(weak_stats[1:],1):
            if fw >= spacing*.04: continue
            fragment = weak_labels[fy:fy+fh,fx:fx+fw] == k
            distances = baseline_distance[fy:fy+fh,fx:fx+fw][fragment]
            values = local_strength[fy:fy+fh,fx:fx+fw][fragment]
            if np.median(distances) > spacing*.07 and np.percentile(values,90) < reference_strength*.75:
                selected[fy:fy+fh,fx:fx+fw][fragment] = False
        # These observed baseline strokes have already passed raw color,
        # connectivity and speck rejection. Do not apply the excursion-sized
        # fragment thresholds to their one-pixel-wide horizontal portions.
        selected |= recovered[y0:y1,x0:x1] & ~text_mask[y0:y1,x0:x1] & (baseline_distance < step*.18)
        observed = selected.any(0)
        width = selected.shape[1]
        low = np.full(width, np.nan, np.float32)
        high = low.copy(); center = low.copy()
        ambiguous = np.zeros(width, bool)
        for x in np.flatnonzero(observed):
            ys = np.flatnonzero(selected[:, x])
            low[x], high[x] = ys[0] + y0, ys[-1] + y0
            center[x] = np.median(ys) + y0
            ambiguous[x] = bool(np.any(np.diff(ys) > 2))
        valid = observed & ~ambiguous
        traces.append(dict(mask=selected, mask_origin=[x0, y0],
                           x_px=np.arange(x0, x1, dtype=np.float32), y_px=center,
                           y_min_px=low, y_max_px=high, observed=observed,
                           valid=valid, ambiguous=ambiguous,
                           baseline_y_px=float(lead['baseline_y_px']),
                           coverage=float(observed.mean()), unambiguous_fraction=float(valid.mean()),
                           method='local_ink_component_graph', baseline_method='layout_geometric_only'))
    owner = np.full(ink.shape, -1, np.int16)
    claims = np.zeros(ink.shape, np.uint8)
    union = np.zeros_like(ink)
    for trace in traces:
        x,y = trace['mask_origin']; h,w = trace['mask'].shape
        union[y:y+h,x:x+w] |= trace['mask']
        claims[y:y+h,x:x+w] += trace['mask']
    # Attribute whole connected fragments; no midpoint split of a tall stroke.
    _, components, stats, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8),8)
    for k,(cx,cy,cw,ch,area) in enumerate(stats[1:],1):
        ys,xs = np.nonzero(components[cy:cy+ch,cx:cx+cw] == k)
        ys += cy; xs += cx
        scores = []
        for i,trace in enumerate(traces):
            x,y = trace['mask_origin']; h,w = trace['mask'].shape
            inside = (xs>=x)&(xs<x+w)&(ys>=y)&(ys<y+h)
            if not inside.any(): continue
            ids = np.flatnonzero(inside)
            ids = ids[trace['mask'][ys[ids]-y,xs[ids]-x]]
            if not len(ids): continue
            baseline = trace['baseline_y_px']+leads[i]['baseline_slope']*(xs[ids]-x-w/2)
            near = np.abs(ys[ids]-baseline)<step*.12
            support = len(np.unique(xs[ids][near]))
            score = support + .01*len(ids)
            scores.append((score,i,ids))
        if not scores: continue
        scores.sort(key=lambda v:v[0],reverse=True)
        _,i,ids = scores[0]
        owner[ys[ids],xs[ids]]=i
        # Preserve observed pixels outside the winning lead's column/window;
        # their attribution remains provisional and explicitly uncertain.
        for _, other, remaining in scores[1:]:
            remaining = remaining[owner[ys[remaining],xs[remaining]] < 0]
            owner[ys[remaining],xs[remaining]] = other
        if len(scores)>1:
            # A connected crossing is unresolved, even when a provisional owner
            # is needed to display it. It must never become a valid sample.
            claims[ys,xs]=np.maximum(claims[ys,xs],2)
    # Recover short red-grid crossings after layout and attribution. Joining
    # fragments before ownership can incorrectly join neighbouring leads.
    # Closing only identifies bracketed gaps; export observed dark pixels only.
    raw = context.image.astype(np.float32)
    scale = max(9, round(min(ink.shape) * .016) | 1)
    paper = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale)))
    raw_neutral = np.maximum(0, paper - raw).min(axis=2)
    dark_crossing = (raw.max(axis=2) < 180) & (np.ptp(raw, axis=2) < 80) & (raw_neutral > 20) & ~frame & ~text_mask
    previously_owned = owner >= 0
    for i, trace in enumerate(traces):
        x, y = trace['mask_origin']; h, w = trace['mask'].shape
        seed = (owner[y:y+h,x:x+w] == i).astype(np.uint8)
        bridges = np.zeros((h,w),bool)
        for kernel in (np.ones((1,11),np.uint8), np.ones((11,1),np.uint8),
                       np.eye(9,dtype=np.uint8), np.fliplr(np.eye(9,dtype=np.uint8))):
            bridges |= cv2.morphologyEx(seed,cv2.MORPH_CLOSE,kernel).astype(bool)
        recovered_crossing = bridges & dark_crossing[y:y+h,x:x+w] & ~previously_owned[y:y+h,x:x+w]
        claims[y:y+h,x:x+w] += recovered_crossing
        available = recovered_crossing & (owner[y:y+h,x:x+w] < 0)
        owner[y:y+h,x:x+w][available] = i
        trace['mask'] |= available
    # Follow observed neutral ink through longer gaps and curved strokes.
    # Unlike directional closing, this is not limited to straight short gaps.
    # Bound growth spatially and prohibit saturated grid pixels. Connectivity
    # is evaluated separately for each already attributed lead.
    neutral_stroke = (np.ptp(raw, axis=2) < 50) & (raw_neutral > 20) & ~frame & ~text_mask
    frozen_owner = owner.copy()
    for i, trace in enumerate(traces):
        x, y = trace['mask_origin']; h, w = trace['mask'].shape
        seed = frozen_owner[y:y+h,x:x+w] == i
        if not seed.any():
            continue
        distance = cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_L2, 3)
        local_darkness = raw[y:y+h,x:x+w].max(axis=2)
        # Adapt to observed ink cores so shaded paper/dark grid cannot pass
        # merely because their absolute RGB intensity is low.
        darkness_limit = min(165., float(np.percentile(local_darkness[seed], 25)) + 20.)
        candidates = neutral_stroke[y:y+h,x:x+w] & (local_darkness < darkness_limit) & (distance < leads[i]['row_spacing_px']*.15)
        candidates &= (frozen_owner[y:y+h,x:x+w] < 0) | seed
        count, connected = cv2.connectedComponents((candidates | seed).astype(np.uint8), 8)
        anchored = np.zeros(count, bool)
        anchored[np.unique(connected[seed])] = True
        anchored[0] = False
        recovered_stroke = candidates & anchored[connected] & ~seed
        claims[y:y+h,x:x+w] += recovered_stroke
        available = recovered_stroke & (owner[y:y+h,x:x+w] < 0)
        owner[y:y+h,x:x+w][available] = i
        trace['mask'] |= available
    for i, trace in enumerate(traces):
        x, y = trace['mask_origin']; h, w = trace['mask'].shape
        trace['mask'] &= owner[y:y+h, x:x+w] == i
        selected = trace['mask']
        attribution_uncertain = ((claims[y:y+h,x:x+w] > 1) & selected).any(0)
        observed = selected.any(0)
        low = np.full(w, np.nan, np.float32); high = low.copy(); center = low.copy()
        ambiguous = np.zeros(w, bool)
        for col in np.flatnonzero(observed):
            ys = np.flatnonzero(selected[:, col])
            low[col], high[col], center[col] = ys[0]+y, ys[-1]+y, np.median(ys)+y
            ambiguous[col] = bool(np.any(np.diff(ys)>2))
        ambiguous |= attribution_uncertain
        trace.update(attribution_uncertain=attribution_uncertain, observed=observed, valid=observed & ~ambiguous, ambiguous=ambiguous,
                     y_px=center, y_min_px=low, y_max_px=high, coverage=float(observed.mean()),
                     unambiguous_fraction=float((observed & ~ambiguous).mean()))
    return Extraction(leads, traces, dict(
        color_gate={'block': True, 'reason': 'NEW_INDEPENDENT_METHOD_AWAITING_HUMAN_REVIEW'},
        activation={'automatic': False, 'reason': 'Extraction experiment; appearance gate not validated'},
        candidate_pixels=int(ink.sum()), frame_pixels=int(frame.sum()), text_pixels=int(text_mask.sum()),
        grid_pixels=int((grid > 10).sum()), column_transitions=transition_fit,
        limitations='Component attribution can retain text or omit disconnected excursions. No interpolation.'))

