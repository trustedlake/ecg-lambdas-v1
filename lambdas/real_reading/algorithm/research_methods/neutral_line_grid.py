"""Experimental neutral long-line grid specialist; pixels and layout only.

Ambiguous grid crossings are withheld or flagged, never interpolated. RGB is retained.
This module is deliberately absent from the production registry and router.
"""
import cv2
import numpy as np
from .contracts import MethodAbstention
from .common import trace_evidence

VERSION = 'neutral-line-grid-v12'


def evidence(image, gated=True):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    chroma = np.ptp(image.astype(np.int16), axis=2)
    dark = gray < 150
    if gated and (dark.mean() < .005 or float((chroma[dark] <= 3).mean()) < .98):
        raise MethodAbstention('Not a neutral-ink page')
    smooth = cv2.medianBlur(gray, 3)
    binary = (smooth < 150).astype(np.uint8)
    h, w = gray.shape
    length = max(100, int(min(h, w) * .25))
    lines = cv2.HoughLinesP(binary, 1, np.pi/1800, threshold=length//2,
                            minLineLength=length, maxLineGap=12)
    excluded = np.zeros_like(binary)
    thick_stroke = np.zeros_like(binary)
    counts = [0, 0]
    families = [[], []]
    if lines is not None:
        for x0,y0,x1,y1 in lines.reshape(-1, 4):
            dx,dy = abs(int(x1-x0)),abs(int(y1-y0))
            if min(dx,dy) > .06*max(dx,dy):
                continue
            counts[int(dy > dx)] += 1
            if dx > dy:
                slope=float(y1-y0)/float(x1-x0)
                families[0].append((float(y0)+slope*(w/2-x0),slope))
            else:
                slope=float(x1-x0)/float(y1-y0)
                families[1].append((float(x0)+slope*(h/2-y0),slope))
    for axis, family in enumerate(families):
        groups=[]
        for intercept,slope in sorted(family):
            if not groups or intercept-np.median([p[0] for p in groups[-1]])>5:
                groups.append([])
            groups[-1].append((intercept,slope))
        for group in groups:
            center,slope=np.median(group,axis=0)
            if axis==0:
                p0=(0,round(center-slope*w/2));p1=(w-1,round(center+slope*w/2))
            else:
                p0=(round(center-slope*h/2),0);p1=(round(center+slope*h/2),h-1)
            cv2.line(excluded,p0,p1,1,3 if axis else 5)
            # Local thickening over a measured long grid line is independent
            # evidence of a superimposed stroke. Short grid intersections do
            # not qualify; original pixels only, with uncertainty retained.
            if axis==0:continue
            along=np.arange(h)
            across=np.rint(center+slope*(along-len(along)/2)).astype(int)
            strips=[]
            for offset in range(-5,6):
                transverse=np.clip(across+offset,0,(h if axis==0 else w)-1)
                strips.append(gray[transverse,along] if axis==0 else gray[along,transverse])
            density=np.maximum(0,150-np.asarray(strips,dtype=np.float32)).sum(0)/150
            excess=density>float(np.percentile(density,40))+.9
            groups_along=np.split(np.flatnonzero(excess),np.flatnonzero(np.diff(np.flatnonzero(excess))>3)+1)
            for run in groups_along:
                if len(run)<7 or run[-1]-run[0] > .2*len(along):continue
                positions=np.arange(run[0],run[-1]+1)
                for offset in range(-3,4):
                    transverse=np.clip(across[positions]+offset,0,(h if axis==0 else w)-1)
                    if axis==0:thick_stroke[transverse,positions]=1
                    else:thick_stroke[positions,transverse]=1
    if gated and min(counts) < 8:
        raise MethodAbstention('No dense long neutral grid in both axes')
    envelope = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((21,21),np.uint8))
    contrast = np.maximum(0, envelope.astype(np.float32)-gray)
    raw = ((gray < 150) & (contrast > 35) & ((excluded == 0)|(thick_stroke>0))).astype(np.uint8)
    # Export only dark source pixels, never grouping/dilation pixels.
    raw[gray >= 170] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = stats[:,cv2.CC_STAT_AREA] >= 4
    keep[0] = False
    return keep[labels].astype(np.uint8), contrast, excluded, counts


def extract(context, gated=True):
    raw, contrast, excluded, counts = evidence(context.image, gated)
    result = trace_evidence(context, raw, contrast)
    # Group across the withheld grid width, but export only measured pixels.
    # This keeps both sides of steep strokes instead of a single DP branch.
    for lead, trace in zip(result.leads, result.traces):
        x0, x1 = lead['trace_x']
        top, bottom = lead['box'][1], lead['box'][3]
        measured = raw[top:bottom, x0:x1]
        grouped = cv2.dilate(measured, np.ones((13,13),np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped,8)
        tx, ty = trace['mask_origin']
        anchors = np.zeros(raw.shape, np.uint8)
        th, tw = trace['mask'].shape
        anchors[ty:ty+th, tx:tx+tw] = trace['mask']
        local=anchors[top:bottom,x0:x1]>0
        yy,xx=np.nonzero(local)
        fit=np.array([0.,lead['baseline_y_px']-top])
        if len(xx)>=20:
            selected_fit=np.abs(yy-(lead['baseline_y_px']-top)) < .22*lead['row_spacing_px']
            for _ in range(4):
                if selected_fit.sum()<20: break
                fit=np.polyfit(xx[selected_fit],yy[selected_fit],1)
                residual=np.abs(yy-np.polyval(fit,xx))
                selected_fit=residual<=max(4,float(np.percentile(residual,55)))
        near=np.abs(np.arange(bottom-top)[:,None]-np.polyval(fit,np.arange(x1-x0))[None,:]) <= max(6,.06*lead['row_spacing_px'])
        touched = np.unique(labels[(measured>0) & near])
        keep = np.zeros(n, bool)
        keep[touched] = True
        keep[0] = False
        selected = keep[labels] & (measured > 0)
        # Remove detached specks and long near-horizontal branches far from
        # the baseline. This does not select a lead by name or identity.
        n0, components, stats0, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8),8)
        retain = stats0[:,cv2.CC_STAT_AREA]>=10
        retain[0]=False
        selected &= retain[components]
        line_length=max(25,round((x1-x0)*.08))
        lines=cv2.HoughLinesP(selected.astype(np.uint8),1,np.pi/720,
                            threshold=line_length,minLineLength=line_length,maxLineGap=3)
        removed=np.zeros(selected.shape,np.uint8)
        if lines is not None:
            for ax,ay,bx,by in lines.reshape(-1,4):
                if abs(int(by-ay)) > .08*max(1,abs(int(bx-ax))):continue
                middle=(float(ax)+float(bx))/2
                if abs((float(ay)+float(by))/2-np.polyval(fit,middle)) <= .14*lead['row_spacing_px']:continue
                cv2.line(removed,(int(ax),int(ay)),(int(bx),int(by)),1,3)
        selected &= removed==0
        observed = selected.any(0)
        width = x1-x0
        y = np.full(width,np.nan,np.float32); low=y.copy(); high=y.copy()
        ambiguous = np.zeros(width,bool)
        for x in np.flatnonzero(observed):
            points=np.flatnonzero(selected[:,x])+top
            y[x]=np.median(points); low[x]=points[0]; high[x]=points[-1]
            ambiguous[x]=bool(np.any(np.diff(points)>3) or np.ptp(points)>max(8,.12*lead['row_spacing_px']))
        ambiguous |= (selected & (excluded[top:bottom,x0:x1]>0)).any(0)
        valid=observed & ~ambiguous
        trace.update(mask=selected,mask_origin=[x0,top],x_px=np.arange(x0,x1,dtype=np.float32),
                     y_px=y,y_min_px=low,y_max_px=high,observed=observed,
                     valid=valid,ambiguous=ambiguous,coverage=float(observed.mean()),
                     unambiguous_fraction=float(valid.mean()))
    result.diagnostics['ownership']={'method':'disjoint_explicit_layout_boxes',
                                    'shared_pixels_after':result.validate(context.image.shape)['shared_pixels']}
    result.diagnostics.update(long_lines_by_axis=counts,
        excluded_grid_pixels=int(excluded.sum()),
        limitation='Measured thick strokes on vertical grid retained but ambiguous; other intersections withheld; experimental.')
    return result
