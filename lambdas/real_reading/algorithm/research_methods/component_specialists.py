"""Experimental specialists: measured components, with exclusive row ownership.

No identity, votes or reference signal enters extraction. Connectivity may bridge
small gaps for grouping only; output masks always contain measured pixels only.
"""
import cv2
import numpy as np
from .contracts import Extraction, MethodAbstention
from .frozen_peaks34.ecg.signal_tinted import locate_leads
from .frozen_peaks34.ecg.trace_ownership import _distances


def evidence(image, monochrome=False):
    smooth=cv2.GaussianBlur(image,(3,3),.6)
    envelope=cv2.morphologyEx(smooth,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    rgb=np.minimum(255,255*smooth.astype(np.float32)/np.maximum(envelope,1))
    # Red illumination retains black/purple ink and discounts red paper grids.
    channel = (cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) if monochrome else
               np.clip(rgb.max(2)+.7*np.ptp(rgb,axis=2),0,255).astype(np.uint8))
    background = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, np.ones((31,31), np.uint8))
    contrast = np.maximum(0, background.astype(np.float32)-channel)
    if monochrome:
        # Long straight grid support in several orientations; never inpaint it.
        length = max(61, int(min(image.shape[:2])*.13)) | 1
        grid = np.zeros_like(contrast)
        for vertical in (False, True):
            for slope in (-.12, -.06, 0, .06, .12):
                k = np.zeros((length,length), np.uint8)
                a = int(length//2 - slope*length/2)
                b = int(length//2 + slope*length/2)
                cv2.line(k, (a,0) if vertical else (0,a),
                         (b,length-1) if vertical else (length-1,b), 1)
                grid = np.maximum(grid, cv2.morphologyEx(contrast,cv2.MORPH_OPEN,k))
        contrast = np.maximum(0, contrast-grid)
    else:
        # A pixel dominated by red is grid-like; luminance alone cannot decide.
        red_excess = np.maximum(0, rgb[...,0]-np.maximum(rgb[...,1],rgb[...,2]))
        contrast = np.maximum(0, contrast-1.5*red_excess)
    median=float(np.median(contrast));mad=float(np.median(abs(contrast-median)))
    threshold=max(12.,median+4*max(2.,mad))
    mask = (contrast >= threshold).astype(np.uint8)
    original_rgb=np.minimum(255,255*image.astype(np.float32)/np.maximum(envelope,1))
    original_channel=(cv2.cvtColor(original_rgb.astype(np.uint8),cv2.COLOR_RGB2GRAY) if monochrome else
                      np.clip(original_rgb.max(2)+.7*np.ptp(original_rgb,axis=2),0,255))
    # Smoothing may spread ink into white pixels: never export those pixels.
    mask[(255-original_channel)<threshold]=0
    if monochrome:
        edges=cv2.Canny(channel,40,110)
        length=max(80,int(min(image.shape[:2])*.30))
        lines=cv2.HoughLinesP(edges,1,np.pi/720,threshold=max(35,length//3),minLineLength=length,maxLineGap=12)
        excluded=np.zeros_like(mask)
        if lines is not None:
            for x0,y0,x1,y1 in lines.reshape(-1,4):
                dx=abs(int(x1)-int(x0));dy=abs(int(y1)-int(y0))
                if min(dx,dy)<=.18*max(dx,dy):
                    cv2.line(excluded,(int(x0),int(y0)),(int(x1),int(y1)),1,5)
        mask[excluded>0]=0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 4
    keep[0] = False
    return keep[labels].astype(np.uint8), contrast


def component_trace(context, monochrome=False):
    raw, contrast = evidence(context.image, monochrome)
    if monochrome:
        from .frozen_peaks34.ecg.signal_dark import ink_evidence
        try:
            measured,_=ink_evidence(context.image,context.grid)
        except ValueError as error:
            raise MethodAbstention(str(error)) from error
        raw &= measured.astype(np.uint8)
    if raw.mean() < .0001:
        raise MethodAbstention('Insufficient measured ink after grid separation')
    leads = locate_leads(raw, context.config, context.geometry)
    h,w = raw.shape
    # Remove page-length straight structures before following connected strokes.
    vertical=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((max(31,int(.40*h)),1),np.uint8))
    horizontal=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,max(31,int(.40*w))),np.uint8))
    interior_top=max(0,int(min(l['baseline_y_px']-.3*l['row_spacing_px'] for l in leads)))
    interior_bottom=min(h,int(max(l['baseline_y_px']+.3*l['row_spacing_px'] for l in leads))+1)
    horizontal[interior_top:interior_bottom]=0
    raw[cv2.dilate(vertical|horizontal,np.ones((3,3),np.uint8))>0]=0
    # Connectivity only; no closing pixel is exported.
    grouped = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((7,1),np.uint8))
    if monochrome:
        grouped=cv2.morphologyEx(grouped,cv2.MORPH_CLOSE,np.ones((1,7),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped,8)
    rows = sorted({lead['row'] for lead in leads})
    owner=np.full((h,w),-1,np.int16)
    best=np.full((h,w),np.inf,np.float32);second=best.copy()
    for row in rows:
        row_leads=[lead for lead in leads if lead['row']==row]
        step=row_leads[0]['row_spacing_px']
        centers=[lead['baseline_y_px'] for lead in row_leads]
        y0=max(0,int(min(centers)-1.4*step));y1=min(h,int(max(centers)+1.4*step))
        anchors=np.zeros((y1-y0,w),bool)
        for lead in row_leads:
            x0,x1=lead['trace_x'];base=lead['baseline_y_px']
            top=max(y0,int(base-step*.10));bottom=min(y1,int(base+step*.10)+1)
            horizontal=cv2.morphologyEx(raw[top:bottom,x0:x1],cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
            anchors[top-y0:bottom-y0,x0:x1]=horizontal>0
        distance=_distances(grouped[y0:y1].astype(bool),anchors)
        current=best[y0:y1];runner=second[y0:y1];owners=owner[y0:y1]
        better=distance<current
        runner[:]=np.where(better,current,np.minimum(runner,distance))
        owners[better]=row
        current[:]=np.minimum(current,distance)
    unique=np.isfinite(best)&(second>best+3)
    owner[~unique]=-1
    traces=[]
    for i,lead in enumerate(leads):
        x0,x1=lead['trace_x']; base=lead['baseline_y_px']; step=lead['row_spacing_px']
        selected=(owner[:,x0:x1]==lead['row']) & (raw[:,x0:x1]>0)
        width=x1-x0
        observed=selected.any(0); y=np.full(width,np.nan,np.float32); low=y.copy(); high=y.copy()
        ambiguous=np.zeros(width,bool)
        for x in np.flatnonzero(observed):
            points=np.flatnonzero(selected[:,x]); low[x]=points[0]; high[x]=points[-1]
            y[x]=np.median(points)
            # Multiple branches or an almost vertical stroke do not determine
            # one reliable amplitude at this pixel resolution.
            ambiguous[x]=bool(np.any(np.diff(points)>3) or np.ptp(points)>max(8,.12*step))
        valid=observed & ~ambiguous
        traces.append({'x_px':np.arange(x0,x1,dtype=np.float32),'y_px':y,
                       'y_min_px':low,'y_max_px':high,'observed':observed,
                       'valid':valid,'ambiguous':ambiguous,'mask':selected,
                       'mask_origin':[x0,0], 'baseline_y_px':float(np.nanmedian(y)) if observed.any() else float(base),
                       'coverage':float(observed.mean()),'unambiguous_fraction':float(valid.mean()),
                       'method':'exclusive_connected_components',
                       'baseline_method':'median_observed_not_clinical_reference'})
    return Extraction(leads,traces,{'components':n-1,'unassigned_measured_pixels':int(((raw>0)&(owner<0)).sum()),
                       'evidence':'achromatic_long_line_subtraction' if monochrome else 'red_channel_local_contrast',
                       'limitation':'Crossing components withheld; vertical envelopes uncertain; no inpainting.'})
