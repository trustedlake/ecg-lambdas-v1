"""Recover measured narrow strokes attached to a trace, with joint ownership."""
import cv2
import numpy as np
from .signal_tinted import stroke_evidence
from .signal_neutral import summarize


def recover(image, leads, traces):
    measured, _ = stroke_evidence(image)
    # Detect components on the whole page: cropping must not turn a sustained
    # grid line into a short admissible peak.
    support = cv2.morphologyEx(measured.astype('uint8'), cv2.MORPH_OPEN,
                               np.ones((7, 1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    occupied = np.zeros(measured.shape, np.uint16)
    claims = np.zeros(measured.shape, np.uint16)
    candidates = []
    for t in traces:
        x, y = t['mask_origin']; h, w = t['mask'].shape
        occupied[y:y+h, x:x+w] += t['mask']
    for lead, t in zip(leads, traces):
        x, y = t['mask_origin']; h, w = t['mask'].shape
        local_labels = labels[y:y+h, x:x+w]
        touching = cv2.dilate(t['mask'].astype('uint8'), np.ones((3, 3), np.uint8)) > 0
        keep = np.zeros(n, bool)
        step = lead['row_spacing_px']
        for k in np.unique(local_labels[touching]):
            if not k: continue
            xx, yy, ww, hh, area = stats[k]
            keep[k] = (hh >= max(7, 1.2*ww) and hh <= .65*step
                       and ww <= .2*step and yy >= y and yy+hh <= y+h)
        candidate = cv2.dilate(keep[local_labels].astype('uint8'),
                               np.ones((3, 3), np.uint8)) > 0
        candidate &= measured[y:y+h, x:x+w]
        candidate &= occupied[y:y+h, x:x+w] == 0
        candidates.append(candidate)
        claims[y:y+h, x:x+w] += candidate
    report = []
    for t, candidate in zip(traces, candidates):
        x, y = t['mask_origin']; h, w = t['mask'].shape
        candidate &= claims[y:y+h, x:x+w] == 1
        added = int(candidate.sum())
        if added:
            t['mask'] |= candidate
            summarize(t, candidate.any(0))
            t['method'] = 'contrast_continuity_with_measured_narrow_strokes'
        report.append({'added_pixels': added})
    return report


def suppress_persistent_grid(image, traces):
    """Remove sustained vertical background, retaining darker crossing ink.

    Full-page support prevents a cropped grid segment masquerading as a peak.
    A bright quantile estimates background without treating aligned dark peaks
    as grid. No names or row IDs enter this decision.
    """
    red=image[...,0].astype(np.float32)
    narrow=cv2.blur(red,(31,1))-red
    length=max(31,int(image.shape[0]*.15)//2*2+1)
    from scipy.ndimage import percentile_filter, minimum_filter1d
    support=cv2.dilate((narrow>3).astype('uint8'),np.ones((1,5),np.uint8))
    persistence=max(length,int(image.shape[0]*.45)//2*2+1)
    sustained=cv2.boxFilter(support.astype(np.float32),-1,(1,persistence))>.8
    # Distinct chromaticity is trace evidence even when red brightness matches.
    f=image.astype(np.float32)
    # Sample only for the slowly varying background estimate, never the mask.
    sampled=np.stack([minimum_filter1d(f[...,k],size=5,axis=1)[::4]
                      for k in range(3)],axis=2)
    local_small=percentile_filter(sampled,percentile=50,size=(max(3,persistence//4|1),1,1),mode='nearest')
    local=cv2.resize(local_small,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
    background=local[...,0]
    chroma=f-f.mean(2,keepdims=True)
    grid_chroma=local-local.mean(2,keepdims=True)
    same_color=np.linalg.norm(chroma-grid_chroma,axis=2)<5
    distinct=(red<background-8) | (np.linalg.norm(chroma-grid_chroma,axis=2)>12)
    # Strong horizontal contrast protects dense aligned peaks even when
    # their repeated dark cores contaminate the column background estimate.
    distinct |= narrow > 80
    protected=cv2.dilate(distinct.astype('uint8'),np.ones((5,15),np.uint8))>0
    excluded=sustained & (red>=background-2) & same_color & ~protected
    report=[]
    for t in traces:
        x,y=t['mask_origin'];h,w=t['mask'].shape
        removed=t['mask'] & excluded[y:y+h,x:x+w]
        if removed.any():
            t['mask'] &= ~removed
            summarize(t,removed.any(0))
        report.append(int(removed.sum()))
    return report
